"""Self-supervised pre-training scaffold — TCP objective (Section 11.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from train.feature_schema import (
    MARKET_FEATURE_DIM,
    PER_ASSET_FEATURE_DIM,
    RETURN_FEATURE_INDEX,
)

TOTAL_FEATURE_DIM = PER_ASSET_FEATURE_DIM + MARKET_FEATURE_DIM
SYNTHETIC_SHARD_PREFIX = "FDA_SYNTH_"
UNKNOWN_SECTOR_ID = -1

__all__ = [
    "MARKET_FEATURE_DIM",
    "PER_ASSET_FEATURE_DIM",
    "RETURN_FEATURE_INDEX",
    "SYNTHETIC_SHARD_PREFIX",
    "TOTAL_FEATURE_DIM",
    "UNKNOWN_SECTOR_ID",
]


# ── Contrastive projection ───────────────────────────────────────────────────


class ProjectionHead(nn.Module):
    """Two-layer projector used by temporal contrastive pretraining."""

    def __init__(self, d_model: int = 768, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ReturnPredictionHead(nn.Module):
    """Auxiliary next-step prediction head — spec §7.9.

    Outputs (return, log_vol) at t+1. Phase-1 (§11.3) supervises only the
    return output via MSE; log_vol is unused during pretraining and consumed
    by the surprise signal at inference.

    Zero-initialised so early predictions start near zero, avoiding the
    ~3-order-of-magnitude MSE blow-up that occurs with default Linear init
    against O(1e-3) log-return targets.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[..., 0], out[..., 1]


def infer_regime_label(feature_seq: np.ndarray) -> int:
    """Heuristic 4-state regime label for synthetic shards lacking HMM labels.

    Proxies: VIX/100 at market offset 184 (volatility) and log-return at index 3.
    Thresholds calibrated against 10yr SPY data to target the §8.4 regime
    distribution (~55/20/15/10 across bull/hi-vol/mean-rev/crisis).

    Canonical IDs (spec §11.3):
      0 = low-vol bull  1 = high-vol trending  2 = mean-reverting  3 = crisis
    """
    vol_proxy = float(feature_seq[:, 184].mean())
    ret_proxy = float(feature_seq[:, 3].mean())

    if vol_proxy > 0.28:
        return 3  # crisis — sustained elevated VIX (top ~10%)
    if vol_proxy > 0.19:
        return 1  # high-vol trending — VIX elevated but not crisis
    if ret_proxy >= 0.0:
        return 0  # low-vol bull
    return 2  # mean-reverting


# ── HMM regime labeling (§11.2 Phase 0, §11.3) ──────────────────────────────


def _shard_returns(path: Path) -> np.ndarray:
    """Load [return, |return|] as 2-D HMM features.

    Adding |return| gives EM a clean volatility axis; 1-D return-only collapses
    high-vol-bull/bear into ambiguous clusters and prevents 4-regime separation.
    """
    raw = np.memmap(path, dtype="float32", mode="r")
    n_rows = raw.size // TOTAL_FEATURE_DIM
    data = raw.reshape(n_rows, TOTAL_FEATURE_DIM)
    returns = data[:, RETURN_FEATURE_INDEX].astype(np.float64)
    return np.column_stack([returns, np.abs(returns)])


def _canonicalize_hmm(
    hmm: object, raw_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw HMM state integers to spec canonical IDs (§11.3).

    Ordering: highest mean return → 0 (bull), lowest → 3 (crisis);
    of the two middle states, higher mean |return| → 1 (hi-vol), lower → 2 (mean-rev).
    Returns (remap, canonical_labels) where remap[raw_state] = canonical_id.
    """
    means = hmm.means_  # type: ignore[attr-defined]  shape (n_states, 2)
    mean_ret = means[:, 0]  # mean log-return per state
    mean_abs = means[:, 1]  # mean |return| — volatility proxy
    n = len(mean_ret)
    mean_order = np.argsort(-mean_ret)  # descending: highest mean → index 0

    remap = np.empty(n, dtype=np.int64)
    remap[mean_order[0]] = 0   # highest mean return → bull
    remap[mean_order[-1]] = 3  # lowest mean return  → crisis
    mid = sorted(mean_order[1:-1].tolist(), key=lambda s: -mean_abs[s])
    remap[mid[0]] = 1  # higher |return| → high-vol trending
    remap[mid[1]] = 2  # lower  |return| → mean-reverting

    return remap, remap[raw_labels]


def build_spy_regime_labels(
    data_dir: Path,
    n_states: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit a 4-state Gaussian HMM on SPY returns and return per-row canonical
    regime labels for train and validation splits (§11.2 / §11.3).

    HMM fit on train split only (no look-ahead); labels cached for fast restarts.
    Returns (train_labels, val_labels) or (None, None) if SPY shard is absent.
    """
    from hmmlearn.hmm import GaussianHMM

    spy_train = data_dir / "train" / "SPY.f32bin"
    spy_val = data_dir / "validation" / "SPY.f32bin"
    cache_train = data_dir / "spy_regime_labels_train.npy"
    cache_val = data_dir / "spy_regime_labels_val.npy"

    if not spy_train.exists():
        return None, None

    # Cache hit: verify the label count still matches the shard row count.
    if cache_train.exists():
        cached = np.load(cache_train)
        raw_check = np.memmap(spy_train, dtype="float32", mode="r")
        expected = raw_check.size // TOTAL_FEATURE_DIM
        if cached.shape[0] == expected:
            val_labels = np.load(cache_val) if cache_val.exists() else None
            return cached, val_labels

    # Fit on training split only (no look-ahead).
    train_returns = _shard_returns(spy_train)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=500,
        random_state=42,
        verbose=False,
    )
    hmm.fit(train_returns)
    remap, train_labels = _canonicalize_hmm(hmm, hmm.predict(train_returns))
    np.save(cache_train, train_labels)

    counts = np.bincount(train_labels, minlength=n_states)
    print(
        f"HMM regime labels fit on {len(train_labels)} SPY rows — "
        f"bull={counts[0]}  hi-vol={counts[1]}  "
        f"mean-rev={counts[2]}  crisis={counts[3]}"
    )

    val_labels: np.ndarray | None = None
    if spy_val.exists():
        val_returns = _shard_returns(spy_val)
        raw_val = hmm.predict(val_returns)
        val_labels = remap[raw_val]
        np.save(cache_val, val_labels)

    return train_labels, val_labels


# ── Data loading ─────────────────────────────────────────────────────────────


class ShardDataset(IterableDataset):
    """Streams TCP training tuples from .f32bin memory-mapped shards.

    Each .f32bin shard contains rows of shape (per_asset_feature_dim + market_dim)
    = 184 + 96 = 280 floats, stored as little-endian float32.

    For each anchor window W_a, the dataset emits:
      - adjacent positive window W_b (offset by `positive_offset` rows),
      - one-step-ahead return target for auxiliary prediction loss,
      - metadata (asset/regime/sector IDs, source weight).
    """

    PER_ASSET_DIM = PER_ASSET_FEATURE_DIM
    FEATURE_DIM = (
        TOTAL_FEATURE_DIM  # 184 per-asset (120 continuous + 64 text) + 96 market-wide
    )
    MARKET_DIM = MARKET_FEATURE_DIM

    def __init__(
        self,
        shard_paths: list[Path],
        seq_len: int = 64,
        shuffle: bool = True,
        positive_offset: int = 4,
        synthetic_prefix: str = SYNTHETIC_SHARD_PREFIX,
        synthetic_loss_weight: float = 0.5,
        symbol_to_sector: dict[str, int] | None = None,
        symbol_to_id: dict[str, int] | None = None,
        global_regime_labels: np.ndarray | None = None,
    ) -> None:
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.positive_offset = positive_offset
        self.synthetic_prefix = synthetic_prefix
        self.synthetic_loss_weight = synthetic_loss_weight
        self.symbol_to_sector = symbol_to_sector or {}
        # When provided, IDs are used directly; otherwise derived from all shard
        # paths sorted by stem — consistent across DataLoader workers.
        self.symbol_to_id = symbol_to_id or {}
        # Per-row HMM regime labels aligned to the SPY shard row order (all
        # real shards share the same trading calendar).  None → heuristic fallback.
        self.global_regime_labels = global_regime_labels
        if self.positive_offset == 0:
            raise ValueError("positive_offset must be non-zero for TCP views")
        if self.synthetic_loss_weight <= 0.0:
            raise ValueError(
                "synthetic_loss_weight must be positive, "
                f"got {self.synthetic_loss_weight}"
            )

    def _load_rows(self, path: Path) -> np.ndarray:
        raw = np.memmap(path, dtype="float32", mode="r")
        n_floats = int(raw.size)
        if n_floats % self.FEATURE_DIM != 0:
            raise ValueError(
                f"shard {path} has unsupported row width "
                f"(floats={n_floats}, expected multiple of {self.FEATURE_DIM})"
            )
        return raw.reshape(-1, self.FEATURE_DIM)

    def _iter_shard(
        self,
        path: Path,
        asset_id: int,
        sector_id: int,
        sample_weight: float,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float, int, int, int, float, int]]:
        data = self._load_rows(path)
        n_rows = data.shape[0]

        # +1 lets exact-length shards (n_rows == seq_len + offset) yield their
        # single pair instead of being silently dropped at n_seqs <= 0.
        min_start = max(0, -self.positive_offset)
        max_start = n_rows - self.seq_len - max(self.positive_offset, 0) + 1
        n_seqs = max_start - min_start
        if n_seqs <= 0:
            return

        indices = np.arange(min_start, max_start)
        if self.shuffle:
            np.random.shuffle(indices)

        is_synthetic = path.stem.startswith(self.synthetic_prefix)
        n_label_rows = (
            len(self.global_regime_labels)
            if self.global_regime_labels is not None
            else 0
        )

        for start in indices:
            start_idx = int(start)
            pos_start = start_idx + self.positive_offset
            seq_a = data[start_idx : start_idx + self.seq_len]  # (T, F)
            seq_b = data[pos_start : pos_start + self.seq_len]  # (T, F)
            next_return = float(data[start_idx + self.seq_len, RETURN_FEATURE_INDEX])
            # Use HMM label at the last step of the anchor window so the
            # regime reflects the current market state at inference time.
            # Synthetic shards have no date-aligned HMM labels → heuristic.
            label_idx = start_idx + self.seq_len - 1
            if (
                not is_synthetic
                and self.global_regime_labels is not None
                and label_idx < n_label_rows
            ):
                regime_id = int(self.global_regime_labels[label_idx])
            else:
                regime_id = infer_regime_label(seq_a)
            yield (
                seq_a,
                seq_b,
                next_return,
                asset_id,
                regime_id,
                sector_id,
                sample_weight,
                start_idx,
            )

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:
        paths = list(self.shard_paths)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]

        if self.shuffle:
            import random

            random.shuffle(paths)
        if self.symbol_to_id:
            asset_to_id = self.symbol_to_id
        else:
            sorted_all = sorted(self.shard_paths, key=lambda x: x.stem)
            asset_to_id = {p.stem: i for i, p in enumerate(sorted_all)}
        iterators = {
            path: iter(
                self._iter_shard(
                    path,
                    asset_to_id[path.stem],
                    self.symbol_to_sector.get(path.stem, UNKNOWN_SECTOR_ID),
                    self.synthetic_loss_weight
                    if path.stem.startswith(self.synthetic_prefix)
                    else 1.0,
                )
            )
            for path in paths
        }
        active = list(paths)
        rng = np.random.default_rng()

        while active:
            if self.shuffle:
                rng.shuffle(active)
            next_active: list[Path] = []
            for path in active:
                iterator = iterators[path]
                try:
                    (
                        seq_a,
                        seq_b,
                        next_return,
                        asset_id,
                        regime_id,
                        sector_id,
                        sample_weight,
                        time_index,
                    ) = next(iterator)
                except StopIteration:
                    continue
                yield (
                    torch.from_numpy(seq_a.copy()),
                    torch.from_numpy(seq_b.copy()),
                    torch.tensor(next_return, dtype=torch.float32),
                    torch.tensor(asset_id, dtype=torch.long),
                    torch.tensor(regime_id, dtype=torch.long),
                    torch.tensor(sector_id, dtype=torch.long),
                    torch.tensor(sample_weight, dtype=torch.float32),
                    torch.tensor(time_index, dtype=torch.long),
                )
                next_active.append(path)
            active = next_active


class TarShardDataset(ShardDataset):
    """Streams TCP training tuples from WebDataset-format tar archives.

    Tars may be local paths or HTTP(S) URLs (e.g. HuggingFace resolve URLs).
    Each tar contains many `FDA_SYNTH_NNNNNNNN.f32bin` entries produced by
    `python.tools.pack_fda_shards`. The tar itself is treated as one logical
    shard for worker-splitting; each entry inside yields a synthetic sample.

    Only the synthetic path is supported here — real-shard .f32bin files
    carry per-asset semantics (asset_id, HMM regime labels) that rely on
    filesystem layout, so they keep using the base `ShardDataset`.
    """

    def __init__(
        self,
        tar_sources: list[str],
        seq_len: int = 64,
        shuffle: bool = True,
        positive_offset: int = 4,
        synthetic_prefix: str = SYNTHETIC_SHARD_PREFIX,
        synthetic_loss_weight: float = 0.5,
    ) -> None:
        # Skip parent __init__ path-validation (no filesystem paths here).
        self.tar_sources = list(tar_sources)
        self.shard_paths = []  # unused, satisfy base-class attr expectations
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.positive_offset = positive_offset
        self.synthetic_prefix = synthetic_prefix
        self.synthetic_loss_weight = synthetic_loss_weight
        self.symbol_to_sector = {}
        self.symbol_to_id = {}
        self.global_regime_labels = None
        if self.positive_offset == 0:
            raise ValueError("positive_offset must be non-zero for TCP views")
        if self.synthetic_loss_weight <= 0.0:
            raise ValueError(
                "synthetic_loss_weight must be positive, "
                f"got {self.synthetic_loss_weight}"
            )
        if not self.tar_sources:
            raise ValueError("tar_sources must be non-empty")

    @staticmethod
    def _open_tar(source: str):
        """Open a tar for streaming read from a local path or HTTP(S) URL."""
        import tarfile

        if source.startswith("http://") or source.startswith("https://"):
            from urllib.request import urlopen

            # Stream mode — no seek. tarfile handles sequential reads fine.
            return tarfile.open(fileobj=urlopen(source), mode="r|")
        return tarfile.open(source, mode="r")

    def _iter_tar(self, source: str) -> Iterator[
        tuple[np.ndarray, np.ndarray, float, int, int, int, float, int]
    ]:
        tf = self._open_tar(source)
        try:
            for member in tf:
                if not member.isfile():
                    continue
                if not member.name.endswith(".f32bin"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                buf = f.read()
                arr = np.frombuffer(buf, dtype=np.float32)
                if arr.size % self.FEATURE_DIM != 0:
                    raise ValueError(
                        f"tar entry {member.name} has unsupported row width "
                        f"(floats={arr.size}, expected multiple of {self.FEATURE_DIM})"
                    )
                data = arr.reshape(-1, self.FEATURE_DIM)
                # Synth entries inherit FDA_SYNTH_ prefix by construction.
                is_synthetic = Path(member.name).stem.startswith(self.synthetic_prefix)
                sample_weight = (
                    self.synthetic_loss_weight if is_synthetic else 1.0
                )
                # +1 lets exact-length shards yield one pair (see _iter_shard).
                n_rows = data.shape[0]
                min_start = max(0, -self.positive_offset)
                max_start = (
                    n_rows - self.seq_len - max(self.positive_offset, 0) + 1
                )
                if max_start <= min_start:
                    continue
                indices = np.arange(min_start, max_start)
                if self.shuffle:
                    np.random.shuffle(indices)
                for start in indices:
                    start_idx = int(start)
                    pos_start = start_idx + self.positive_offset
                    seq_a = data[start_idx : start_idx + self.seq_len]
                    seq_b = data[pos_start : pos_start + self.seq_len]
                    next_return = float(
                        data[start_idx + self.seq_len, RETURN_FEATURE_INDEX]
                    )
                    regime_id = infer_regime_label(seq_a)
                    yield (
                        seq_a,
                        seq_b,
                        next_return,
                        0,  # asset_id — synthetic windows are not asset-specific
                        regime_id,
                        UNKNOWN_SECTOR_ID,
                        sample_weight,
                        start_idx,
                    )
        finally:
            tf.close()

    def __iter__(self):
        sources = list(self.tar_sources)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            sources = sources[worker.id :: worker.num_workers]
        if self.shuffle:
            import random

            random.shuffle(sources)
        for src in sources:
            for (
                seq_a,
                seq_b,
                next_return,
                asset_id,
                regime_id,
                sector_id,
                sample_weight,
                time_index,
            ) in self._iter_tar(src):
                yield (
                    torch.from_numpy(seq_a.copy()),
                    torch.from_numpy(seq_b.copy()),
                    torch.tensor(next_return, dtype=torch.float32),
                    torch.tensor(asset_id, dtype=torch.long),
                    torch.tensor(regime_id, dtype=torch.long),
                    torch.tensor(sector_id, dtype=torch.long),
                    torch.tensor(sample_weight, dtype=torch.float32),
                    torch.tensor(time_index, dtype=torch.long),
                )


def build_dataloader(
    shard_paths: list[Path],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    positive_offset: int = 4,
    synthetic_loss_weight: float = 0.5,
    symbol_to_sector: dict[str, int] | None = None,
    symbol_to_id: dict[str, int] | None = None,
    global_regime_labels: np.ndarray | None = None,
) -> DataLoader:
    ds = ShardDataset(
        shard_paths,
        seq_len=seq_len,
        shuffle=shuffle,
        positive_offset=positive_offset,
        synthetic_loss_weight=synthetic_loss_weight,
        symbol_to_sector=symbol_to_sector,
        symbol_to_id=symbol_to_id,
        global_regime_labels=global_regime_labels,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_split_loaders(
    data_dir: Path,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    positive_offset: int = 4,
    synthetic_loss_weight: float = 0.5,
    symbol_to_sector: dict[str, int] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from {data_dir}/train/ and {data_dir}/validation/.

    Fits HMM regime labels on the train split (§11.2 / §11.3) and assigns
    globally-consistent asset IDs for correct multi-worker batching.
    Raises RuntimeError if no train shards are found.
    """
    from market_data.dataset_ingestion import enumerate_shards

    train_shards = enumerate_shards(data_dir, "train")
    val_shards = enumerate_shards(data_dir, "validation")

    if not train_shards:
        raise RuntimeError(
            f"No train shards found at {data_dir}/train. "
            "Run `python -m market_data.pipeline collect` first."
        )

    # Globally-consistent asset IDs (required for correct multi-worker batching).
    all_stems = sorted({p.stem for p in train_shards + val_shards})
    symbol_to_id = {stem: idx for idx, stem in enumerate(all_stems)}

    train_regime_labels, val_regime_labels = build_spy_regime_labels(data_dir)
    if train_regime_labels is None:
        print(
            "SPY shard not found — regime labels will use heuristic fallback. "
            "Run 'make collect' and ensure SPY is in symbols.txt."
        )

    train_loader = build_dataloader(
        train_shards,
        seq_len,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        positive_offset=positive_offset,
        synthetic_loss_weight=synthetic_loss_weight,
        symbol_to_sector=symbol_to_sector,
        symbol_to_id=symbol_to_id,
        global_regime_labels=train_regime_labels,
    )
    val_loader = build_dataloader(
        val_shards,
        seq_len,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        positive_offset=positive_offset,
        synthetic_loss_weight=synthetic_loss_weight,
        symbol_to_sector=symbol_to_sector,
        symbol_to_id=symbol_to_id,
        global_regime_labels=val_regime_labels,
    )

    return train_loader, val_loader


# ── Checkpoint ───────────────────────────────────────────────────────────────


@dataclass
class Checkpoint:
    step: int
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    best_val_loss: float
    best_val_contrastive_acc: float = 0.0
    format_version: int = 2
    architecture_version: str = "v4.0"
    pretraining_objective: str = "tcp"


def save_checkpoint(ckpt: Checkpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": ckpt.format_version,
            "architecture_version": ckpt.architecture_version,
            "pretraining_objective": ckpt.pretraining_objective,
            "step": ckpt.step,
            "model_state": ckpt.model_state,
            "optimizer_state": ckpt.optimizer_state,
            "scheduler_state": ckpt.scheduler_state,
            "best_val_loss": ckpt.best_val_loss,
            "best_val_contrastive_acc": ckpt.best_val_contrastive_acc,
        },
        tmp,
    )
    tmp.rename(path)


def load_checkpoint(path: Path) -> Checkpoint:
    data = torch.load(path, map_location="cpu", weights_only=True)
    version = data.get("format_version", 0)
    if version != 2:
        raise ValueError(f"Unsupported checkpoint format version {version}")
    if data.get("architecture_version") != "v4.0":
        raise ValueError("Checkpoint architecture_version is not v4.0")
    if data.get("pretraining_objective") != "tcp":
        raise ValueError("Checkpoint pretraining objective is not tcp")
    return Checkpoint(
        step=data["step"],
        model_state=data["model_state"],
        optimizer_state=data["optimizer_state"],
        scheduler_state=data["scheduler_state"],
        best_val_loss=data["best_val_loss"],
        best_val_contrastive_acc=float(data["best_val_contrastive_acc"]),
        format_version=version,
        architecture_version=data["architecture_version"],
        pretraining_objective=data["pretraining_objective"],
    )


# ── Loss functions ───────────────────────────────────────────────────────────


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    if weights.ndim != 1 or weights.shape[0] != values.shape[0]:
        raise ValueError(
            "weights must be rank-1 and align with batch dimension, "
            f"got weights={tuple(weights.shape)}, values={tuple(values.shape)}"
        )
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def _multi_positive_info_nce_losses(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError(
            "positive_mask must match logits shape, "
            f"got logits={tuple(logits.shape)} mask={tuple(positive_mask.shape)}"
        )
    masked_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    log_num = torch.logsumexp(masked_logits, dim=1)
    log_den = torch.logsumexp(logits, dim=1)
    return log_den - log_num


def _pairwise_return_correlation(return_windows: torch.Tensor) -> torch.Tensor:
    if return_windows.ndim != 2:
        raise ValueError(
            f"return_windows must have shape (B,T), got {tuple(return_windows.shape)}"
        )
    if return_windows.shape[1] <= 1:
        return torch.zeros(
            return_windows.shape[0],
            return_windows.shape[0],
            device=return_windows.device,
            dtype=return_windows.dtype,
        )
    centered = return_windows - return_windows.mean(dim=1, keepdim=True)
    norms = centered.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (centered @ centered.T) / (norms @ norms.T)


def _build_contrastive_logits_and_mask(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    asset_ids: torch.Tensor | None = None,
    regime_ids: torch.Tensor | None = None,
    sector_ids: torch.Tensor | None = None,
    return_windows: torch.Tensor | None = None,
    time_indices: torch.Tensor | None = None,
    temperature: float = 0.1,
    hard_negative_boost: float = 0.25,
    correlation_threshold: float = 0.6,
    min_negative_gap: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.T) / temperature
    positive_mask = torch.eye(z_a.shape[0], device=z_a.device, dtype=torch.bool)

    if asset_ids is not None and regime_ids is not None:
        if asset_ids.ndim != 1 or regime_ids.ndim != 1:
            raise ValueError("asset_ids and regime_ids must be rank-1 tensors")
        if asset_ids.shape[0] != z_a.shape[0] or regime_ids.shape[0] != z_a.shape[0]:
            raise ValueError("metadata batch dimension must match embeddings")
        same_asset = asset_ids[:, None] == asset_ids[None, :]
        diff_regime = regime_ids[:, None] != regime_ids[None, :]
        hard_negatives = same_asset & diff_regime
        logits = logits + hard_negatives.to(logits.dtype) * hard_negative_boost

    if (
        asset_ids is not None
        and regime_ids is not None
        and sector_ids is not None
        and return_windows is not None
    ):
        if sector_ids.ndim != 1 or sector_ids.shape[0] != z_a.shape[0]:
            raise ValueError("sector_ids must be rank-1 and batch-aligned")
        corr = _pairwise_return_correlation(return_windows)
        known_sector = sector_ids >= 0
        same_sector = sector_ids[:, None] == sector_ids[None, :]
        same_regime = regime_ids[:, None] == regime_ids[None, :]
        diff_asset = asset_ids[:, None] != asset_ids[None, :]
        additional_positives = (
            known_sector[:, None]
            & known_sector[None, :]
            & same_sector
            & same_regime
            & diff_asset
            & (corr > correlation_threshold)
        )
        positive_mask = positive_mask | additional_positives

    if asset_ids is not None and time_indices is not None and min_negative_gap >= 0:
        if time_indices.ndim != 1 or time_indices.shape[0] != z_a.shape[0]:
            raise ValueError("time_indices must be rank-1 and batch-aligned")
        same_asset = asset_ids[:, None] == asset_ids[None, :]
        time_delta = (time_indices[:, None] - time_indices[None, :]).abs()
        nearby_same_asset = same_asset & (time_delta <= min_negative_gap)
        disallowed_negatives = nearby_same_asset & ~positive_mask
        logits = logits.masked_fill(disallowed_negatives, float("-inf"))

    return logits, positive_mask


def regime_aware_contrastive_terms(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    asset_ids: torch.Tensor | None = None,
    regime_ids: torch.Tensor | None = None,
    sector_ids: torch.Tensor | None = None,
    return_windows: torch.Tensor | None = None,
    time_indices: torch.Tensor | None = None,
    temperature: float = 0.1,
    hard_negative_boost: float = 0.25,
    correlation_threshold: float = 0.6,
    min_negative_gap: int = 50,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric multi-positive InfoNCE with regime-aware hard negatives."""
    logits, positive_mask = _build_contrastive_logits_and_mask(
        z_a,
        z_b,
        asset_ids=asset_ids,
        regime_ids=regime_ids,
        sector_ids=sector_ids,
        return_windows=return_windows,
        time_indices=time_indices,
        temperature=temperature,
        hard_negative_boost=hard_negative_boost,
        correlation_threshold=correlation_threshold,
        min_negative_gap=min_negative_gap,
    )
    weights = None
    if sample_weights is not None:
        weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
    losses_ab = _multi_positive_info_nce_losses(logits, positive_mask)
    losses_ba = _multi_positive_info_nce_losses(logits.T, positive_mask.T)
    loss_ab = _weighted_mean(losses_ab, weights)
    loss_ba = _weighted_mean(losses_ba, weights)
    loss = 0.5 * (loss_ab + loss_ba)

    # Accuracy: fraction of anchors where the best positive outranks the best
    # negative.  We mask using the full positive_mask (not just the diagonal) so
    # that multi-asset positives are excluded from the negative pool — counting
    # them as "mistakes" would underestimate accuracy and trigger early stopping
    # prematurely.
    neg_fill = torch.finfo(logits.dtype).min
    neg_masked = logits.masked_fill(positive_mask, neg_fill)
    pos_masked = logits.masked_fill(~positive_mask, neg_fill)
    max_neg_row = neg_masked.max(dim=1).values
    max_neg_col = neg_masked.max(dim=0).values
    max_pos_row = pos_masked.max(dim=1).values
    max_pos_col = pos_masked.max(dim=0).values
    acc_ab = (max_pos_row > max_neg_row).float().mean()
    acc_ba = (max_pos_col > max_neg_col).float().mean()
    contrastive_acc = 0.5 * (acc_ab + acc_ba)
    return loss, contrastive_acc


# ── Trainer ──────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    total_steps: int = 20_000
    batch_size: int = 128
    seq_len: int = 64
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1_000
    save_every: int = 5_000
    eval_max_batches: int = 50
    contrastive_temperature: float = 0.1
    contrastive_weight: float = 0.7
    prediction_weight: float = 0.3
    positive_pair_offset: int = 4
    fda_loss_weight: float = 0.5
    hard_negative_boost: float = 0.25
    multi_asset_corr_threshold: float = 0.6
    negative_min_separation: int = 50
    early_stop_patience_evals: int = 20
    early_stop_min_delta: float = 1e-4
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    checkpoint_name: str = "backbone_pretrain.pt"
    tsne_every: int = 5_000
    tsne_reports_dir: Path = field(default_factory=lambda: Path("reports"))
    compile: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name


class SelfSupervisedPretrainer:
    """Trains the hybrid RA-GDN + SWA backbone with v4 TCP objective."""

    def __init__(
        self,
        backbone: nn.Module,
        fusion: nn.Module,
        regime_detector: nn.Module,
        input_projection: nn.Module,
        d_model: int = 768,
        n_assets: int = 100,
        device: torch.device | None = None,
        gradient_checkpointing: bool = False,
        bptt_horizon: int = 0,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 0 = full BPTT; K > 0 detaches states every K steps, capping VRAM at O(K)
        # instead of O(T) — important on 24 GB GPUs with T=64, B=128.
        self.bptt_horizon = bptt_horizon
        self.backbone = backbone.to(self.device)
        if gradient_checkpointing and hasattr(
            backbone, "enable_gradient_checkpointing"
        ):
            backbone.enable_gradient_checkpointing()
        self.fusion = fusion.to(self.device)
        self.regime_detector = regime_detector.to(self.device)
        self.input_projection = input_projection.to(self.device)
        self.projection_head = ProjectionHead(d_model=d_model).to(self.device)
        self.prediction_head = ReturnPredictionHead(d_model=d_model).to(self.device)

        self._all_modules = nn.ModuleList(
            [
                backbone,
                fusion,
                regime_detector,
                input_projection,
                self.projection_head,
                self.prediction_head,
            ]
        )

    def _cosine_lr(self, step: int, cfg: TrainConfig) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(
            cfg.total_steps - cfg.warmup_steps, 1
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min(cfg.min_lr / max(cfg.lr, 1e-12), 1.0)
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) != 8:
            raise ValueError(f"expected 8-tuple batch, got arity {len(batch)}")
        (
            features,
            positives,
            next_returns,
            asset_ids,
            regime_ids,
            sector_ids,
            sample_weights,
            time_indices,
        ) = batch
        return (
            features,
            positives,
            next_returns,
            asset_ids,
            regime_ids,
            sector_ids,
            sample_weights,
            time_indices,
        )

    def _model_state(self) -> dict:
        backbone_mod = getattr(self.backbone, "_orig_mod", self.backbone)
        return {
            "backbone": backbone_mod.state_dict(),
            "fusion": self.fusion.state_dict(),
            "regime_detector": self.regime_detector.state_dict(),
            "input_projection": self.input_projection.state_dict(),
            "projection_head": self.projection_head.state_dict(),
            "prediction_head": self.prediction_head.state_dict(),
        }

    def _load_model_state(self, state: dict) -> None:
        backbone_mod = getattr(self.backbone, "_orig_mod", self.backbone)
        backbone_mod.load_state_dict(state["backbone"])
        self.fusion.load_state_dict(state["fusion"])
        self.regime_detector.load_state_dict(state["regime_detector"])
        self.input_projection.load_state_dict(state["input_projection"])
        self.projection_head.load_state_dict(state["projection_head"])
        self.prediction_head.load_state_dict(state["prediction_head"])

    def evaluate(
        self,
        val_loader: DataLoader,
        cfg: TrainConfig,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        self._all_modules.eval()
        max_eval_batches = cfg.eval_max_batches if max_batches is None else max_batches
        totals: dict[str, float] = {
            "contrastive": 0.0,
            "prediction": 0.0,
            "total": 0.0,
            "contrastive_acc": 0.0,
        }
        pred_values: list[torch.Tensor] = []
        target_values: list[torch.Tensor] = []
        n = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_eval_batches:
                    break
                (
                    features,
                    positives,
                    next_returns,
                    asset_ids,
                    regime_ids,
                    sector_ids,
                    sample_weights,
                    time_indices,
                ) = self._unpack_batch(batch)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    losses = self._forward_losses(
                        features,
                        positives,
                        next_returns=next_returns,
                        asset_ids=asset_ids,
                        regime_ids=regime_ids,
                        sector_ids=sector_ids,
                        sample_weights=sample_weights,
                        time_indices=time_indices,
                        temperature=cfg.contrastive_temperature,
                        contrastive_weight=cfg.contrastive_weight,
                        prediction_weight=cfg.prediction_weight,
                        hard_negative_boost=cfg.hard_negative_boost,
                        correlation_threshold=cfg.multi_asset_corr_threshold,
                        min_negative_gap=cfg.negative_min_separation,
                        return_aux=True,
                    )
                for k in ("contrastive", "prediction", "total", "contrastive_acc"):
                    v = losses[k]
                    totals[k] += v.item()
                pred_values.append(losses["pred_returns"].detach().cpu())
                target_values.append(losses["target_returns"].detach().cpu())
                n += 1
        self._all_modules.train()
        metrics = {k: v / max(n, 1) for k, v in totals.items()}
        if pred_values:
            pred = torch.cat(pred_values).to(torch.float32)
            target = torch.cat(target_values).to(torch.float32)
            sse = torch.sum((pred - target) ** 2)
            sst = torch.sum((target - target.mean()) ** 2)
            metrics["prediction_r2"] = (1.0 - (sse / sst)).item() if sst > 0 else 0.0
            # Diagnostic: surface scale mismatches so catastrophic R² can be
            # traced to pred/target distribution rather than modelling failure.
            print(
                f"  val pred  mean={pred.mean().item():+.3e} std={pred.std().item():.3e}  "
                f"target mean={target.mean().item():+.3e} std={target.std().item():.3e}  "
                f"SSE={sse.item():.4f}  SST={sst.item():.4f}"
            )
        else:
            metrics["prediction_r2"] = 0.0

        # §11.3 diagnostic gates — separate methods keep the hot eval loop lean.
        metrics["swa_entropy"] = self.eval_swa_entropy(
            val_loader, max_batches=min(max_eval_batches, 10)
        )
        metrics["regime_separation"] = self.eval_regime_separation(
            val_loader, max_batches=min(max_eval_batches, 20)
        )
        return metrics

    def eval_swa_entropy(
        self,
        val_loader: DataLoader,
        max_batches: int = 10,
    ) -> float:
        """Mean normalized SWA attention entropy over val batches (§11.3 gate: < 0.80).

        Derived from the SWA cache states returned by _encode_sequence — no extra
        forward pass required.
        """
        if not self.backbone.swa_blocks:
            return float("nan")

        self._all_modules.eval()
        entropies: list[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                features = self._unpack_batch(batch)[0].to(self.device)
                _, states = self._encode_sequence(features, return_states=True)
                entropies.append(self.backbone.last_swa_entropy(states))
        self._all_modules.train()
        return float(np.mean(entropies)) if entropies else float("nan")

    def eval_regime_separation(
        self,
        val_loader: DataLoader,
        max_batches: int = 20,
    ) -> float:
        """Silhouette score of k=4 clustering on backbone embeddings (§11.3 gate: > 0.10).

        Operationalises the qualitative "t-SNE shows 4 clusters" criterion with
        k-means silhouette. Does not use HMM labels — label-based separation
        breaks when the val period is dominated by a single regime.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        self._all_modules.eval()
        all_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                features = self._unpack_batch(batch)[0].to(self.device)
                emb = self._encode_sequence(features)  # (B, D)
                all_embs.append(F.normalize(emb.float(), dim=-1).cpu())
        self._all_modules.train()

        if not all_embs:
            return float("nan")

        embs_np = torch.cat(all_embs, dim=0).numpy()  # (N, D)
        if len(embs_np) < 8:
            return float("nan")

        km = KMeans(n_clusters=4, n_init=5, random_state=0)
        labels = km.fit_predict(embs_np)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(embs_np, labels, metric="cosine", sample_size=min(500, len(embs_np))))

    def eval_tsne_plot(
        self,
        val_loader: DataLoader,
        step: int,
        reports_dir: Path,
        max_batches: int = 20,
    ) -> Path | None:
        """t-SNE scatter coloured by regime label for the §11.3 cluster gate.

        Saves a PNG to reports_dir; returns the path or None if matplotlib/sklearn
        are unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
        except ImportError:
            print("  t-SNE plot skipped — install matplotlib and scikit-learn")
            return None

        self._all_modules.eval()
        all_embs: list[torch.Tensor] = []
        all_regimes: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                features, _, _, _, regime_ids, _, _, _ = self._unpack_batch(batch)
                features = features.to(self.device)
                emb = self._encode_sequence(features)
                all_embs.append(F.normalize(emb.float(), dim=-1).cpu())
                all_regimes.append(regime_ids.cpu())
        self._all_modules.train()

        if not all_embs:
            return None
        embs_np = torch.cat(all_embs, dim=0).numpy()
        regimes_np = torch.cat(all_regimes, dim=0).numpy()
        if len(embs_np) < 8:
            return None

        perplexity = min(30, len(embs_np) - 1)
        coords = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(embs_np)

        colors = ["#2ecc71", "#e67e22", "#3498db", "#e74c3c"]
        regime_labels = ["bull", "hi-vol", "mean-rev", "crisis"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for rid in range(4):
            mask = regimes_np == rid
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors[rid], label=regime_labels[rid],
                alpha=0.6, s=12, linewidths=0,
            )
        ax.legend(title="regime", fontsize=8)
        ax.set_title(f"Backbone t-SNE — step {step:,}  (§11.3 cluster gate)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"pretrain_step{step:07d}_tsne.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  t-SNE → {out_path}")
        return out_path

    def _encode_sequence(
        self,
        features: torch.Tensor,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a temporal window to a pooled representation (§11.3: z = MLP(pool(backbone(W)))).

        n_assets=1 — cross-asset fusion is Phase-2 only (§11.4a).
        Portfolio and calendar inputs are zeros; only relevant after head training.
        """
        B, T, _ = features.shape
        asset_feats = features[..., :184].unsqueeze(2)  # (B, T, 1, 184)
        market_feats = features[..., 184:]              # (B, T, 96)

        n_assets = 1
        portfolio_s = torch.zeros(B, 201, device=self.device)
        portfolio_summary = torch.zeros(B, 3, device=self.device)
        calendar = torch.zeros(B, 20, device=self.device)
        states = self.backbone.init_state(B, n_assets, self.device)

        # Per-step regime detection (§7.2). ema_alpha=0.95 → ~14-step half-life.
        ema_alpha = 0.95
        regime_ema = torch.zeros(B, 64, device=self.device)
        prev_emb: torch.Tensor | None = None
        regime_embs_list: list[torch.Tensor] = []
        posteriors_list: list[torch.Tensor] = []
        for t in range(T):
            re_t, post_t, _ = self.regime_detector(
                market_feats[:, t], portfolio_summary, regime_ema, prev_emb=prev_emb
            )
            regime_embs_list.append(re_t)
            posteriors_list.append(post_t)
            regime_ema = ema_alpha * regime_ema + (1.0 - ema_alpha) * re_t.detach()
            prev_emb = re_t
        regime_embs_t = torch.stack(regime_embs_list, dim=1)  # (B, T, 64)
        posteriors_t = torch.stack(posteriors_list, dim=1)    # (B, T, 4)

        # InputProjection is stateless over T — flatten then restore.
        asset_in = asset_feats.reshape(B * T, n_assets, -1)
        market_in = market_feats.reshape(B * T, -1)
        portfolio_in = portfolio_s.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        calendar_in = calendar.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        regime_in = regime_embs_t.reshape(B * T, -1)
        asset_tok_all, _, _, _ = self.input_projection(
            asset_in, market_in, portfolio_in, calendar_in, regime_in
        )
        asset_tok_all = asset_tok_all.view(B, T, n_assets, -1)

        pooled_steps: list[torch.Tensor] = []
        for t in range(T):
            if self.bptt_horizon > 0 and t > 0 and t % self.bptt_horizon == 0:
                states = [s.detach() for s in states]
            asset_tok = asset_tok_all[:, t]       # (B, 1, D)
            re_t = regime_embs_t[:, t]
            post_t = posteriors_t[:, t]
            out, states = self.backbone(asset_tok, re_t, post_t, states)
            pooled_steps.append(out.squeeze(1))   # (B, D)

        pooled = torch.stack(pooled_steps, dim=1).mean(dim=1)  # (B, D)
        if return_states:
            return pooled, states
        return pooled

    def _forward_losses(
        self,
        features: torch.Tensor,  # (B, T, 280) — per-asset (184) + market (96)
        positives: torch.Tensor,  # (B, T, 280) — temporally adjacent window
        next_returns: torch.Tensor | None = None,  # (B,) one-step-ahead return target
        asset_ids: torch.Tensor | None = None,
        regime_ids: torch.Tensor | None = None,
        sector_ids: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        time_indices: torch.Tensor | None = None,
        temperature: float = 0.1,
        contrastive_weight: float = 0.7,
        prediction_weight: float = 0.3,
        hard_negative_boost: float = 0.25,
        correlation_threshold: float = 0.6,
        min_negative_gap: int = 50,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = features.to(self.device)
        positives = positives.to(self.device)
        if next_returns is None:
            next_returns = positives[:, -1, RETURN_FEATURE_INDEX]
        next_returns = next_returns.to(self.device).to(torch.float32)
        if asset_ids is not None:
            asset_ids = asset_ids.to(self.device)
        if regime_ids is not None:
            regime_ids = regime_ids.to(self.device)
        if sector_ids is not None:
            sector_ids = sector_ids.to(self.device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device).to(torch.float32)
        if time_indices is not None:
            time_indices = time_indices.to(self.device)
        view_a = self._encode_sequence(features)
        view_b = self._encode_sequence(positives)
        z_a = self.projection_head(view_a)
        z_b = self.projection_head(view_b)
        loss_contrastive, contrastive_acc = regime_aware_contrastive_terms(
            z_a,
            z_b,
            asset_ids=asset_ids,
            regime_ids=regime_ids,
            sector_ids=sector_ids,
            return_windows=features[:, :, RETURN_FEATURE_INDEX],
            time_indices=time_indices,
            temperature=temperature,
            hard_negative_boost=hard_negative_boost,
            correlation_threshold=correlation_threshold,
            min_negative_gap=min_negative_gap,
            sample_weights=sample_weights,
        )

        # §11.3: L_pred = MSE(predict_return, actual_return_{t+1}).
        # The head also outputs log-vol (§7.9, used for surprise signal at
        # inference), but vol is not supervised during Phase-1 pretraining.
        pred_returns_raw, _ = self.prediction_head(view_a)
        pred_returns = pred_returns_raw.to(torch.float32)
        pred_errors = (pred_returns - next_returns) ** 2

        if sample_weights is None:
            loss_prediction = pred_errors.mean()
        else:
            denom = sample_weights.sum().clamp_min(1e-8)
            loss_prediction = (pred_errors * sample_weights).sum() / denom

        loss_total = (
            contrastive_weight * loss_contrastive + prediction_weight * loss_prediction
        )
        out: dict[str, torch.Tensor] = {
            "contrastive": loss_contrastive,
            "prediction": loss_prediction,
            "total": loss_total,
            "contrastive_acc": contrastive_acc,
        }
        if return_aux:
            out["pred_returns"] = pred_returns
            out["target_returns"] = next_returns
        return out

    def train(
        self,
        train_loader: DataLoader,
        cfg: TrainConfig,
        val_loader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> None:
        # Prediction head sits on near-unpredictable log-return targets; at the
        # backbone lr it chases batch-mean drift and oscillates. Give it a 10×
        # lower lr so it settles on the optimal constant (predict ≈ 0) and only
        # moves off it when backbone features carry real signal.
        pred_head_params = list(self.prediction_head.parameters())
        pred_head_ids = {id(p) for p in pred_head_params}
        # Fusion is Phase-2 only (§11.4a) — never called during Phase-1 _encode_sequence.
        # Exclude its params from the optimizer so weight decay doesn't decay them
        # toward zero before Phase 2 starts.
        fusion_param_ids = {id(p) for p in self.fusion.parameters()}
        other_params = [
            p for p in self._all_modules.parameters()
            if id(p) not in pred_head_ids and id(p) not in fusion_param_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": cfg.lr},
                {"params": pred_head_params, "lr": cfg.lr * 0.1},
            ],
            lr=cfg.lr,
            weight_decay=0.01,
            fused=(self.device.type == "cuda"),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self._cosine_lr(step, cfg)
        )

        start_step = 0
        best_val_loss = float("inf")
        best_val_contrastive_acc = 0.0
        stale_eval_count = 0

        if resume_from is not None and resume_from.exists():
            ckpt = load_checkpoint(resume_from)
            self._load_model_state(ckpt.model_state)
            # Skip optimizer state if param groups changed (e.g. prediction head
            # gained its own group) to avoid shape errors; weights load either way.
            groups_match = len(ckpt.optimizer_state.get("param_groups", [])) == len(
                optimizer.param_groups
            )
            if groups_match:
                try:
                    optimizer.load_state_dict(ckpt.optimizer_state)
                    scheduler.load_state_dict(ckpt.scheduler_state)
                except (ValueError, RuntimeError) as e:
                    print(
                        f"Optimizer/scheduler state load failed ({e}); "
                        f"continuing with fresh Adam moments."
                    )
            else:
                print(
                    f"Optimizer param groups changed since checkpoint "
                    f"({len(ckpt.optimizer_state.get('param_groups', []))} "
                    f"→ {len(optimizer.param_groups)}); continuing with "
                    f"fresh Adam moments and scheduler state."
                )
                # Advance scheduler to match resume step so warmup/cosine align.
                for _ in range(ckpt.step):
                    scheduler.step()
            start_step = ckpt.step
            best_val_loss = ckpt.best_val_loss
            best_val_contrastive_acc = ckpt.best_val_contrastive_acc
            print(
                f"Resumed from step {start_step}, "
                f"best_val_loss={best_val_loss:.4f}, "
                f"best_val_contrastive_acc={best_val_contrastive_acc:.4f}"
            )

        if cfg.compile:
            print("Compiling backbone (one-time warmup)...")
            self.backbone = torch.compile(self.backbone)
            self._all_modules[0] = self.backbone
            print("Compiled.")

        self._all_modules.train()
        step = start_step
        data_iter = iter(train_loader)

        while step < cfg.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            (
                features,
                positives,
                next_returns,
                asset_ids,
                regime_ids,
                sector_ids,
                sample_weights,
                time_indices,
            ) = self._unpack_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                losses = self._forward_losses(
                    features,
                    positives,
                    next_returns=next_returns,
                    asset_ids=asset_ids,
                    regime_ids=regime_ids,
                    sector_ids=sector_ids,
                    sample_weights=sample_weights,
                    time_indices=time_indices,
                    temperature=cfg.contrastive_temperature,
                    contrastive_weight=cfg.contrastive_weight,
                    prediction_weight=cfg.prediction_weight,
                    hard_negative_boost=cfg.hard_negative_boost,
                    correlation_threshold=cfg.multi_asset_corr_threshold,
                    min_negative_gap=cfg.negative_min_separation,
                )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self._all_modules.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                parts = "  ".join(
                    f"{k}={losses[k].item():.4f}"
                    for k in ("contrastive", "prediction", "total", "contrastive_acc")
                )
                print(f"step={step:>7d}  {parts}  lr={lr_now:.2e}")

            if val_loader is not None and step % cfg.eval_every == 0:
                val_metrics = self.evaluate(val_loader, cfg)
                val_loss = val_metrics["total"]
                val_contrastive_acc = val_metrics["contrastive_acc"]
                val_pred_r2 = val_metrics["prediction_r2"]
                # At B=128 contrastive_acc saturates near 1.0 quickly — the batch
                # is too small to surface hard negatives. Gate saves and patience
                # on contrastive_acc per spec §11.3; best_val_loss is tracked
                # separately for logging.
                best_val_loss = min(best_val_loss, val_loss)
                if val_contrastive_acc > (
                    best_val_contrastive_acc + cfg.early_stop_min_delta
                ):
                    best_val_contrastive_acc = val_contrastive_acc
                    stale_eval_count = 0
                    save_checkpoint(
                        Checkpoint(
                            step=step,
                            model_state=self._model_state(),
                            optimizer_state=optimizer.state_dict(),
                            scheduler_state=scheduler.state_dict(),
                            best_val_loss=best_val_loss,
                            best_val_contrastive_acc=best_val_contrastive_acc,
                        ),
                        cfg.checkpoint_path,
                    )
                else:
                    stale_eval_count += 1
                    if (
                        cfg.early_stop_patience_evals > 0
                        and stale_eval_count >= cfg.early_stop_patience_evals
                    ):
                        print(
                            "Early stopping on validation contrastive accuracy "
                            f"patience ({cfg.early_stop_patience_evals} evals)."
                        )
                        break
                swa_ent = val_metrics.get("swa_entropy", float("nan"))
                regime_sep = val_metrics.get("regime_separation", float("nan"))
                acc_gate = "PASS" if val_contrastive_acc > 0.60 else "FAIL"
                r2_gate = "PASS" if val_pred_r2 > 0.01 else "FAIL"
                swa_gate = "PASS" if swa_ent < 0.80 else "FAIL"
                sep_gate = "PASS" if regime_sep > 0.10 else "FAIL"
                print(
                    f"  val total={val_loss:.4f}  "
                    f"contrastive_acc={val_contrastive_acc:.4f}[{acc_gate}]  "
                    f"pred_r2={val_pred_r2:.4f}[{r2_gate}]  "
                    f"swa_entropy={swa_ent:.3f}[{swa_gate}]  "
                    f"cluster_silhouette={regime_sep:.3f}[{sep_gate}]  "
                    f"best_contrastive_acc={best_val_contrastive_acc:.4f}  "
                    f"best_loss={best_val_loss:.4f}  "
                    f"stale={stale_eval_count}"
                )

            # §11.3 t-SNE cluster gate — generate plot at tsne_every interval.
            if (
                val_loader is not None
                and cfg.tsne_every > 0
                and step % cfg.tsne_every == 0
            ):
                self.eval_tsne_plot(
                    val_loader, step, cfg.tsne_reports_dir,
                    max_batches=cfg.eval_max_batches,
                )

            if step % cfg.save_every == 0:
                ckpt_path = cfg.checkpoint_dir / f"backbone_step{step}.pt"
                save_checkpoint(
                    Checkpoint(
                        step=step,
                        model_state=self._model_state(),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),
                        best_val_loss=best_val_loss,
                        best_val_contrastive_acc=best_val_contrastive_acc,
                    ),
                    ckpt_path,
                )

        print(
            f"Training complete at step {step}. "
            f"Best contrastive_acc={best_val_contrastive_acc:.4f}, "
            f"best val total={best_val_loss:.4f}"
        )
