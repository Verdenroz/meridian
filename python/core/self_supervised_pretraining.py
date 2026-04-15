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

PER_ASSET_FEATURE_DIM = 184
MARKET_FEATURE_DIM = 96
TOTAL_FEATURE_DIM = PER_ASSET_FEATURE_DIM + MARKET_FEATURE_DIM
RETURN_FEATURE_INDEX = 3
SYNTHETIC_SHARD_PREFIX = "FDA_SYNTH_"
UNKNOWN_SECTOR_ID = -1


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
    """Auxiliary next-return prediction head used in Phase-1 TCP training.

    Targets are per-bar log-returns at O(1e-3) magnitude. The head is
    zero-initialised so early predictions start at ~0 instead of the
    natural Linear init range (~O(1e-1)), avoiding 3-order-of-magnitude
    MSE blow-up and catastrophic val R² during warmup.
    """

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, 1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def infer_regime_label(feature_seq: np.ndarray) -> int:
    """Heuristic 4-state regime label — fallback for synthetic shards and
    smoke tests where no HMM labels are available.

    Uses two robust proxies available in every shard row:
      - volatility proxy: market block index 0 (global offset 184)
      - return proxy: close log-return at index 3

    Canonical IDs match the spec:
      0 = low-vol bull  1 = high-vol trending
      2 = mean-reverting  3 = crisis
    """
    vol_proxy = float(feature_seq[:, 184].mean())
    ret_proxy = float(feature_seq[:, 3].mean())

    if vol_proxy > 1.5:
        return 3  # crisis
    if abs(ret_proxy) > 0.5 or vol_proxy > 0.6:
        return 1  # high-vol trending
    if ret_proxy >= 0.0:
        return 0  # low-vol trending
    return 2  # mean-reverting


# ── HMM regime labeling (§11.2 Phase 0, §11.3) ──────────────────────────────


def _shard_returns(path: Path) -> np.ndarray:
    """Load [return, |return|] as 2-D HMM features for vol-regime separation.

    A 1-D return-only feature collapses high-vol-bull and high-vol-bear into
    ambiguous clusters, leaving the HMM unable to separate vol regimes from
    directional bias. Adding |return| gives EM a clean volatility axis so all
    four canonical regimes receive meaningful probability mass.
    """
    raw = np.memmap(path, dtype="float32", mode="r")
    n_rows = raw.size // TOTAL_FEATURE_DIM
    data = raw.reshape(n_rows, TOTAL_FEATURE_DIM)
    returns = data[:, RETURN_FEATURE_INDEX].astype(np.float64)
    return np.column_stack([returns, np.abs(returns)])


def _canonicalize_hmm(
    hmm: object, raw_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw HMM state integers to the spec's canonical regime IDs.

    Canonicalization rules (spec §11.3) for 2-D [return, |return|] features:
      0 = low-vol bull      → highest mean return
      1 = high-vol trending → higher mean |return| of the two middle states
      2 = mean-reverting    → lower mean |return| of the two middle states
      3 = crisis            → lowest mean return

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
    """Fit a 4-state Gaussian HMM on SPY training-split returns and return
    per-row regime labels for both train and validation splits.

    Labels use the spec's canonical IDs (§11.3):
      0 = low-vol bull    1 = high-vol trending
      2 = mean-reverting  3 = crisis

    The HMM is fit exclusively on the training split to prevent look-ahead
    leakage; the fitted model is then applied to the validation split.
    Labels are cached alongside the data directory for fast restarts.

    Args:
        data_dir:  Root shard directory (contains train/ and validation/).
        n_states:  Number of HMM hidden states — must equal spec.n_regimes=4.

    Returns:
        (train_labels, val_labels) — int64 arrays of shape (n_rows,) per split,
        or None for either element if the SPY shard is absent.
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

        min_start = max(0, -self.positive_offset)
        max_start = n_rows - self.seq_len - max(self.positive_offset, 0)
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
    """Build train and validation DataLoaders from split-aware shard directories.

    Expects shards at:
      {data_dir}/train/*.f32bin       — used for training
      {data_dir}/validation/*.f32bin  — used for evaluation

    Args:
        data_dir:    Root shard directory (output of market_data.pipeline collect).
        seq_len:     Sequence length per sample.
        batch_size:  Batch size.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader)

    Raises:
        RuntimeError: If train shards are missing.
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

    # HMM regime labels from SPY (§11.2 / §11.3).  The HMM is fit on the
    # training split only; the fitted model predicts val labels to avoid
    # look-ahead.  Falls back silently to heuristic if SPY is absent.
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
    multi_asset_corr_threshold: float = 0.6
    negative_min_separation: int = 50
    early_stop_patience_evals: int = 20
    early_stop_min_delta: float = 1e-4
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    checkpoint_name: str = "backbone_pretrain.pt"
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
        # bptt_horizon=0 means full BPTT (T steps).  Any positive value detaches
        # recurrent states every K steps, limiting the backward graph to K steps
        # of state history.  Reduces peak VRAM from O(T) to O(K) for GDN/SWA
        # state chains — critical for 24 GB GPUs with T=64 and B=128.
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

        # ── Asset-identity feature mask ──────────────────────────────────────
        # The 184-dim per-asset vector contains ~110 dims that fingerprint the
        # asset rather than market state — if fed to the contrastive branch
        # unchanged, the encoder shortcuts on asset identity and reaches ~99%
        # contrastive accuracy inside warmup without learning market state.
        #
        # Zero the fingerprint dims at input so both the contrastive and
        # prediction heads see only genuinely time-varying signal. Kept:
        #   price_volume   [0:10]     technical     [10:52]
        #   options        [52:64]    event_state   [102:112]
        #   market-wide    [184:280]
        # Zeroed:
        #   fundamental    [64:94]    risk_analytics      [94:102]
        #   cross_asset    [112:120]  text (MarketReader) [120:184]
        identity_mask = torch.ones(TOTAL_FEATURE_DIM, dtype=torch.float32)
        identity_mask[64:102] = 0.0
        identity_mask[112:184] = 0.0
        self.identity_mask = identity_mask.to(self.device)

        self._all_modules = nn.ModuleList(
            [
                backbone,
                fusion,
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
        return metrics

    def _encode_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """Encode one temporal window into a pooled sequence representation."""
        B, T, _ = features.shape
        asset_feats = features[..., :184].unsqueeze(2)  # (B, T, N=1, 184)
        market_feats = features[..., 184:]  # (B, T, 96)

        n_assets = 1
        regime_emb = torch.zeros(B, 64, device=self.device)
        posteriors = torch.full((B, 4), 0.25, device=self.device)
        portfolio_s = torch.zeros(B, 201, device=self.device)
        calendar = torch.zeros(B, 20, device=self.device)
        sector_ids = torch.zeros(n_assets, dtype=torch.long, device=self.device)
        episodic_tok = torch.zeros(B, 8, 768, device=self.device)
        states = self.backbone.init_state(B, n_assets, self.device)

        # ── Vectorize InputProjection over T (#3) ────────────────────────────
        # InputProjection is stateless across time, so we run all T steps in one
        # call by flattening (B, T) → (B*T). Numerically identical to looping.
        asset_in = asset_feats.reshape(B * T, n_assets, -1)  # (B*T, 1, 184)
        market_in = market_feats.reshape(B * T, -1)  # (B*T, 96)
        portfolio_in = portfolio_s.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        calendar_in = calendar.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        regime_in = regime_emb.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        asset_tok_all, mkt_tok_all, port_tok_all, _ = self.input_projection(
            asset_in, market_in, portfolio_in, calendar_in, regime_in
        )
        D = asset_tok_all.shape[-1]
        asset_tok_all = asset_tok_all.view(B, T, n_assets, D)
        mkt_tok_all = mkt_tok_all.view(B, T, D)
        port_tok_all = port_tok_all.view(B, T, D)

        # ── Precompute fusion sector bias once (#5) ──────────────────────────
        # Sector bias is a function of constant sector_ids and N_total tokens.
        # Compute once, reuse for all T fusion calls.
        n_total = n_assets + 2 + episodic_tok.shape[1]  # assets + market + port + episodic
        extra_ids = torch.full(
            (n_total - n_assets,), -1, dtype=torch.long, device=self.device
        )
        full_ids = torch.cat([sector_ids, extra_ids], dim=0)
        precomputed_bias = self.fusion.sector_bias.bias_matrix(full_ids, n_total)

        pooled_steps: list[torch.Tensor] = []
        for t in range(T):
            # Truncated BPTT: detach recurrent state every `bptt_horizon` steps
            # so the backward graph is bounded to K steps of state history.
            # bptt_horizon=0 keeps the default full-sequence BPTT.
            if self.bptt_horizon > 0 and t > 0 and t % self.bptt_horizon == 0:
                states = [s.detach() for s in states]
            asset_tok = asset_tok_all[:, t]  # (B, 1, D)
            mkt_tok = mkt_tok_all[:, t]  # (B, D)
            port_tok = port_tok_all[:, t]  # (B, D)
            out, states = self.backbone(asset_tok, regime_emb, posteriors, states)
            fused, _ = self.fusion(
                out,
                mkt_tok,
                port_tok,
                episodic_tok,
                sector_ids,
                precomputed_bias=precomputed_bias,
            )
            pooled_steps.append(fused.squeeze(1))  # (B, D)

        return torch.stack(pooled_steps, dim=1).mean(dim=1)  # (B, D)

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
        correlation_threshold: float = 0.6,
        min_negative_gap: int = 50,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = features.to(self.device)
        positives = positives.to(self.device)
        # Read the raw-return target before masking zeros it out (if ever).
        if next_returns is None:
            next_returns = positives[:, -1, RETURN_FEATURE_INDEX]
        # Mask asset-identity dims (fundamental / risk_analytics / cross_asset /
        # text) so the contrastive encoder cannot shortcut on asset fingerprints.
        # RETURN_FEATURE_INDEX=3 is in price_volume [0:10] and remains unmasked.
        features = features * self.identity_mask
        positives = positives * self.identity_mask
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
            correlation_threshold=correlation_threshold,
            min_negative_gap=min_negative_gap,
            sample_weights=sample_weights,
        )

        pred_returns = self.prediction_head(view_a).to(torch.float32)
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
        optimizer = torch.optim.AdamW(
            self._all_modules.parameters(),
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
            optimizer.load_state_dict(ckpt.optimizer_state)
            scheduler.load_state_dict(ckpt.scheduler_state)
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
                # Track best loss independently so it stays current even when
                # contrastive_acc plateaus (the two can decouple in later steps).
                best_val_loss = min(best_val_loss, val_loss)
                print(
                    f"  val total={val_loss:.4f}  "
                    f"contrastive_acc={val_contrastive_acc:.4f}  "
                    f"pred_r2={val_pred_r2:.4f}  "
                    f"best_loss={best_val_loss:.4f}"
                )
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
            f"Best val total={best_val_loss:.4f}, "
            f"best contrastive_acc={best_val_contrastive_acc:.4f}"
        )
