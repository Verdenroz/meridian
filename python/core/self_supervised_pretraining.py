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

# Auxiliary prediction target: forward realized volatility, not return level.
# One-step return level is a martingale — unpredictable no matter how much data
# we pour in — so its R² gate could never pass. Realized vol clusters strongly,
# is the axis meridian's regime taxonomy turns on, and the abundant cross-asset /
# cross-regime pretraining data makes it genuinely learnable.
PREDICTION_HORIZON: int = 10
_LOG_VOL_FLOOR: float = 1e-6

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
    """Auxiliary next-step prediction head.

    Outputs (return, log_vol) at t+1. During pretraining log_vol is supervised
    via MSE against forward realized volatility (the predictable, regime-defining
    signal); the return output is left unsupervised (one-step level is a martingale).

    Weights are zero-initialised; the log_vol bias is primed to a ~1% daily-vol
    prior so the head starts near the target mean instead of ~4.6 nats away.
    """

    # log(0.01) ≈ -4.6 — a typical daily log-return volatility.
    _LOG_VOL_BIAS_PRIOR: float = math.log(0.01)

    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        with torch.no_grad():
            self.net.bias[1] = self._LOG_VOL_BIAS_PRIOR

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[..., 0], out[..., 1]


# ── Asset adversarial head ───────────────────────────────────────────────────


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class AssetAdversarialHead(nn.Module):
    """Predicts asset_id via a gradient-reversal layer to push the encoder toward asset-invariant representations."""

    def __init__(self, d_model: int, n_assets: int) -> None:
        super().__init__()
        hidden = max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(x, lambda_))


def infer_regime_label(feature_seq: np.ndarray) -> int:
    """Heuristic 4-state regime label for synthetic shards lacking HMM labels.

    Proxies: VIX/100 at market offset 184 (volatility) and log-return at index 3.
    Canonical IDs: 0 = low-vol bull  1 = high-vol trending  2 = mean-reverting  3 = crisis
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


def next_window_log_vol(
    data: np.ndarray,
    start_idx: int,
    seq_len: int,
    horizon: int = PREDICTION_HORIZON,
) -> float:
    """Log realized volatility of the ``horizon`` returns following the anchor window.

    Log-space keeps the target ~O(1) (vs O(1e-2) return level), so the auxiliary
    MSE carries real gradient weight instead of being numerically starved. The
    slice auto-clamps near a shard's tail; the <2-obs fallback to the anchor
    window is a guard that effectively never fires under the emit bounds.
    """
    end = start_idx + seq_len
    fut = data[end : end + horizon, RETURN_FEATURE_INDEX].astype(np.float64)
    if fut.shape[0] < 2:
        fut = data[start_idx:end, RETURN_FEATURE_INDEX].astype(np.float64)
    return math.log(float(np.std(fut)) + _LOG_VOL_FLOOR)


# Yahoo non-equity tickers: crypto trades 7 days/week, FX/futures and indices use
# their own calendars — none align positionally to the SPY trading calendar.
_NON_EQUITY_SUFFIXES = ("-USD", "=X", "=F")


def _is_spy_calendar_symbol(stem: str) -> bool:
    """True when a shard follows the SPY/NYSE trading calendar (i.e. an equity).

    Crypto/FX/futures rows don't correspond date-for-date to the SPY-derived
    regime labels, so positional alignment is invalid for them.
    """
    if stem.startswith("^"):
        return False
    return not any(stem.endswith(suf) for suf in _NON_EQUITY_SUFFIXES)


def _global_label_offset(n_rows: int, n_label_rows: int) -> int | None:
    """Right-alignment offset mapping a shard row to a global SPY-calendar label.

    All real shards end on the shared train cutoff, so the last shard row aligns
    to the last label row: shard row i ↔ label row (i + n_label_rows - n_rows).
    The offset is negative when a shard has more history than SPY's own shard —
    its earliest rows predate the label calendar and map to negative indices,
    which callers mask out. Returns None only when no labels exist. Callers must
    gate on `_is_spy_calendar_symbol` first to exclude non-equity calendars.
    """
    if n_label_rows <= 0:
        return None
    return n_label_rows - n_rows


# ── HMM regime labeling ──────────────────────────────────────────────────────


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


def _canonicalize_hmm3(
    hmm: object, raw_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw 3-state HMM integers to canonical IDs: 0=bull, 1=hi-vol, 2=mean-rev.

    Sorted by mean |return|: lowest → bull, highest → hi-vol, middle → mean-rev.
    """
    means = hmm.means_  # type: ignore[attr-defined]
    mean_abs = means[:, 1]
    if len(mean_abs) != 3:
        raise ValueError(f"_canonicalize_hmm3 expects 3 states, got {len(mean_abs)}")

    abs_order = np.argsort(mean_abs)
    remap = np.empty(3, dtype=np.int64)
    remap[abs_order[0]] = 0
    remap[abs_order[2]] = 1
    remap[abs_order[1]] = 2

    return remap, remap[raw_labels]


def build_spy_regime_labels(
    data_dir: Path,
    crisis_percentile: float = 95.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit 4-class regime labels on SPY returns for train and val splits.

    Crisis (id=3): |ret| ≥ training-set percentile threshold (no look-ahead).
    Remaining rows: 3-state Gaussian HMM → bull / hi-vol / mean-rev.
    Returns (train_labels, val_labels) or (None, None) if SPY shard is absent.
    """
    from hmmlearn.hmm import GaussianHMM

    spy_train = data_dir / "train" / "SPY.f32bin"
    spy_val = data_dir / "validation" / "SPY.f32bin"
    cache_train = data_dir / "spy_regime_labels_train.npy"
    cache_val = data_dir / "spy_regime_labels_val.npy"

    if not spy_train.exists():
        return None, None

    if cache_train.exists():
        cached = np.load(cache_train)
        raw_check = np.memmap(spy_train, dtype="float32", mode="r")
        expected = raw_check.size // TOTAL_FEATURE_DIM
        if cached.shape[0] == expected:
            val_labels = np.load(cache_val) if cache_val.exists() else None
            return cached, val_labels

    train_returns = _shard_returns(spy_train)
    abs_train = train_returns[:, 1]
    crisis_threshold = float(np.percentile(abs_train, crisis_percentile))
    train_crisis = abs_train >= crisis_threshold

    train_non_crisis_idx = np.where(~train_crisis)[0]
    train_non_crisis_returns = train_returns[train_non_crisis_idx]
    hmm = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=500,
        random_state=42,
        verbose=False,
    )
    hmm.fit(train_non_crisis_returns)
    raw_train_nc = hmm.predict(train_non_crisis_returns)
    remap, canonical_nc = _canonicalize_hmm3(hmm, raw_train_nc)

    train_labels = np.empty(len(train_returns), dtype=np.int64)
    train_labels[train_non_crisis_idx] = canonical_nc
    train_labels[train_crisis] = 3
    np.save(cache_train, train_labels)

    counts = np.bincount(train_labels, minlength=4)
    print(
        f"Regime labels fit on {len(train_labels)} SPY rows "
        f"(crisis pctile={crisis_percentile}, |ret|≥{crisis_threshold:.4f}) — "
        f"bull={counts[0]}  hi-vol={counts[1]}  "
        f"mean-rev={counts[2]}  crisis={counts[3]}"
    )

    val_labels: np.ndarray | None = None
    if spy_val.exists():
        val_returns = _shard_returns(spy_val)
        val_crisis = val_returns[:, 1] >= crisis_threshold
        val_non_crisis_idx = np.where(~val_crisis)[0]
        val_labels = np.empty(len(val_returns), dtype=np.int64)
        if len(val_non_crisis_idx) > 0:
            raw_val_nc = hmm.predict(val_returns[val_non_crisis_idx])
            val_labels[val_non_crisis_idx] = remap[raw_val_nc]
        val_labels[val_crisis] = 3
        np.save(cache_val, val_labels)
        v_counts = np.bincount(val_labels, minlength=4)
        print(
            f"Val regime labels ({len(val_labels)} rows) — "
            f"bull={v_counts[0]}  hi-vol={v_counts[1]}  "
            f"mean-rev={v_counts[2]}  crisis={v_counts[3]}"
        )

    return train_labels, val_labels


# ── Data loading ─────────────────────────────────────────────────────────────


class ShardDataset(IterableDataset):
    """Streams TCP training tuples from .f32bin memory-mapped shards.

    Each .f32bin shard contains rows of shape (per_asset_feature_dim + market_dim)
    = 184 + 96 = 280 floats, stored as little-endian float32.

    For each anchor window W_a, the dataset emits:
      - adjacent positive window W_b (offset by `positive_offset` rows),
      - forward log realized-vol target for the auxiliary prediction loss,
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
        # Shards are right-aligned to the train cutoff, so row i maps to global
        # label row (i + offset). Without this offset, short-history shards
        # (recent IPOs) read the wrong regime. Synthetic and non-SPY-calendar
        # shards (offset None) fall back to the heuristic label.
        label_offset = (
            _global_label_offset(n_rows, n_label_rows)
            if (
                not is_synthetic
                and self.global_regime_labels is not None
                and _is_spy_calendar_symbol(path.stem)
            )
            else None
        )

        for start in indices:
            start_idx = int(start)
            pos_start = start_idx + self.positive_offset
            seq_a = data[start_idx : start_idx + self.seq_len]  # (T, F)
            seq_b = data[pos_start : pos_start + self.seq_len]  # (T, F)
            vol_target = next_window_log_vol(data, start_idx, self.seq_len)
            # Use HMM label at the last step of the anchor window so the
            # regime reflects the current market state at inference time.
            if label_offset is not None:
                label_idx = start_idx + self.seq_len - 1 + label_offset
                regime_id = (
                    int(self.global_regime_labels[label_idx])
                    if 0 <= label_idx < n_label_rows
                    else infer_regime_label(seq_a)
                )
            else:
                regime_id = infer_regime_label(seq_a)
            yield (
                seq_a,
                seq_b,
                vol_target,
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
                        vol_target,
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
                    torch.tensor(vol_target, dtype=torch.float32),
                    torch.tensor(asset_id, dtype=torch.long),
                    torch.tensor(regime_id, dtype=torch.long),
                    torch.tensor(sector_id, dtype=torch.long),
                    torch.tensor(sample_weight, dtype=torch.float32),
                    torch.tensor(time_index, dtype=torch.long),
                )
                next_active.append(path)
            active = next_active


# ── Regime-balanced sampling ─────────────────────────────────────────────────


def _shard_n_rows(path: Path) -> int:
    raw = np.memmap(path, dtype="float32", mode="r")
    return int(raw.size) // TOTAL_FEATURE_DIM


class RegimeBalancedShardDataset(IterableDataset):
    """Streams TCP tuples sampled uniformly across the 4 HMM regimes.

    Real shards end on the shared train cutoff, so each is right-aligned to the
    SPY label calendar (shard row i ↔ label row i + n_labels - shard_len) and
    bucketed per shard — short-history shards no longer truncate the reachable
    calendar. Synthetic shards are interleaved at `synthetic_fraction` to
    preserve their loss contribution.
    """

    PER_ASSET_DIM = PER_ASSET_FEATURE_DIM
    FEATURE_DIM = TOTAL_FEATURE_DIM
    MARKET_DIM = MARKET_FEATURE_DIM

    def __init__(
        self,
        shard_paths: list[Path],
        seq_len: int,
        positive_offset: int,
        global_regime_labels: np.ndarray,
        symbol_to_id: dict[str, int],
        symbol_to_sector: dict[str, int] | None = None,
        synthetic_prefix: str = SYNTHETIC_SHARD_PREFIX,
        synthetic_loss_weight: float = 0.5,
        synthetic_fraction: float = 0.2,
        n_regimes: int = 4,
        seed: int = 0,
    ) -> None:
        if positive_offset == 0:
            raise ValueError("positive_offset must be non-zero for TCP views")
        if global_regime_labels is None or len(global_regime_labels) == 0:
            raise ValueError("global_regime_labels required for regime balancing")

        self.seq_len = seq_len
        self.positive_offset = positive_offset
        self.global_regime_labels = global_regime_labels
        self.symbol_to_id = symbol_to_id
        self.symbol_to_sector = symbol_to_sector or {}
        self.synthetic_prefix = synthetic_prefix
        self.synthetic_loss_weight = synthetic_loss_weight
        self.synthetic_fraction = float(synthetic_fraction)
        self.n_regimes = n_regimes
        self.seed = seed

        self.real_paths = [
            p for p in shard_paths if not p.stem.startswith(synthetic_prefix)
        ]
        self.synthetic_paths = [
            p for p in shard_paths if p.stem.startswith(synthetic_prefix)
        ]
        if not self.real_paths:
            raise ValueError("RegimeBalancedShardDataset needs at least one real shard")

        # Per-shard, right-aligned regime bucketing. Each real shard ends on the
        # shared train cutoff, so shard row i maps to global label row
        # (i + n_label_rows - shard_len). Bucketing per shard (rather than over
        # the shortest shard's length, as a naive shared index would) ensures the
        # full calendar — including the rare mean-rev/crisis regimes that occur
        # outside any single short shard's window — is reachable. Each bucket
        # stores (shard_idx, start_idx) pairs so the sampled regime always
        # matches the window actually emitted.
        n_label_rows = len(global_regime_labels)
        min_start = max(0, -positive_offset)
        self.regime_real_paths: list[Path] = []
        bucket_shard: dict[int, list[np.ndarray]] = {r: [] for r in range(n_regimes)}
        bucket_start: dict[int, list[np.ndarray]] = {r: [] for r in range(n_regimes)}
        skipped_unaligned = 0
        for p in self.real_paths:
            if not _is_spy_calendar_symbol(p.stem):
                skipped_unaligned += 1  # crypto/FX/futures — not SPY-calendar
                continue
            n_rows = _shard_n_rows(p)
            offset = _global_label_offset(n_rows, n_label_rows)
            if offset is None:
                skipped_unaligned += 1
                continue
            max_start = n_rows - seq_len - max(positive_offset, 0) + 1
            if max_start <= min_start:
                continue
            starts = np.arange(min_start, max_start, dtype=np.int64)
            g_idx = starts + seq_len - 1 + offset
            in_range = (g_idx >= 0) & (g_idx < n_label_rows)
            starts = starts[in_range]
            if len(starts) == 0:
                continue
            labels = global_regime_labels[g_idx[in_range]].astype(np.int64)
            shard_idx = len(self.regime_real_paths)
            self.regime_real_paths.append(p)
            for r in range(n_regimes):
                sel = starts[labels == r]
                if len(sel):
                    bucket_shard[r].append(np.full(len(sel), shard_idx, dtype=np.int64))
                    bucket_start[r].append(sel)

        if not self.regime_real_paths:
            raise ValueError(
                "RegimeBalancedShardDataset: no SPY-calendar-aligned real shards "
                f"(seq_len={seq_len} offset={positive_offset})"
            )
        self.bucket_shard: dict[int, np.ndarray] = {
            r: (
                np.concatenate(bucket_shard[r])
                if bucket_shard[r]
                else np.empty(0, dtype=np.int64)
            )
            for r in range(n_regimes)
        }
        self.bucket_start: dict[int, np.ndarray] = {
            r: (
                np.concatenate(bucket_start[r])
                if bucket_start[r]
                else np.empty(0, dtype=np.int64)
            )
            for r in range(n_regimes)
        }
        self.present_regimes: list[int] = [
            r for r in range(n_regimes) if len(self.bucket_start[r]) > 0
        ]
        if not self.present_regimes:
            raise ValueError("no valid regime buckets — regime labels missing")
        empty_regimes = [r for r in range(n_regimes) if r not in self.present_regimes]
        if empty_regimes:
            print(
                f"RegimeBalancedShardDataset: regimes {empty_regimes} have 0 "
                f"windows in this split; sampling uniformly across "
                f"{self.present_regimes} only."
            )
        if skipped_unaligned:
            print(
                f"RegimeBalancedShardDataset: skipped {skipped_unaligned} non-equity "
                f"shard(s) (crypto/FX/futures) not on the SPY trading calendar — "
                f"excluded from regime sampling."
            )
        regime_names = ["bull", "hi-vol", "mean-rev", "crisis"]
        bucket_summary = "  ".join(
            f"{regime_names[r]}={len(self.bucket_start[r])}" for r in range(n_regimes)
        )
        print(f"RegimeBalancedShardDataset bucket sizes:  {bucket_summary}")

    def _load_rows(self, path: Path) -> np.ndarray:
        raw = np.memmap(path, dtype="float32", mode="r")
        n_floats = int(raw.size)
        if n_floats % self.FEATURE_DIM != 0:
            raise ValueError(
                f"shard {path} has unsupported row width "
                f"(floats={n_floats}, expected multiple of {self.FEATURE_DIM})"
            )
        return raw.reshape(-1, self.FEATURE_DIM)

    def _emit_real(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int, int, float, int]:
        regime_id = int(self.present_regimes[rng.integers(len(self.present_regimes))])
        shard_pool = self.bucket_shard[regime_id]
        start_pool = self.bucket_start[regime_id]
        j = int(rng.integers(len(start_pool)))
        # (shard, start) are paired so the window's true regime matches regime_id;
        # a random shard would break the right-aligned label correspondence.
        path = self.regime_real_paths[int(shard_pool[j])]
        start_idx = int(start_pool[j])
        data = self._load_rows(path)

        pos_start = start_idx + self.positive_offset
        seq_a = data[start_idx : start_idx + self.seq_len].copy()
        seq_b = data[pos_start : pos_start + self.seq_len].copy()
        vol_target = next_window_log_vol(data, start_idx, self.seq_len)
        asset_id = int(self.symbol_to_id.get(path.stem, 0))
        sector_id = int(self.symbol_to_sector.get(path.stem, UNKNOWN_SECTOR_ID))
        return (
            seq_a,
            seq_b,
            vol_target,
            asset_id,
            regime_id,
            sector_id,
            1.0,
            start_idx,
        )

    def _emit_synthetic(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int, int, float, int] | None:
        path = self.synthetic_paths[rng.integers(len(self.synthetic_paths))]
        data = self._load_rows(path)
        n_rows = data.shape[0]
        min_start = max(0, -self.positive_offset)
        max_start = n_rows - self.seq_len - max(self.positive_offset, 0) + 1
        if max_start <= min_start:
            return None
        start_idx = int(rng.integers(min_start, max_start))
        pos_start = start_idx + self.positive_offset
        seq_a = data[start_idx : start_idx + self.seq_len].copy()
        seq_b = data[pos_start : pos_start + self.seq_len].copy()
        vol_target = next_window_log_vol(data, start_idx, self.seq_len)
        regime_id = infer_regime_label(seq_a)
        return (
            seq_a,
            seq_b,
            vol_target,
            0,
            regime_id,
            UNKNOWN_SECTOR_ID,
            self.synthetic_loss_weight,
            start_idx,
        )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker.id if worker is not None else 0)
        rng = np.random.default_rng(worker_seed)
        while True:
            use_synth = (
                len(self.synthetic_paths) > 0 and rng.random() < self.synthetic_fraction
            )
            if use_synth:
                t = self._emit_synthetic(rng)
                if t is None:
                    continue
            else:
                t = self._emit_real(rng)
            (
                seq_a,
                seq_b,
                vol_target,
                asset_id,
                regime_id,
                sector_id,
                sample_weight,
                start_idx,
            ) = t
            yield (
                torch.from_numpy(seq_a),
                torch.from_numpy(seq_b),
                torch.tensor(vol_target, dtype=torch.float32),
                torch.tensor(asset_id, dtype=torch.long),
                torch.tensor(regime_id, dtype=torch.long),
                torch.tensor(sector_id, dtype=torch.long),
                torch.tensor(sample_weight, dtype=torch.float32),
                torch.tensor(start_idx, dtype=torch.long),
            )


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

    def _iter_tar(
        self, source: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float, int, int, int, float, int]]:
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
                sample_weight = self.synthetic_loss_weight if is_synthetic else 1.0
                # +1 lets exact-length shards yield one pair (see _iter_shard).
                n_rows = data.shape[0]
                min_start = max(0, -self.positive_offset)
                max_start = n_rows - self.seq_len - max(self.positive_offset, 0) + 1
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
                    vol_target = next_window_log_vol(data, start_idx, self.seq_len)
                    regime_id = infer_regime_label(seq_a)
                    yield (
                        seq_a,
                        seq_b,
                        vol_target,
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
                vol_target,
                asset_id,
                regime_id,
                sector_id,
                sample_weight,
                time_index,
            ) in self._iter_tar(src):
                yield (
                    torch.from_numpy(seq_a.copy()),
                    torch.from_numpy(seq_b.copy()),
                    torch.tensor(vol_target, dtype=torch.float32),
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


def build_regime_balanced_loader(
    shard_paths: list[Path],
    seq_len: int,
    batch_size: int,
    positive_offset: int,
    global_regime_labels: np.ndarray,
    symbol_to_id: dict[str, int],
    symbol_to_sector: dict[str, int] | None = None,
    synthetic_loss_weight: float = 0.5,
    synthetic_fraction: float = 0.2,
    num_workers: int = 4,
    seed: int = 0,
) -> DataLoader:
    ds = RegimeBalancedShardDataset(
        shard_paths,
        seq_len=seq_len,
        positive_offset=positive_offset,
        global_regime_labels=global_regime_labels,
        symbol_to_id=symbol_to_id,
        symbol_to_sector=symbol_to_sector,
        synthetic_loss_weight=synthetic_loss_weight,
        synthetic_fraction=synthetic_fraction,
        seed=seed,
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
    regime_balanced: bool = False,
    synthetic_fraction: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build train/val DataLoaders from {data_dir}/train/ and {data_dir}/validation/.

    Fits HMM regime labels on the train split and assigns
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
            "Run 'make collect' and ensure SPY is in portfolio_symbols.txt."
        )

    if regime_balanced and train_regime_labels is not None:
        train_loader = build_regime_balanced_loader(
            train_shards,
            seq_len=seq_len,
            batch_size=batch_size,
            positive_offset=positive_offset,
            global_regime_labels=train_regime_labels,
            symbol_to_id=symbol_to_id,
            symbol_to_sector=symbol_to_sector,
            synthetic_loss_weight=synthetic_loss_weight,
            synthetic_fraction=synthetic_fraction,
            num_workers=num_workers,
        )
        print(
            "Train loader: regime-balanced sampling enabled "
            f"(target ~{1.0 / 4:.0%} per regime, synthetic fraction "
            f"{synthetic_fraction:.0%})"
        )
    else:
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

    # Regime-stratified loader from train data for cluster/t-SNE evaluation.
    # Val may lack rare regimes; this guarantees all 4 are represented.
    diagnostic_loader: DataLoader | None = None
    if train_regime_labels is not None:
        diagnostic_loader = build_regime_balanced_loader(
            train_shards,
            seq_len=seq_len,
            batch_size=batch_size,
            positive_offset=positive_offset,
            global_regime_labels=train_regime_labels,
            symbol_to_id=symbol_to_id,
            symbol_to_sector=symbol_to_sector,
            synthetic_loss_weight=synthetic_loss_weight,
            synthetic_fraction=0.0,  # no synth for diagnostic — real regimes only
            num_workers=max(1, num_workers // 2),
            seed=12345,
        )

    return train_loader, val_loader, diagnostic_loader


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
    require_same_sector: bool = True,
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

    if asset_ids is not None and regime_ids is not None and return_windows is not None:
        if sector_ids is not None and (
            sector_ids.ndim != 1 or sector_ids.shape[0] != z_a.shape[0]
        ):
            raise ValueError("sector_ids must be rank-1 and batch-aligned")
        corr = _pairwise_return_correlation(return_windows)
        same_regime = regime_ids[:, None] == regime_ids[None, :]
        diff_asset = asset_ids[:, None] != asset_ids[None, :]
        additional_positives = same_regime & diff_asset & (corr > correlation_threshold)
        if require_same_sector and sector_ids is not None:
            known_sector = sector_ids >= 0
            same_sector = sector_ids[:, None] == sector_ids[None, :]
            additional_positives = (
                additional_positives
                & known_sector[:, None]
                & known_sector[None, :]
                & same_sector
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
    require_same_sector: bool = True,
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
        require_same_sector=require_same_sector,
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


def regime_supervised_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    regime_ids: torch.Tensor,
    temperature: float = 0.1,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervised-contrastive (SupCon) term grouping windows by HMM regime.

    Instance-discrimination contrastive learning (one positive per anchor)
    produces no regime geometry. SupCon (Khosla et al. 2020) treats every
    same-regime window in the batch as a positive, supplying a dense
    regime-grouping gradient. It groups by regime, not asset, so it does not
    fight the adversarial head's asset-invariance.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = torch.cat([z_a, z_b], dim=0).to(torch.float32)
    z = F.normalize(z, dim=-1)
    labels = torch.cat([regime_ids, regime_ids], dim=0)
    n = z.shape[0]
    logits = (z @ z.T) / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    # Exclude self from both numerator and denominator (exp(-inf)=0).
    logits = logits.masked_fill(self_mask, float("-inf"))
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    # masked_fill (not multiply): log_prob is -inf at self, and -inf * 0 = NaN.
    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)
    per_anchor = -pos_log_prob / pos_count.clamp_min(1)
    valid = (pos_count > 0).to(torch.float32)
    if sample_weights is not None:
        w = torch.cat([sample_weights, sample_weights], dim=0).to(torch.float32) * valid
        return (per_anchor * w).sum() / w.sum().clamp_min(1e-8)
    return (per_anchor * valid).sum() / valid.sum().clamp_min(1.0)


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
    contrastive_weight: float = 0.6
    prediction_weight: float = 0.2
    supcon_weight: float = 0.2
    positive_pair_offset: int = 4
    fda_loss_weight: float = 0.5
    hard_negative_boost: float = 0.25
    multi_asset_corr_threshold: float = 0.3
    negative_min_separation: int = 50
    require_same_sector: bool = True
    # Asset-adversarial loss to break the asset-identity shortcut.
    # Schedule ramps λ from 0 → adversarial_lambda over `adversarial_warmup_steps`.
    adversarial_lambda: float = 0.0
    adversarial_warmup_steps: int = 2_000
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
        adversarial_n_assets: int | None = None,
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
        # n_assets controls the portfolio state tensor width and must match the
        # trading universe (spec.n_assets).  adversarial_n_assets sizes the
        # gradient-reversal head and must match the number of unique symbols in
        # the pretraining data — these diverge when pretraining on 500 symbols
        # but the portfolio only trades 100.  When None, defaults to n_assets
        # (backward compatible for same-size pretrain/portfolio universes).
        self.adversarial_n_assets = (
            adversarial_n_assets if adversarial_n_assets is not None else n_assets
        )
        self.adversarial_head = AssetAdversarialHead(
            d_model=d_model, n_assets=self.adversarial_n_assets
        ).to(self.device)
        self.n_assets = n_assets

        self._all_modules = nn.ModuleList(
            [
                backbone,
                fusion,
                regime_detector,
                input_projection,
                self.projection_head,
                self.prediction_head,
                self.adversarial_head,
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
            vol_targets,
            asset_ids,
            regime_ids,
            sector_ids,
            sample_weights,
            time_indices,
        ) = batch
        return (
            features,
            positives,
            vol_targets,
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
            "adversarial_head": self.adversarial_head.state_dict(),
        }

    def _load_model_state(self, state: dict) -> None:
        backbone_mod = getattr(self.backbone, "_orig_mod", self.backbone)
        backbone_mod.load_state_dict(state["backbone"])
        self.fusion.load_state_dict(state["fusion"])
        self.regime_detector.load_state_dict(state["regime_detector"])
        self.input_projection.load_state_dict(state["input_projection"])
        self.projection_head.load_state_dict(state["projection_head"])
        self.prediction_head.load_state_dict(state["prediction_head"])
        # Adversarial head is new; older checkpoints don't have it. Init from
        # scratch in that case rather than failing the load.
        adv_state = state.get("adversarial_head")
        if adv_state is not None:
            try:
                self.adversarial_head.load_state_dict(adv_state)
            except RuntimeError as e:
                print(f"adversarial_head state mismatch ({e}); initializing fresh.")

    def evaluate(
        self,
        val_loader: DataLoader,
        cfg: TrainConfig,
        max_batches: int | None = None,
        diagnostic_loader: DataLoader | None = None,
    ) -> dict[str, float]:
        self._all_modules.eval()
        max_eval_batches = cfg.eval_max_batches if max_batches is None else max_batches
        totals: dict[str, float] = {
            "contrastive": 0.0,
            "supcon": 0.0,
            "prediction": 0.0,
            "adversarial": 0.0,
            "adversarial_acc": 0.0,
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
                    vol_targets,
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
                        vol_targets=vol_targets,
                        asset_ids=asset_ids,
                        regime_ids=regime_ids,
                        sector_ids=sector_ids,
                        sample_weights=sample_weights,
                        time_indices=time_indices,
                        temperature=cfg.contrastive_temperature,
                        contrastive_weight=cfg.contrastive_weight,
                        prediction_weight=cfg.prediction_weight,
                        supcon_weight=cfg.supcon_weight,
                        hard_negative_boost=cfg.hard_negative_boost,
                        correlation_threshold=cfg.multi_asset_corr_threshold,
                        min_negative_gap=cfg.negative_min_separation,
                        require_same_sector=cfg.require_same_sector,
                        adversarial_lambda=cfg.adversarial_lambda,
                        return_aux=True,
                    )
                for k in (
                    "contrastive",
                    "supcon",
                    "prediction",
                    "adversarial",
                    "adversarial_acc",
                    "total",
                    "contrastive_acc",
                ):
                    v = losses[k]
                    totals[k] += v.item()
                pred_values.append(losses["pred_logvol"].detach().cpu())
                target_values.append(losses["target_logvol"].detach().cpu())
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
                f"  val logvol  mean={pred.mean().item():+.3e} std={pred.std().item():.3e}  "
                f"target mean={target.mean().item():+.3e} std={target.std().item():.3e}  "
                f"SSE={sse.item():.4f}  SST={sst.item():.4f}"
            )
        else:
            metrics["prediction_r2"] = 0.0

        # Diagnostic gates — separate methods keep the hot eval loop lean.
        metrics["swa_entropy"] = self.eval_swa_entropy(
            val_loader, max_batches=min(max_eval_batches, 10)
        )
        # Cluster silhouette runs on the regime-stratified diagnostic loader
        # when available — val (calendar 2023) has 0% hi-vol/crisis and cannot
        # show 4 clusters regardless of model quality.
        sep_loader = diagnostic_loader if diagnostic_loader is not None else val_loader
        metrics["regime_separation"] = self.eval_regime_separation(
            sep_loader, max_batches=min(max_eval_batches, 20)
        )
        return metrics

    def eval_swa_entropy(
        self,
        val_loader: DataLoader,
        max_batches: int = 10,
    ) -> float:
        """Mean normalized SWA attention entropy over val batches (gate: < 0.80)."""
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
        """Silhouette of backbone embeddings against true regime labels (gate: > 0.10).

        Scores separation of the actual regimes, not generic k-means clusterability.
        """
        from sklearn.metrics import silhouette_score

        self._all_modules.eval()
        all_embs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                unpacked = self._unpack_batch(batch)
                features = unpacked[0].to(self.device)
                regime_ids = unpacked[4]
                emb = self._encode_sequence(features)  # (B, D)
                all_embs.append(F.normalize(emb.float(), dim=-1).cpu())
                all_labels.append(regime_ids.cpu())
        self._all_modules.train()

        if not all_embs:
            return float("nan")

        embs_np = torch.cat(all_embs, dim=0).numpy()  # (N, D)
        labels_np = torch.cat(all_labels, dim=0).numpy()  # (N,)
        # silhouette needs ≥2 regimes present and ≥1 extra sample beyond label count.
        if len(embs_np) < 8 or len(np.unique(labels_np)) < 2:
            return float("nan")
        return float(
            silhouette_score(
                embs_np,
                labels_np,
                metric="cosine",
                sample_size=min(500, len(embs_np)),
                random_state=0,
            )
        )

    def eval_tsne_plot(
        self,
        val_loader: DataLoader,
        step: int,
        reports_dir: Path,
        max_batches: int = 20,
    ) -> Path | None:
        """t-SNE scatter coloured by regime label. Saves a PNG to reports_dir."""
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
        coords = TSNE(
            n_components=2, perplexity=perplexity, random_state=0
        ).fit_transform(embs_np)

        colors = ["#2ecc71", "#e67e22", "#3498db", "#e74c3c"]
        regime_labels = ["bull", "hi-vol", "mean-rev", "crisis"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for rid in range(4):
            mask = regimes_np == rid
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[rid],
                label=regime_labels[rid],
                alpha=0.6,
                s=12,
                linewidths=0,
            )
        ax.legend(title="regime", fontsize=8)
        ax.set_title(f"Backbone t-SNE — step {step:,}  (cluster gate)")
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
        """Encode a temporal window to a pooled representation: z = MLP(pool(backbone(W))).

        Portfolio and calendar inputs are zeros during pretraining; only used after head training.
        """
        B, T, _ = features.shape
        asset_feats = features[..., :184].unsqueeze(2)  # (B, T, 1, 184)
        market_feats = features[..., 184:]  # (B, T, 96)

        n_assets = 1
        # Portfolio state is zeros during pretraining; width must match the
        # InputProjection's portfolio_proj (= 2 * max-universe + 1).
        portfolio_s = torch.zeros(B, 2 * self.n_assets + 1, device=self.device)
        portfolio_summary = torch.zeros(B, 3, device=self.device)
        calendar = torch.zeros(B, 20, device=self.device)
        states = self.backbone.init_state(B, n_assets, self.device)

        # Per-step regime detection. ema_alpha=0.95 → ~14-step half-life.
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
        posteriors_t = torch.stack(posteriors_list, dim=1)  # (B, T, 4)

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
            asset_tok = asset_tok_all[:, t]  # (B, 1, D)
            re_t = regime_embs_t[:, t]
            post_t = posteriors_t[:, t]
            out, states = self.backbone(asset_tok, re_t, post_t, states)
            pooled_steps.append(out.squeeze(1))  # (B, D)

        pooled = torch.stack(pooled_steps, dim=1).mean(dim=1)  # (B, D)
        if return_states:
            return pooled, states
        return pooled

    def _forward_losses(
        self,
        features: torch.Tensor,  # (B, T, 280) — per-asset (184) + market (96)
        positives: torch.Tensor,  # (B, T, 280) — temporally adjacent window
        vol_targets: torch.Tensor | None = None,  # (B,) forward log realized-vol target
        asset_ids: torch.Tensor | None = None,
        regime_ids: torch.Tensor | None = None,
        sector_ids: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        time_indices: torch.Tensor | None = None,
        temperature: float = 0.1,
        contrastive_weight: float = 0.7,
        prediction_weight: float = 0.3,
        supcon_weight: float = 0.0,
        hard_negative_boost: float = 0.25,
        correlation_threshold: float = 0.6,
        min_negative_gap: int = 50,
        require_same_sector: bool = True,
        adversarial_lambda: float = 0.0,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = features.to(self.device)
        positives = positives.to(self.device)
        if vol_targets is None:
            # Fallback (no dataset target): log realized-vol of the positive window.
            window_returns = positives[:, :, RETURN_FEATURE_INDEX].to(torch.float32)
            vol_targets = torch.log(window_returns.std(dim=1) + _LOG_VOL_FLOOR)
        vol_targets = vol_targets.to(self.device).to(torch.float32)
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
            require_same_sector=require_same_sector,
            sample_weights=sample_weights,
        )

        # Supervise the log_vol head output (index 1); the return output (index 0)
        # is a martingale and stays unsupervised during pretraining.
        _, pred_logvol_raw = self.prediction_head(view_a)
        pred_logvol = pred_logvol_raw.to(torch.float32)
        pred_errors = (pred_logvol - vol_targets) ** 2

        if sample_weights is None:
            loss_prediction = pred_errors.mean()
        else:
            denom = sample_weights.sum().clamp_min(1e-8)
            loss_prediction = (pred_errors * sample_weights).sum() / denom

        if adversarial_lambda > 0.0 and asset_ids is not None:
            asset_logits = self.adversarial_head(view_a, lambda_=adversarial_lambda)
            adv_targets = asset_ids.clamp(0, self.adversarial_n_assets - 1)
            adv_per_sample = F.cross_entropy(
                asset_logits.float(), adv_targets, reduction="none"
            )
            if sample_weights is None:
                loss_adversarial = adv_per_sample.mean()
            else:
                denom = sample_weights.sum().clamp_min(1e-8)
                loss_adversarial = (adv_per_sample * sample_weights).sum() / denom
            adv_acc = (
                (asset_logits.argmax(dim=-1) == adv_targets).float().mean().detach()
            )
        else:
            loss_adversarial = torch.zeros((), device=self.device, dtype=torch.float32)
            adv_acc = torch.zeros((), device=self.device, dtype=torch.float32)

        if supcon_weight > 0.0 and regime_ids is not None:
            # SupCon shapes the backbone embeddings, not the projection head, so
            # its regime-grouping gradient reaches the space the gate measures.
            loss_supcon = regime_supervised_contrastive_loss(
                view_a,
                view_b,
                regime_ids,
                temperature=temperature,
                sample_weights=sample_weights,
            )
        else:
            loss_supcon = torch.zeros((), device=self.device, dtype=torch.float32)

        loss_total = (
            contrastive_weight * loss_contrastive
            + supcon_weight * loss_supcon
            + prediction_weight * loss_prediction
            + loss_adversarial
        )
        out: dict[str, torch.Tensor] = {
            "contrastive": loss_contrastive,
            "supcon": loss_supcon,
            "prediction": loss_prediction,
            "adversarial": loss_adversarial,
            "adversarial_acc": adv_acc,
            "total": loss_total,
            "contrastive_acc": contrastive_acc,
        }
        if return_aux:
            out["pred_logvol"] = pred_logvol
            out["target_logvol"] = vol_targets
        return out

    def train(
        self,
        train_loader: DataLoader,
        cfg: TrainConfig,
        val_loader: DataLoader | None = None,
        resume_from: Path | None = None,
        diagnostic_loader: DataLoader | None = None,
    ) -> None:
        # Prediction head now learns forward realized-vol (a genuinely learnable,
        # regime-defining target), so it runs at the backbone lr rather than the
        # old 10× reduction that pinned it near a constant for the martingale return.
        pred_head_params = list(self.prediction_head.parameters())
        pred_head_ids = {id(p) for p in pred_head_params}
        # Fusion is only active during portfolio training; exclude from optimizer
        # so weight decay doesn't pull it toward zero during pretraining.
        fusion_param_ids = {id(p) for p in self.fusion.parameters()}
        adv_head_params = list(self.adversarial_head.parameters())
        adv_head_ids = {id(p) for p in adv_head_params}
        other_params = [
            p
            for p in self._all_modules.parameters()
            if id(p) not in pred_head_ids
            and id(p) not in fusion_param_ids
            and id(p) not in adv_head_ids
        ]
        # Gradient-reversal makes the adversarial head a feedback loop
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": cfg.lr},
                {"params": pred_head_params, "lr": cfg.lr},
                {"params": adv_head_params, "lr": cfg.lr * 0.2, "weight_decay": 0.0},
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
                vol_targets,
                asset_ids,
                regime_ids,
                sector_ids,
                sample_weights,
                time_indices,
            ) = self._unpack_batch(batch)

            # Adversarial λ ramps from 0 → cfg.adversarial_lambda over warmup
            # so early optimization isn't destabilised by a strong reversed
            # gradient before the encoder has any structure to fight against.
            if cfg.adversarial_lambda > 0.0 and cfg.adversarial_warmup_steps > 0:
                ramp = min(1.0, step / max(cfg.adversarial_warmup_steps, 1))
                adv_lambda_now = cfg.adversarial_lambda * ramp
            else:
                adv_lambda_now = cfg.adversarial_lambda

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                losses = self._forward_losses(
                    features,
                    positives,
                    vol_targets=vol_targets,
                    asset_ids=asset_ids,
                    regime_ids=regime_ids,
                    sector_ids=sector_ids,
                    sample_weights=sample_weights,
                    time_indices=time_indices,
                    temperature=cfg.contrastive_temperature,
                    contrastive_weight=cfg.contrastive_weight,
                    prediction_weight=cfg.prediction_weight,
                    supcon_weight=cfg.supcon_weight,
                    hard_negative_boost=cfg.hard_negative_boost,
                    correlation_threshold=cfg.multi_asset_corr_threshold,
                    min_negative_gap=cfg.negative_min_separation,
                    require_same_sector=cfg.require_same_sector,
                    adversarial_lambda=adv_lambda_now,
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
                    for k in (
                        "contrastive",
                        "supcon",
                        "prediction",
                        "adversarial",
                        "adversarial_acc",
                        "total",
                        "contrastive_acc",
                    )
                )
                print(
                    f"step={step:>7d}  {parts}  "
                    f"adv_lambda={adv_lambda_now:.3f}  lr={lr_now:.2e}"
                )

            if val_loader is not None and step % cfg.eval_every == 0:
                val_metrics = self.evaluate(
                    val_loader, cfg, diagnostic_loader=diagnostic_loader
                )
                val_loss = val_metrics["total"]
                val_contrastive_acc = val_metrics["contrastive_acc"]
                val_pred_r2 = val_metrics["prediction_r2"]
                # Track the best contrastive acc for reporting/checkpoint
                if val_contrastive_acc > (
                    best_val_contrastive_acc + cfg.early_stop_min_delta
                ):
                    best_val_contrastive_acc = val_contrastive_acc

                if val_loss < best_val_loss - cfg.early_stop_min_delta:
                    best_val_loss = val_loss
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
                            "Early stopping on validation loss "
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

            tsne_source = (
                diagnostic_loader if diagnostic_loader is not None else val_loader
            )
            if (
                tsne_source is not None
                and cfg.tsne_every > 0
                and step % cfg.tsne_every == 0
            ):
                self.eval_tsne_plot(
                    tsne_source,
                    step,
                    cfg.tsne_reports_dir,
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
