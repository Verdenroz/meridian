"""Portfolio optimization training — Meridian-R and Meridian-S heads.

Trains the two portfolio output heads on top of the pre-trained backbone.

Meridian-R (return head):
  Backbone staged fine-tuning. Loss = -DSR + λ_ret·Return + λ_turn·Turnover + λ_vol·VolAux.
  ~50K steps.

Meridian-S (Sharpe head):
  Backbone staged fine-tuning (same schedule as return head). Loss = -DSR.
  ~30K steps.

Usage:

    # Train return head (loads pre-trained backbone checkpoint)
    python -m train.portfolio_train --head return \\
        --data-dir data/mmaps \\
        --resume-backbone checkpoints/pretrain/backbone_pretrain.pt \\
        --steps 50000 \\
        --checkpoint-dir checkpoints/portfolio

    # Then train Sharpe head (loads return head checkpoint)
    python -m train.portfolio_train --head sharpe \\
        --data-dir data/mmaps \\
        --resume-backbone checkpoints/portfolio/return_head_best.pt \\
        --steps 30000 \\
        --checkpoint-dir checkpoints/portfolio

    # Smoke test
    python -m train.portfolio_train --head return --smoke --steps 3
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import struct
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _activation_checkpoint
from core.cross_asset_fusion import CrossAssetFusion
from core.regime_aware_backbone import InputProjection, RAGDNBackbone, RegimeDetector
from core.self_supervised_pretraining import (
    Checkpoint,
    infer_regime_label,
    load_checkpoint,
    save_checkpoint,
)
from optimization.losses import dsr_loss, return_head_loss
from optimization.portfolio_heads import (
    AuxHeadSurprise,
    MeridianR,
    MeridianS,
    PortfolioVolHead,
)
from torch.utils.data import DataLoader, IterableDataset

from train.feature_schema import BLOCK_PRICE_VOLUME, BLOCK_TECHNICAL
from train.spec import DEFAULT_SPEC, MeridianSpec

# ── Multi-asset dataset ───────────────────────────────────────────────────────

_PTA_RETURN_CHANNELS = (0, 3, 5, 6, 7, 8, 9)
_PTA_TECH_MA_DIM = 8
_PTA_TECH_MOMENTUM_DIM = 11
_PTA_TECH_VOL_START = BLOCK_TECHNICAL.start + _PTA_TECH_MA_DIM + _PTA_TECH_MOMENTUM_DIM
_PTA_TECH_VOL_DIM = 5
_PTA_VOLATILITY_CHANNELS = tuple(
    range(_PTA_TECH_VOL_START, _PTA_TECH_VOL_START + _PTA_TECH_VOL_DIM)
)
_PTA_PERMUTED_FEATURE_SLICE = slice(BLOCK_PRICE_VOLUME.start, BLOCK_TECHNICAL.end)
_PTA_PERMUTED_FEATURE_DIM = BLOCK_TECHNICAL.end - BLOCK_PRICE_VOLUME.start
_SYNTH_PREFIX = "FDA_SYNTH_"
_SECTOR_HEADER_RE = re.compile(r"([A-Za-z][A-Za-z\s&/\-]+)\s*\(\d+\)")
_GICS_SECTOR_NAMES = (
    "Information Technology",
    "Financials",
    "Health Care",
    "Industrials",
    "Consumer Discretionary",
    "Communication Services",
    "Consumer Staples",
    "Energy",
    "Materials",
    "Real Estate",
    "Utilities",
)
_SECTOR_NAME_TO_ID = {name: idx for idx, name in enumerate(_GICS_SECTOR_NAMES)}


def load_symbol_sector_map(symbols_file: Path) -> dict[str, int]:
    """Parse symbols.txt-style section headers into symbol→sector-id mapping.

    Only known 11 GICS-equity sections are mapped. Non-equity sections (e.g. ETFs)
    are intentionally left unmapped.
    """
    symbol_to_sector: dict[str, int] = {}
    current_sector: int | None = None

    for raw_line in symbols_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _SECTOR_HEADER_RE.search(line)
            if m is not None:
                section_name = " ".join(m.group(1).split())
                current_sector = _SECTOR_NAME_TO_ID.get(section_name)
            continue

        symbol = line.split("#", 1)[0].strip()
        if not symbol:
            continue
        if current_sector is not None:
            symbol_to_sector[symbol] = current_sector

    return symbol_to_sector


def build_asset_sector_ids(
    shard_paths: list[Path],
    symbol_to_sector: dict[str, int],
) -> list[int]:
    """Build sector-id list aligned to shard_paths order.

    Unknown symbols receive unique negative ids so they remain fixed during
    sector-preserving permutations.
    """
    sector_ids: list[int] = []
    next_unknown_group = -1
    for p in shard_paths:
        symbol = p.stem
        sid = symbol_to_sector.get(symbol)
        if sid is None:
            sector_ids.append(next_unknown_group)
            next_unknown_group -= 1
        else:
            sector_ids.append(int(sid))
    return sector_ids


@dataclass(frozen=True)
class PTAAugmentationConfig:
    """PTA controls aligned with Section 11.4b defaults."""

    apply_prob: float = 0.5
    permutations_per_batch: int = 4
    sign_flip_prob: float = 0.5
    vol_scale_prob: float = 0.7
    vol_scale_values: tuple[float, ...] = (0.8, 0.9, 1.1, 1.2)

    def validate(self) -> None:
        if not (0.0 <= self.apply_prob <= 1.0):
            raise ValueError(f"pta apply_prob must be in [0,1], got {self.apply_prob}")
        if self.permutations_per_batch < 0:
            raise ValueError(
                "pta permutations_per_batch must be >= 0, "
                f"got {self.permutations_per_batch}"
            )
        if not (0.0 <= self.sign_flip_prob <= 1.0):
            raise ValueError(
                f"pta sign_flip_prob must be in [0,1], got {self.sign_flip_prob}"
            )
        if not (0.0 <= self.vol_scale_prob <= 1.0):
            raise ValueError(
                f"pta vol_scale_prob must be in [0,1], got {self.vol_scale_prob}"
            )
        if not self.vol_scale_values:
            raise ValueError("pta vol_scale_values must contain at least one scale")
        if any(scale <= 0.0 for scale in self.vol_scale_values):
            raise ValueError(
                f"pta vol_scale_values must be positive, got {self.vol_scale_values}"
            )


def _draw_event(
    prob: float,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> bool:
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    return torch.rand((), generator=generator, device=device).item() < prob


def _scale_return_channels_inplace(
    features: torch.Tensor,
    factor: torch.Tensor | float,
) -> None:
    for ch in _PTA_RETURN_CHANNELS:
        features[..., ch] = features[..., ch] * factor


def _scale_volatility_channels_inplace(
    features: torch.Tensor,
    factor: torch.Tensor | float,
) -> None:
    # Technical volatility slots include ATR and Bollinger %B; scale with returns
    # for PTA vol-scaling consistency (spec §11.4b, transformation 3).
    for ch in _PTA_VOLATILITY_CHANNELS:
        features[..., ch] = features[..., ch] * factor


def apply_pta_permutation(
    features: torch.Tensor,
    returns: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asset-permutation PTA view for single-batch tensors.

    Per Risk #13 mitigation, only Blocks 1–2 (price/volume + technical) are
    permuted across assets; later per-asset blocks remain slot-stable.
    """
    n_assets = returns.shape[0]
    perm = torch.randperm(n_assets, generator=generator, device=returns.device)
    permuted = features.clone()
    permuted[..., _PTA_PERMUTED_FEATURE_SLICE] = features[perm][
        ..., _PTA_PERMUTED_FEATURE_SLICE
    ]
    return permuted, returns[perm], perm


def apply_pta_batch_augmentations(
    features: torch.Tensor,
    returns: torch.Tensor,
    generator: torch.Generator | None = None,
    cfg: PTAAugmentationConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTA transforms used in portfolio head training.

    1. Apply with probability p_aug (spec default: 0.5)
    2. Asset permutation
    3. Optional sign-coupled return flips
    4. Optional volatility scaling
    """
    cfg = cfg or PTAAugmentationConfig()
    cfg.validate()

    if not _draw_event(cfg.apply_prob, device=returns.device, generator=generator):
        return features, returns

    features, returns, _ = apply_pta_permutation(features, returns, generator=generator)

    if _draw_event(cfg.sign_flip_prob, device=returns.device, generator=generator):
        returns = -returns
        _scale_return_channels_inplace(features, -1.0)

    if (
        cfg.vol_scale_prob > 0.0
        and cfg.vol_scale_values
        and _draw_event(cfg.vol_scale_prob, device=returns.device, generator=generator)
    ):
        scale_idx = int(
            torch.randint(
                len(cfg.vol_scale_values),
                (1,),
                generator=generator,
                device=returns.device,
            ).item()
        )
        scale = returns.new_tensor(cfg.vol_scale_values[scale_idx])
        returns = returns * scale
        _scale_return_channels_inplace(features, scale)
        _scale_volatility_channels_inplace(features, scale)

    return features, returns


def _permute_assets_batch(
    features: torch.Tensor,
    returns: torch.Tensor,
    sector_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one random asset permutation per batch item.

    Args:
        features: (B, N, T, F)
        returns:  (B, N, T)
        sector_ids: (N,) optional sector group id per asset; non-negative ids are
            permuted within-group, negative ids remain fixed.

    Per Risk #13 mitigation, only Blocks 1–2 are permuted in feature space.
    """
    B, N, T, _ = features.shape
    base_idx = torch.arange(N, device=features.device)
    if sector_ids is None:
        perms = torch.stack(
            [
                torch.randperm(N, generator=generator, device=features.device)
                for _ in range(B)
            ],
            dim=0,
        )
    else:
        if sector_ids.ndim != 1 or sector_ids.shape[0] != N:
            raise ValueError(
                "sector_ids must be rank-1 with length N, got "
                f"shape {tuple(sector_ids.shape)} for N={N}"
            )
        groups: list[torch.Tensor] = []
        for sid in sorted(set(int(x) for x in sector_ids.tolist())):
            if sid < 0:
                continue
            idx = torch.where(sector_ids == sid)[0]
            if idx.numel() > 1:
                groups.append(idx)

        perms_list: list[torch.Tensor] = []
        for _ in range(B):
            perm = base_idx.clone()
            for idx in groups:
                shuffled = idx[torch.randperm(idx.numel(), generator=generator, device=idx.device)]
                perm[idx] = shuffled
            perms_list.append(perm)
        perms = torch.stack(perms_list, dim=0)

    permuted = features.clone()
    feat_idx = perms[:, :, None, None].expand(B, N, T, _PTA_PERMUTED_FEATURE_DIM)
    permuted[..., _PTA_PERMUTED_FEATURE_SLICE] = features[
        ..., _PTA_PERMUTED_FEATURE_SLICE
    ].gather(1, feat_idx)
    ret_idx = perms[:, :, None].expand(B, N, T)
    return permuted, returns.gather(1, ret_idx)


def expand_pta_training_batch(
    features: torch.Tensor,
    shifted_returns: torch.Tensor,
    sector_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    cfg: PTAAugmentationConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create spec-style PTA views for a full training batch.

    This mirrors Section 11.4b semantics:
    - permutation views (default: 4) with probability p_aug
    - one sign-flip view with probability p_sign_flip
    - one view per volatility scale with probability p_vol_scale
    - permutations are within-sector when sector IDs are provided

    Per Risk #13 mitigation, permutation views only swap Blocks 1–2 features.

    Args:
        features: (B, N, T, F)
        shifted_returns: (B, N, T)
        sector_ids: (N,) sector group IDs aligned to asset axis.
    Returns:
        (aug_features, aug_shifted_returns) concatenated along batch axis.
    """
    if features.ndim != 4:
        raise ValueError(
            f"expected features rank 4 (B,N,T,F), got shape {tuple(features.shape)}"
        )
    if shifted_returns.ndim != 3:
        raise ValueError(
            "expected shifted_returns rank 3 (B,N,T), "
            f"got shape {tuple(shifted_returns.shape)}"
        )
    if features.shape[:3] != shifted_returns.shape:
        raise ValueError(
            "features and shifted_returns must share (B,N,T), got "
            f"{tuple(features.shape[:3])} vs {tuple(shifted_returns.shape)}"
        )
    if sector_ids is not None:
        if sector_ids.ndim != 1 or sector_ids.shape[0] != features.shape[1]:
            raise ValueError(
                "sector_ids must be rank-1 with length N, got "
                f"shape {tuple(sector_ids.shape)} for N={features.shape[1]}"
            )

    cfg = cfg or PTAAugmentationConfig()
    cfg.validate()

    feature_views = [features]
    return_views = [shifted_returns]

    if cfg.permutations_per_batch > 0 and _draw_event(
        cfg.apply_prob,
        device=features.device,
        generator=generator,
    ):
        for _ in range(cfg.permutations_per_batch):
            perm_feats, perm_rets = _permute_assets_batch(
                features,
                shifted_returns,
                sector_ids=sector_ids,
                generator=generator,
            )
            feature_views.append(perm_feats)
            return_views.append(perm_rets)

    if _draw_event(cfg.sign_flip_prob, device=features.device, generator=generator):
        sign_feats = features.clone()
        sign_rets = -shifted_returns
        _scale_return_channels_inplace(sign_feats, -1.0)
        feature_views.append(sign_feats)
        return_views.append(sign_rets)

    if (
        cfg.vol_scale_values
        and _draw_event(cfg.vol_scale_prob, device=features.device, generator=generator)
    ):
        for scale in cfg.vol_scale_values:
            s = shifted_returns.new_tensor(scale)
            scaled_feats = features.clone()
            scaled_rets = shifted_returns * s
            _scale_return_channels_inplace(scaled_feats, s)
            _scale_volatility_channels_inplace(scaled_feats, s)
            feature_views.append(scaled_feats)
            return_views.append(scaled_rets)

    return torch.cat(feature_views, dim=0), torch.cat(return_views, dim=0)


def select_portfolio_shards(
    shard_paths: list[Path],
    max_assets: int = 100,
    required_assets: int | None = None,
) -> list[Path]:
    """Select real per-asset shards for portfolio training.

    FDA synthetic shards are sequence samples, not asset identity shards, so they
    must not be interpreted as additional assets in multi-asset head training.
    """
    real_paths = sorted(p for p in shard_paths if not p.name.startswith(_SYNTH_PREFIX))
    if len(real_paths) > max_assets:
        raise ValueError(
            f"found {len(real_paths)} real asset shards, exceeds spec max_assets={max_assets}"
        )
    if required_assets is not None and len(real_paths) != required_assets:
        raise ValueError(
            "portfolio training requires an exact asset universe for cross-asset "
            f"objectives: expected {required_assets}, found {len(real_paths)}"
        )
    return real_paths


def prepare_return_targets(shifted_returns: torch.Tensor) -> torch.Tensor:
    """Convert DataLoader return targets (B, N, T) to loss layout (B, T, N)."""
    if shifted_returns.ndim != 3:
        raise ValueError(
            f"expected shifted returns rank 3 (B,N,T), got shape {tuple(shifted_returns.shape)}"
        )
    return shifted_returns.permute(0, 2, 1).contiguous()


def realized_correlation_matrix(
    shifted_returns: torch.Tensor,
    corr_window: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-batch realised cross-asset correlation matrices.

    Args:
        shifted_returns: (B, N, T)
        corr_window: number of future steps used for realised correlation target.
    Returns:
        corr: (B, N, N)
    """
    if shifted_returns.ndim != 3:
        raise ValueError(
            "expected shifted_returns rank 3 (B,N,T), "
            f"got shape {tuple(shifted_returns.shape)}"
        )
    if corr_window < 2:
        raise ValueError(f"corr_window must be >= 2, got {corr_window}")

    window = min(corr_window, shifted_returns.shape[-1])
    rets = shifted_returns[:, :, :window]
    rets = rets - rets.mean(dim=-1, keepdim=True)
    denom = max(window - 1, 1)
    cov = torch.matmul(rets, rets.transpose(1, 2)) / denom  # (B, N, N)

    var = torch.diagonal(cov, dim1=1, dim2=2).clamp(min=eps)
    std = torch.sqrt(var)
    scale = std.unsqueeze(2) * std.unsqueeze(1)
    corr = (cov / scale.clamp(min=eps)).clamp(min=-1.0, max=1.0)

    eye = torch.eye(corr.shape[1], device=corr.device, dtype=corr.dtype).unsqueeze(0)
    corr = corr * (1.0 - eye) + eye
    return corr


def correlation_frobenius_loss(
    fused_tokens: torch.Tensor,
    shifted_returns: torch.Tensor,
    corr_window: int = 20,
) -> torch.Tensor:
    """Frobenius loss between predicted and realised cross-asset correlations.

    Args:
        fused_tokens: (B, T, N, D)
        shifted_returns: (B, N, T)
    """
    if fused_tokens.ndim != 4:
        raise ValueError(
            f"expected fused_tokens rank 4 (B,T,N,D), got {tuple(fused_tokens.shape)}"
        )
    if shifted_returns.ndim != 3:
        raise ValueError(
            "expected shifted_returns rank 3 (B,N,T), "
            f"got {tuple(shifted_returns.shape)}"
        )
    B, T, N, _ = fused_tokens.shape
    if shifted_returns.shape[0] != B or shifted_returns.shape[1] != N:
        raise ValueError(
            "fused_tokens and shifted_returns must agree on (B,N), got "
            f"fused (B={B},N={N}) vs returns {tuple(shifted_returns.shape)}"
        )
    if T <= 0:
        raise ValueError("fused_tokens time axis must be non-empty")

    asset_repr = F.normalize(fused_tokens.mean(dim=1), dim=-1)  # (B, N, D)
    pred_corr = torch.matmul(asset_repr, asset_repr.transpose(1, 2)).clamp(-1.0, 1.0)
    target_corr = realized_correlation_matrix(
        shifted_returns.to(fused_tokens.device), corr_window=corr_window
    )
    diff = pred_corr - target_corr.detach()
    return diff.pow(2).sum(dim=(1, 2)).mean()


def _load_shard_rows(path: Path, feature_dim: int = 280) -> np.ndarray:
    """Load shard rows as (rows, feature_dim)."""
    raw = np.memmap(path, dtype="float32", mode="r")
    n_floats = int(raw.size)
    if n_floats % feature_dim != 0:
        raise ValueError(
            f"shard {path} has unsupported row width "
            f"(floats={n_floats}, expected multiple of {feature_dim})"
        )
    return raw.reshape(-1, feature_dim)


class MultiAssetShardDataset(IterableDataset):
    """Yields synchronized (features, returns[, future_returns]) windows across N assets.

    Loads one f32bin shard per symbol, aligns them by truncating to the
    shortest shard length, then samples T+1-step windows at the same
    position offset across all assets.

    Each sample (future_return_horizon == 0):
        features:      (N, T, F)        — F=280 (184 per-asset + 96 market)
        returns:       (N, T)           — close log-return proxy (feature index 3)

    Each sample (future_return_horizon > 0):
        features:      (N, T, F)
        returns:       (N, T)
        future_returns:(N, future_return_horizon) — returns at steps T+1..T+horizon,
                        completely out of the feature window; used as the correlation
                        prediction target in the cross-asset warmup (spec §11.4a).
    """

    FEATURE_DIM = 280
    RETURN_IDX = 3  # index of close log-return within price_volume block

    def __init__(
        self,
        shard_paths: list[Path],
        seq_len: int = 64,
        shuffle: bool = True,
        stride: int = 1,
        augment: bool = False,
        future_return_horizon: int = 0,
        pta_cfg: PTAAugmentationConfig | None = None,
    ) -> None:
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.stride = stride
        self.augment = augment
        self.future_return_horizon = future_return_horizon
        self.pta_cfg = pta_cfg or PTAAugmentationConfig()

    def _load_shards(self) -> np.ndarray | None:
        """Load and align all shards to shape (N, rows, F)."""
        arrays = []
        for path in self.shard_paths:
            arrays.append(
                _load_shard_rows(
                    path,
                    feature_dim=self.FEATURE_DIM,
                )
            )
        if not arrays:
            return None
        min_rows = min(a.shape[0] for a in arrays)
        # Tail-align: keep the most recent min_rows rows (shared calendar end)
        aligned = np.stack([a[-min_rows:] for a in arrays], axis=0)  # (N, T, F)
        return aligned

    def __iter__(self) -> Iterator[tuple]:
        data = self._load_shards()  # (N, rows, F)
        if data is None:
            return

        N, rows, F = data.shape
        horizon = self.future_return_horizon
        # Need seq_len+1 contiguous rows for features/returns, plus horizon more for
        # the out-of-window future correlation target (spec §11.4a).
        n_seqs = rows - self.seq_len - horizon
        if n_seqs <= 0:
            return

        indices = np.arange(0, n_seqs, self.stride)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in indices:
            window = data[:, start : start + self.seq_len + 1, :]  # (N, T+1, F)
            feats = torch.from_numpy(window[:, :-1, :].copy())  # (N, T, F)
            returns = torch.from_numpy(
                window[:, 1:, self.RETURN_IDX].copy()  # (N, T) next-step returns
            )
            if self.augment:
                feats, returns = apply_pta_batch_augmentations(
                    feats,
                    returns,
                    cfg=self.pta_cfg,
                )
            if horizon > 0:
                # Future returns start immediately after the training window
                future_slice = data[
                    :,
                    start + self.seq_len + 1 : start + self.seq_len + 1 + horizon,
                    self.RETURN_IDX,
                ]  # (N, horizon)
                yield feats, returns, torch.from_numpy(future_slice.copy())
            else:
                yield feats, returns


def build_portfolio_loader(
    shard_paths: list[Path],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    stride: int = 1,
    augment: bool = False,
    future_return_horizon: int = 0,
    pta_cfg: PTAAugmentationConfig | None = None,
) -> DataLoader:
    ds = MultiAssetShardDataset(
        shard_paths,
        seq_len=seq_len,
        shuffle=shuffle,
        stride=stride,
        augment=augment,
        future_return_horizon=future_return_horizon,
        pta_cfg=pta_cfg,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


class SyntheticWindowPortfolioDataset(IterableDataset):
    """Loads FDA synthetic joint-window shards.

    Each FDA_SYNTH_*.f32bin file contains one or more joint multi-asset windows.
    Each window is stored as a flat float32 array with shape
    (n_assets, seq_len + 1 [+ future_return_horizon], FEATURE_DIM).

    File layout: W windows concatenated (W inferred from file size):
        W × n_assets × rows_per_window × FEATURE_DIM  float32 values.

    This preserves the cross-asset correlation structure generated by the FDA
    diffusion model — sampling independent per-asset windows would destroy it
    (§8.5 requirement). Each shard file must contain at least one complete
    joint window; files whose size is not an exact multiple of the window stride
    are skipped with a warning.

    Each sample (future_return_horizon == 0):
        features:      (N, T, F)
        returns:       (N, T)

    Each sample (future_return_horizon > 0):
        features:      (N, T, F)
        returns:       (N, T)
        future_returns:(N, future_return_horizon) — out-of-window correlation target
    """

    FEATURE_DIM = 280
    RETURN_IDX = 3

    def __init__(
        self,
        shard_paths: list[Path],
        n_assets: int,
        seq_len: int = 64,
        future_return_horizon: int = 0,
        augment: bool = True,
        shuffle: bool = True,
        pta_cfg: PTAAugmentationConfig | None = None,
    ) -> None:
        self.shard_paths = shard_paths
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.future_return_horizon = future_return_horizon
        self.augment = augment
        self.shuffle = shuffle
        self.pta_cfg = pta_cfg or PTAAugmentationConfig()

    def _load_joint_windows(self) -> list[np.ndarray]:
        """Return memory-mapped views of all joint windows across all shards.

        Each entry in the returned list is an (N, rows_per_window, F) array
        backed by a memmap — not copied into RAM until sliced.
        """
        rows_per_window = self.seq_len + 1 + self.future_return_horizon
        floats_per_window = self.n_assets * rows_per_window * self.FEATURE_DIM
        all_windows: list[np.ndarray] = []
        for path in self.shard_paths:
            raw = np.memmap(path, dtype="float32", mode="r")
            n_floats = int(raw.size)
            if n_floats == 0:
                warnings.warn(f"Empty FDA joint shard skipped: {path.name}")
                continue
            if n_floats % floats_per_window != 0:
                warnings.warn(
                    f"FDA joint shard {path.name}: {n_floats} floats is not a "
                    f"multiple of joint window size {floats_per_window} "
                    f"(n_assets={self.n_assets}, rows={rows_per_window}, "
                    f"F={self.FEATURE_DIM}). Skipped."
                )
                continue
            n_wins = n_floats // floats_per_window
            wins = raw.reshape(n_wins, self.n_assets, rows_per_window, self.FEATURE_DIM)
            for i in range(n_wins):
                all_windows.append(wins[i])  # (N, rows_per_window, F) — memmap view
        return all_windows

    def __iter__(self) -> Iterator[tuple]:
        all_windows = self._load_joint_windows()
        if not all_windows:
            return

        rng = np.random.default_rng()
        indices = list(range(len(all_windows)))
        horizon = self.future_return_horizon

        while True:
            if self.shuffle:
                rng.shuffle(indices)
            for idx in indices:
                window = all_windows[idx]  # (N, rows_per_window, F)
                feats = torch.from_numpy(window[:, : self.seq_len, :].copy())  # (N, T, F)
                returns = torch.from_numpy(
                    window[:, 1 : self.seq_len + 1, self.RETURN_IDX].copy()  # (N, T)
                )
                if self.augment:
                    feats, returns = apply_pta_batch_augmentations(
                        feats,
                        returns,
                        cfg=self.pta_cfg,
                    )
                if horizon > 0:
                    future_returns = torch.from_numpy(
                        window[:, self.seq_len + 1 :, self.RETURN_IDX].copy()  # (N, horizon)
                    )
                    yield feats, returns, future_returns
                else:
                    yield feats, returns


def build_synthetic_portfolio_loader(
    shard_paths: list[Path],
    n_assets: int,
    seq_len: int,
    batch_size: int,
    augment: bool = True,
    future_return_horizon: int = 0,
    num_workers: int = 0,
    pta_cfg: PTAAugmentationConfig | None = None,
) -> DataLoader:
    ds = SyntheticWindowPortfolioDataset(
        shard_paths=shard_paths,
        n_assets=n_assets,
        seq_len=seq_len,
        future_return_horizon=future_return_horizon,
        augment=augment,
        pta_cfg=pta_cfg,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


# ── Training config ───────────────────────────────────────────────────────────


@dataclass
class PortfolioTrainConfig:
    head: str = "return"  # "return" or "sharpe"
    total_steps: int = 50_000
    # Spec §11.4a: ~10 H100-hours. At ~5 steps/s on H100 (B=8, N=100, T=64)
    # that is ~180 k steps; 10 k is a practical lower bound for meaningful warmup.
    cross_asset_warmup_steps: int = 10_000
    cross_asset_corr_window: int = 20
    cross_asset_warmup_batch_size: int = 8
    batch_size: int = 4
    seq_len: int = 64
    lr: float = 3e-5  # lower than pre-training — fine-tuning
    warmup_steps: int = 500
    grad_clip: float = 0.5
    lambda_return: float = 0.1
    lambda_turn: float = 0.1
    lambda_vol: float = 0.05
    fda_loss_weight: float = 0.6
    fda_mix_prob: float = 0.5
    enforce_regime_balance: bool = True
    regime_resample_attempts: int = 8
    regime_pool_size: int = 32  # pool size for quantile-based regime selection
    pta_cfg: PTAAugmentationConfig = field(default_factory=PTAAugmentationConfig)
    asset_sector_ids: tuple[int, ...] | None = None
    dsr_eta: float = 0.01
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 5_000
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/portfolio"))
    bptt_chunk: int = 16  # truncated BPTT chunk length (0 = full seq)
    # PTA §11.4b expands B up to 10× (1 orig + 4 perms + 1 sign + 4 vol scales).
    # At N=100 this overflows 80GB. Micro-batching splits the expanded batch
    # into groups, forward+backward each, and accumulates gradients — bit-exact
    # identical to a single large forward because every loss term reduces
    # independently along the PTA batch axis. 0 = no micro-batching.
    pta_micro_batch_size: int = 0
    patience_steps: int = 1000  # early stopping threshold measured in training steps
    patience_min_delta: float = 1e-3  # min val-loss improvement to reset patience counter
    patience_grace_steps: int = 3000  # no early-stop before this step (covers LR warmup + initial plateau)
    freeze_backbone_steps: int = 500
    backbone_lr_scale: float = 0.1
    gradient_checkpointing: bool = False
    compile: bool = False  # torch.compile the backbone (2-4x faster, ~60s warmup)
    debug: bool = False  # enable verbose diagnostics (memory traces, per-group peaks)

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / f"{self.head}_head_best.pt"

    @property
    def warmup_path(self) -> Path:
        return self.checkpoint_dir / f"{self.head}_cross_asset_warmup.pt"

    @property
    def step_path_fn(self):
        return lambda step: self.checkpoint_dir / f"{self.head}_head_step{step}.pt"


# ── Trainer ───────────────────────────────────────────────────────────────────


class PortfolioTrainer:
    """Trains MeridianR or MeridianS on top of the pre-trained backbone."""

    def __init__(
        self,
        backbone: RAGDNBackbone,
        fusion: CrossAssetFusion,
        regime_detector: RegimeDetector,
        input_projection: InputProjection,
        head_r: MeridianR,
        head_s: MeridianS,
        aux_head: AuxHeadSurprise,
        vol_head: PortfolioVolHead,
        spec: MeridianSpec,
        device: torch.device,
    ) -> None:
        self.device = device
        self.spec = spec
        self.backbone = backbone.to(device)
        self.fusion = fusion.to(device)
        self.regime_detector = regime_detector.to(device)
        self.input_projection = input_projection.to(device)
        self.head_r = head_r.to(device)
        self.head_s = head_s.to(device)
        self.aux_head = aux_head.to(device)
        self.vol_head = vol_head.to(device)

    def _cosine_lr(self, step: int, cfg: PortfolioTrainConfig) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(
            cfg.total_steps - cfg.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _batch_regime_ids(features: torch.Tensor) -> torch.Tensor:
        """Infer one regime label per batch item from averaged asset windows."""
        # features: (B, N, T, F) -> average across assets -> (B, T, F)
        seqs = features.mean(dim=1).cpu().numpy()
        labels = [infer_regime_label(seq.astype(np.float32, copy=False)) for seq in seqs]
        return torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _has_all_regimes(regime_ids: torch.Tensor, n_regimes: int = 4) -> bool:
        if regime_ids.numel() < n_regimes:
            return False
        uniq = set(int(x) for x in regime_ids.tolist())
        return uniq.issuperset(set(range(n_regimes)))

    @staticmethod
    def _quantile_regime_labels(features: torch.Tensor) -> torch.Tensor:
        """Pool-relative 4-regime labels via median split on (vol, ret) proxies.

        Unlike the absolute-threshold `infer_regime_label` heuristic (which
        collapses to a single regime when features are z-scored), this splits
        the pool at its own medians so all 4 regimes are populated whenever
        the pool has ≥4 items. Labels match the spec ordering:
            0 = low vol / positive ret  (low-vol bull)
            1 = high vol / positive ret (high-vol trending)
            2 = low vol / negative ret  (mean-reverting)
            3 = high vol / negative ret (crisis)
        """
        seqs = features.mean(dim=1).cpu().numpy()  # (M, T, F)
        vol = seqs[:, :, 184].mean(axis=-1)
        ret = seqs[:, :, 3].mean(axis=-1)
        vol_hi = (vol >= np.median(vol)).astype(np.int64)
        ret_neg = (ret < np.median(ret)).astype(np.int64)
        labels = vol_hi + 2 * ret_neg
        return torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _should_freeze_backbone(step: int, cfg: "PortfolioTrainConfig") -> bool:
        return cfg.freeze_backbone_steps > 0 and step < cfg.freeze_backbone_steps

    @staticmethod
    def _should_stop_early(
        step: int,
        last_improve_step: int,
        cfg: "PortfolioTrainConfig",
    ) -> bool:
        if cfg.patience_steps <= 0:
            return False
        if step < cfg.patience_grace_steps:
            return False
        return (step - last_improve_step) >= cfg.patience_steps

    @staticmethod
    def _validate_regime_balance_config(
        batch_size: int,
        n_regimes: int,
        enforce_regime_balance: bool,
    ) -> None:
        if enforce_regime_balance and batch_size < n_regimes:
            raise ValueError(
                "regime-balanced sampling requires batch_size >= n_regimes "
                f"({n_regimes}), got batch_size={batch_size}"
            )

    @staticmethod
    def _resolve_asset_sector_ids(
        asset_sector_ids: tuple[int, ...] | None,
        n_assets: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if asset_sector_ids is None:
            return None
        if len(asset_sector_ids) != n_assets:
            raise ValueError(
                "asset_sector_ids must align with asset count N, got "
                f"{len(asset_sector_ids)} ids for N={n_assets}"
            )
        return torch.tensor(asset_sector_ids, dtype=torch.long, device=device)

    def _run_cross_asset_warmup(
        self,
        train_loader: DataLoader,
        cfg: "PortfolioTrainConfig",
        backbone_params: list[torch.nn.Parameter],
        synthetic_loader: DataLoader | None = None,
    ) -> None:
        if cfg.cross_asset_warmup_steps <= 0:
            return

        warmup_optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_scale}],
            weight_decay=0.01,
        )
        warmup_sched_cfg = PortfolioTrainConfig(
            total_steps=max(cfg.cross_asset_warmup_steps, 1),
            warmup_steps=max(1, cfg.cross_asset_warmup_steps // 10),
        )
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            warmup_optimizer,
            lr_lambda=lambda s: self._cosine_lr(s, warmup_sched_cfg),
        )

        data_iter = iter(train_loader)
        synth_iter = iter(synthetic_loader) if synthetic_loader is not None else None

        def _pull_batch(source: str) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal data_iter, synth_iter
            if source == "synth":
                assert synthetic_loader is not None
                try:
                    return next(synth_iter)
                except StopIteration:
                    synth_iter = iter(synthetic_loader)
                    return next(synth_iter)
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                return next(data_iter)

        print(
            "Starting cross-asset warm-up: "
            f"{cfg.cross_asset_warmup_steps} steps, corr_window={cfg.cross_asset_corr_window}"
        )
        for warmup_step in range(cfg.cross_asset_warmup_steps):
            use_synth = (
                synth_iter is not None
                and torch.rand(1).item() < cfg.fda_mix_prob
            )
            source = "synth" if use_synth else "real"
            source_weight = cfg.fda_loss_weight if source == "synth" else 1.0
            features, shifted_returns = _pull_batch(source)

            warmup_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                # Memory-optimised warm-up path: returns (B,N,D) time-mean
                # directly via chunked activation checkpointing, avoiding the
                # (B,T,N,D) fused-sequence retention that OOMs below 80 GB.
                asset_mean = self._warmup_fused_mean(features, cfg)
                asset_repr = F.normalize(asset_mean, dim=-1)  # (B, N, D)
                pred_corr = torch.matmul(
                    asset_repr, asset_repr.transpose(1, 2)
                ).clamp(-1.0, 1.0)
                target_corr = realized_correlation_matrix(
                    shifted_returns.to(self.device),
                    corr_window=cfg.cross_asset_corr_window,
                )
                diff = pred_corr - target_corr.detach()
                warmup_loss = diff.pow(2).sum(dim=(1, 2)).mean()
                loss = source_weight * warmup_loss

            loss.backward()
            grad_params = [p for p in backbone_params if p.requires_grad]
            nn.utils.clip_grad_norm_(grad_params, cfg.grad_clip)
            warmup_optimizer.step()
            warmup_scheduler.step()

            if (warmup_step + 1) % max(cfg.log_every, 1) == 0:
                lr_now = warmup_scheduler.get_last_lr()[0]
                print(
                    f"warmup step={warmup_step + 1:>6d}  "
                    f"corr_frob={warmup_loss.item():.4f}  lr={lr_now:.2e}  "
                    f"source={source}"
                )

        self._save(
            cfg.warmup_path,
            cfg.cross_asset_warmup_steps,
            warmup_optimizer,
            warmup_scheduler,
            float("inf"),
        )
        print(f"Cross-asset warm-up complete. Saved: {cfg.warmup_path}")

    def _warmup_fused_mean(
        self,
        features: torch.Tensor,  # (B, N, T, F=280)
        cfg: "PortfolioTrainConfig | None" = None,
        sector_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Memory-optimised warm-up forward.

        Returns the **time-mean** of fused tokens, (B, N, D), which is the only
        quantity consumed by `correlation_frobenius_loss`. Skips the portfolio
        head / vol head / portfolio-state bookkeeping since they do not feed the
        warm-up loss. Processes T in chunks of size `cfg.bptt_chunk` under
        `torch.utils.checkpoint.checkpoint` so the per-timestep activations
        (input projection, fusion, backbone) are recomputed during backward
        instead of being retained for all T steps at once. Gradient semantics
        match the existing `_forward` path on the warm-up training signal
        (backbone state is already detached at bptt_chunk boundaries).
        """
        features = features.to(self.device)
        B, N, T, _ = features.shape
        bptt_chunk = cfg.bptt_chunk if cfg is not None else 0
        chunk_size = bptt_chunk if bptt_chunk > 0 else T

        asset_feats = features[..., :184]
        market_feats = features[..., 184:]
        mkt_mean = market_feats.mean(dim=1)  # (B, T, 96)

        portfolio_s = torch.zeros(
            B, self.spec.portfolio_state_dim, device=self.device
        )
        calendar = torch.zeros(B, self.spec.calendar_dim, device=self.device)
        if sector_ids is None:
            sector_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        else:
            if sector_ids.ndim != 1 or sector_ids.shape[0] != N:
                raise ValueError(
                    "sector_ids must be rank-1 with length N, got "
                    f"shape {tuple(sector_ids.shape)} for N={N}"
                )
            sector_ids = sector_ids.to(self.device, dtype=torch.long)
        episodic = torch.zeros(
            B, self.spec.n_episodic_retrievals, self.spec.d_model, device=self.device
        )
        regime_ema = torch.zeros(B, self.spec.regime_emb_dim, device=self.device)
        prev_regime_emb = torch.zeros(
            B, self.spec.regime_emb_dim, device=self.device
        )
        states = self.backbone.init_state(B, N, self.device)

        # Portfolio summary held constant during warm-up (head is inactive).
        portfolio_summary = torch.stack(
            [
                torch.ones(B, device=self.device),
                torch.zeros(B, device=self.device),
                torch.ones(B, device=self.device),
            ],
            dim=-1,
        )  # (B, 3)

        # Sector bias (constant across timesteps).
        with torch.no_grad():
            n_episodic = episodic.shape[1]
            n_total = N + 1 + 1 + n_episodic
            n_extras = n_total - N
            extra_ids = torch.full(
                (n_extras,), -1, dtype=torch.long, device=self.device
            )
            full_ids = torch.cat([sector_ids, extra_ids], dim=0)
            precomputed_bias = self.fusion.sector_bias.bias_matrix(full_ids, n_total)

        def _run_chunk(
            af_chunk: torch.Tensor,
            mf_chunk: torch.Tensor,
            regime_ema_in: torch.Tensor,
            prev_regime_emb_in: torch.Tensor,
            *state_tensors: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            """Process K timesteps, return (fused_sum, regime_ema, prev_regime, *states).

            Pure tensor I/O for `torch.utils.checkpoint.checkpoint`.
            """
            local_states = list(state_tensors)
            Bc, Nc, K, _ = af_chunk.shape
            fused_sum = torch.zeros(
                Bc, Nc, self.spec.d_model, device=af_chunk.device, dtype=af_chunk.dtype
            )
            r_ema = regime_ema_in
            prev_r = prev_regime_emb_in
            for k in range(K):
                af_t = af_chunk[:, :, k, :]
                mf_t = mf_chunk[:, k, :]
                regime_emb, posteriors, _ = self.regime_detector(
                    mf_t, portfolio_summary, r_ema, prev_r
                )
                r_ema = (0.98 * r_ema + 0.02 * regime_emb).detach()
                prev_r = regime_emb.detach()

                asset_tok, mkt_tok, port_tok, _ = self.input_projection(
                    af_t, mf_t, portfolio_s, calendar, regime_emb
                )
                out, local_states = self.backbone(
                    asset_tok, regime_emb, posteriors, local_states
                )
                fused, _ = self.fusion(
                    out,
                    mkt_tok,
                    port_tok,
                    episodic,
                    sector_ids,
                    precomputed_bias=precomputed_bias,
                )
                fused_sum = fused_sum + fused
            return (fused_sum, r_ema, prev_r, *local_states)

        total_fused_sum = torch.zeros(
            B, N, self.spec.d_model, device=self.device
        )
        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            af_chunk = asset_feats[:, :, t0:t1, :]
            mf_chunk = mkt_mean[:, t0:t1, :]

            # Matches existing bptt_chunk semantics: detach recurrent state
            # between chunks so the backward graph covers at most one chunk.
            states = [s.detach() for s in states]
            regime_ema = regime_ema.detach()
            prev_regime_emb = prev_regime_emb.detach()

            result = _activation_checkpoint(
                _run_chunk,
                af_chunk,
                mf_chunk,
                regime_ema,
                prev_regime_emb,
                *states,
                use_reentrant=False,
            )
            fused_sum_chunk = result[0]
            regime_ema = result[1]
            prev_regime_emb = result[2]
            states = list(result[3:])

            total_fused_sum = total_fused_sum + fused_sum_chunk

        return total_fused_sum / float(T)  # (B, N, D)

    def _forward(
        self,
        features: torch.Tensor,  # (B, N, T, F=280)
        head: str,
        cfg: "PortfolioTrainConfig | None" = None,
        collect_fused: bool = False,
        sector_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Run the full model forward pass over a T-step sequence.

        Uses bf16 autocast to halve activation memory and truncated BPTT
        (cfg.bptt_chunk > 0) to cap the gradient graph length.

        Returns:
            weights:  (B, T, N) — allocation weights at each step
            pred_vol: (B, T, horizon) or None
            fused_seq: (B, T, N, D) or None
        """
        features = features.to(self.device)
        B, N, T, _ = features.shape
        bptt_chunk = cfg.bptt_chunk if cfg is not None else 0

        # Split per-asset (184) and market (96) blocks
        asset_feats = features[..., :184]  # (B, N, T, 184)
        market_feats = features[..., 184:]  # (B, N, T, 96) — take mean across assets

        # Market features: average across assets (they share the same market context)
        mkt_mean = market_feats.mean(dim=1)  # (B, T, 96)

        # Bootstrap portfolio/calendar/sector state
        portfolio_s = torch.zeros(B, self.spec.portfolio_state_dim, device=self.device)
        calendar = torch.zeros(B, self.spec.calendar_dim, device=self.device)
        if sector_ids is None:
            sector_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        else:
            if sector_ids.ndim != 1 or sector_ids.shape[0] != N:
                raise ValueError(
                    "sector_ids must be rank-1 with length N, got "
                    f"shape {tuple(sector_ids.shape)} for N={N}"
                )
            sector_ids = sector_ids.to(self.device, dtype=torch.long)
        episodic = torch.zeros(
            B, self.spec.n_episodic_retrievals, self.spec.d_model, device=self.device
        )

        # Regime detector state (updated each timestep). Initialize prev_regime_emb
        # to zeros (instead of None) so it can be passed as a tensor through
        # torch.utils.checkpoint — matches the cross-asset warm-up path.
        regime_ema = torch.zeros(B, self.spec.regime_emb_dim, device=self.device)
        prev_regime_emb = torch.zeros(B, self.spec.regime_emb_dim, device=self.device)
        prev_port_return = torch.zeros(B, device=self.device)
        portfolio_value = torch.ones(B, device=self.device)

        states = self.backbone.init_state(B, N, self.device)

        # Validate and initialize portfolio-state buffers once per sequence.
        n_full = self.spec.n_assets  # 100
        if N > n_full:
            raise ValueError(
                f"asset axis N={N} exceeds spec.n_assets={n_full} in portfolio forward pass"
            )
        expected_port_dim = 2 * n_full + 1
        if expected_port_dim != self.spec.portfolio_state_dim:
            raise ValueError(
                "portfolio_state_dim must equal 2*n_assets + 1, got "
                f"{self.spec.portfolio_state_dim} for n_assets={n_full}"
            )
        asset_cum_log_returns = torch.zeros(B, n_full, device=self.device)

        # Pre-compute sector bias (constant across all timesteps)
        with torch.no_grad():
            n_episodic = episodic.shape[1]
            n_total = N + 1 + 1 + n_episodic  # assets + market + portfolio + episodic
            n_extras = n_total - N
            extra_ids = torch.full(
                (n_extras,), -1, dtype=torch.long, device=self.device
            )
            full_ids = torch.cat([sector_ids, extra_ids], dim=0)
            precomputed_bias = self.fusion.sector_bias.bias_matrix(full_ids, n_total)

        # Chunked T-loop with activation checkpointing. Each chunk rematerialises
        # its per-step backbone/SWA/fusion/head activations during backward; only
        # the per-step outputs (weights, vol preds, fused) are retained for the
        # DSR loss. Semantically identical to the previous non-chunked loop with
        # bptt_chunk-based state detach — we just move the detach and the
        # checkpoint boundary to the same place. Memory saving scales with
        # T / chunk_size.
        def _run_chunk(
            af_chunk: torch.Tensor,           # (B, N, K, 184)
            mf_chunk: torch.Tensor,           # (B, K, 96)
            portfolio_s_in: torch.Tensor,
            regime_ema_in: torch.Tensor,
            prev_regime_emb_in: torch.Tensor,
            prev_port_return_in: torch.Tensor,
            portfolio_value_in: torch.Tensor,
            asset_cum_log_returns_in: torch.Tensor,
            *state_tensors: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            local_states = list(state_tensors)
            ps = portfolio_s_in
            r_ema = regime_ema_in
            prev_r = prev_regime_emb_in
            prev_pr = prev_port_return_in
            pv = portfolio_value_in
            acum = asset_cum_log_returns_in
            Bc, Nc, K, _ = af_chunk.shape

            w_list: list[torch.Tensor] = []
            v_list: list[torch.Tensor] = []
            f_list: list[torch.Tensor] = []

            # Hoist allocations that are constant across the T-loop.
            ones_Bc = torch.ones(Bc, device=af_chunk.device, dtype=af_chunk.dtype)
            no_pad = Nc == n_full  # skip F.pad no-ops (N==n_full in portfolio training)

            for k in range(K):
                af_t = af_chunk[:, :, k, :]
                mf_t = mf_chunk[:, k, :]

                pv = pv * (1.0 + prev_pr)
                portfolio_summary = torch.stack(
                    [pv, prev_pr, ones_Bc],
                    dim=-1,
                )
                regime_emb, posteriors, _ = self.regime_detector(
                    mf_t, portfolio_summary, r_ema, prev_r
                )
                r_ema = (0.98 * r_ema + 0.02 * regime_emb).detach()
                prev_r = regime_emb.detach()

                asset_tok, mkt_tok, port_tok, _ = self.input_projection(
                    af_t, mf_t, ps, calendar, regime_emb
                )
                out, local_states = self.backbone(
                    asset_tok, regime_emb, posteriors, local_states
                )
                fused, mkt_out = self.fusion(
                    out,
                    mkt_tok,
                    port_tok,
                    episodic,
                    sector_ids,
                    precomputed_bias=precomputed_bias,
                )

                if collect_fused:
                    f_list.append(fused)

                if head == "return":
                    w_t = self.head_r(fused)
                else:
                    w_t = self.head_s(fused)
                w_list.append(w_t)

                if head == "return":
                    v_list.append(self.vol_head(mkt_out))

                w_det = w_t.detach()
                asset_log_ret_det = af_t[:, :, 3].detach()
                if no_pad:
                    w_padded = w_det
                    asset_step_log_returns = asset_log_ret_det
                else:
                    w_padded = F.pad(w_det, (0, n_full - Nc))
                    asset_step_log_returns = F.pad(asset_log_ret_det, (0, n_full - Nc))
                acum = acum + asset_step_log_returns
                unrealized_pnl = torch.expm1(acum)
                step_port_return = (w_t.detach() * af_t[:, :, 3]).sum(dim=-1)
                next_pv = pv * (1.0 + step_port_return)
                pv_log = next_pv.clamp(min=1e-6).log().unsqueeze(-1)
                ps = torch.cat([w_padded, unrealized_pnl, pv_log], dim=-1)
                prev_pr = step_port_return

            w_stack = torch.stack(w_list, dim=1)  # (B, K, N)
            v_stack = (
                torch.stack(v_list, dim=1)
                if v_list
                else torch.zeros(0, device=w_stack.device, dtype=w_stack.dtype)
            )
            f_stack = (
                torch.stack(f_list, dim=1)
                if f_list
                else torch.zeros(0, device=w_stack.device, dtype=w_stack.dtype)
            )
            return (
                w_stack,
                v_stack,
                f_stack,
                ps,
                r_ema,
                prev_r,
                prev_pr,
                pv,
                acum,
                *local_states,
            )

        chunk_size = bptt_chunk if bptt_chunk > 0 else T
        use_checkpoint = bptt_chunk > 0 and torch.is_grad_enabled()

        all_weights: list[torch.Tensor] = []
        all_vol_preds: list[torch.Tensor] = []
        all_fused: list[torch.Tensor] = []

        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            af_chunk = asset_feats[:, :, t0:t1, :]
            mf_chunk = mkt_mean[:, t0:t1, :]

            # Detach recurrent state at the chunk boundary so the backward graph
            # covers at most one chunk — same semantics as the old bptt_chunk
            # in-place detach, just hoisted out of the inner loop.
            states = [s.detach() for s in states]
            portfolio_s = portfolio_s.detach()
            regime_ema = regime_ema.detach()
            prev_regime_emb = prev_regime_emb.detach()
            prev_port_return = prev_port_return.detach()
            portfolio_value = portfolio_value.detach()
            asset_cum_log_returns = asset_cum_log_returns.detach()

            if use_checkpoint:
                result = _activation_checkpoint(
                    _run_chunk,
                    af_chunk,
                    mf_chunk,
                    portfolio_s,
                    regime_ema,
                    prev_regime_emb,
                    prev_port_return,
                    portfolio_value,
                    asset_cum_log_returns,
                    *states,
                    use_reentrant=False,
                )
            else:
                result = _run_chunk(
                    af_chunk,
                    mf_chunk,
                    portfolio_s,
                    regime_ema,
                    prev_regime_emb,
                    prev_port_return,
                    portfolio_value,
                    asset_cum_log_returns,
                    *states,
                )

            w_stack, v_stack, f_stack = result[0], result[1], result[2]
            portfolio_s = result[3]
            regime_ema = result[4]
            prev_regime_emb = result[5]
            prev_port_return = result[6]
            portfolio_value = result[7]
            asset_cum_log_returns = result[8]
            states = list(result[9:])

            all_weights.append(w_stack)
            if head == "return" and v_stack.numel() > 0:
                all_vol_preds.append(v_stack)
            if collect_fused and f_stack.numel() > 0:
                all_fused.append(f_stack)

        weights = torch.cat(all_weights, dim=1)  # (B, T, N)
        pred_vol = torch.cat(all_vol_preds, dim=1) if all_vol_preds else None
        fused_seq = torch.cat(all_fused, dim=1) if all_fused else None
        return weights, pred_vol, fused_seq

    def evaluate(
        self,
        val_loader: DataLoader,
        cfg: PortfolioTrainConfig,
        max_batches: int = 20,
    ) -> dict[str, float]:
        for m in (
            self.backbone,
            self.fusion,
            self.regime_detector,
            self.input_projection,
            self.head_r,
            self.head_s,
            self.vol_head,
        ):
            m.eval()
        totals: dict[str, float] = {"total": 0.0, "dsr": 0.0}
        n = 0
        with (
            torch.no_grad(),
            torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ),
        ):
            for i, (features, shifted_returns) in enumerate(val_loader):
                if i >= max_batches:
                    break
                # DataLoader collates item shape (N, T, F) into (B, N, T, F).
                returns = prepare_return_targets(shifted_returns).to(self.device)
                weights, pred_vol, _ = self._forward(features, cfg.head, cfg)
                if cfg.head == "return":
                    loss_dict = return_head_loss(
                        weights,
                        returns,
                        pred_vol=pred_vol,
                        lambda_ret=cfg.lambda_return,
                        lambda_turn=cfg.lambda_turn,
                        lambda_vol=cfg.lambda_vol,
                        dsr_eta=cfg.dsr_eta,
                    )
                    totals["total"] += loss_dict["total"].item()
                    totals["dsr"] += loss_dict["dsr"].item()
                else:
                    loss = dsr_loss(weights, returns, eta=cfg.dsr_eta)
                    totals["total"] += loss.item()
                    totals["dsr"] += loss.item()
                n += 1
        for m in (
            self.backbone,
            self.fusion,
            self.regime_detector,
            self.input_projection,
            self.head_r,
            self.head_s,
            self.vol_head,
        ):
            m.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    def train(
        self,
        train_loader: DataLoader,
        cfg: PortfolioTrainConfig,
        val_loader: DataLoader | None = None,
        synthetic_loader: DataLoader | None = None,
        cross_asset_warmup_train_loader: DataLoader | None = None,
        cross_asset_warmup_synthetic_loader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> None:
        if cfg.cross_asset_warmup_batch_size < 1:
            raise ValueError(
                f"cross_asset_warmup_batch_size must be >= 1, got {cfg.cross_asset_warmup_batch_size}"
            )
        warmup_train_loader = (
            cross_asset_warmup_train_loader if cross_asset_warmup_train_loader is not None else train_loader
        )
        warmup_synthetic_loader = (
            cross_asset_warmup_synthetic_loader
            if cross_asset_warmup_synthetic_loader is not None
            else synthetic_loader
        )
        # The chunked T-loop in _forward wraps each bptt_chunk-step window in
        # activation checkpointing, which already covers backbone rematerialisation.
        # Stacking backbone-level grad_checkpoint inside that outer checkpoint
        # produces pathological nested-ckpt behaviour on recompute (observed
        # 750MB+ phantom allocations at B=4). Disable the backbone's internal
        # grad_ckpt whenever the chunk T-loop is active.
        chunk_tloop_active = cfg.bptt_chunk > 0
        if cfg.gradient_checkpointing and not chunk_tloop_active:
            self.backbone.enable_gradient_checkpointing()
        elif cfg.gradient_checkpointing and chunk_tloop_active:
            print(
                "[portfolio_train] --gradient-checkpointing ignored: chunk "
                "T-loop (bptt_chunk>0) already handles activation "
                "rematerialisation."
            )

        backbone_params = (
            list(self.backbone.parameters())
            + list(self.fusion.parameters())
            + list(self.regime_detector.parameters())
            + list(self.input_projection.parameters())
        )
        if cfg.head == "return":
            head_params = list(self.head_r.parameters()) + list(self.vol_head.parameters())
        else:
            head_params = list(self.head_s.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": cfg.lr},
                {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_scale},
            ],
            weight_decay=0.01,
        )
        trainable_params = head_params + backbone_params
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: self._cosine_lr(s, cfg)
        )

        start_step = 0
        best_monitor = float("inf")
        last_improve_step = 0

        if resume_from is not None and resume_from.exists():
            ckpt = load_checkpoint(resume_from)
            # Load backbone + fusion from checkpoint (pre-training or earlier portfolio ckpt)
            state = ckpt.model_state
            if "backbone" in state:
                self.backbone.load_state_dict(state["backbone"])
                self.fusion.load_state_dict(state["fusion"])
                self.regime_detector.load_state_dict(state["regime_detector"])
                self.input_projection.load_state_dict(state["input_projection"])
            if "vol_head" in state:
                try:
                    self.vol_head.load_state_dict(state["vol_head"])
                except RuntimeError:
                    pass  # pretrain vol_head has different arch — fresh init is fine
            if f"head_{cfg.head}" in state:
                (self.head_r if cfg.head == "return" else self.head_s).load_state_dict(
                    state[f"head_{cfg.head}"]
                )
                start_step = ckpt.step
                best_monitor = ckpt.best_val_loss
                last_improve_step = start_step
                # Only restore optimizer/scheduler if param groups match
                # (e.g. return→return or sharpe→sharpe resume, NOT return→sharpe)
                ckpt_groups = ckpt.optimizer_state.get("param_groups", [])
                opt_groups = optimizer.state_dict()["param_groups"]
                same_layout = len(ckpt_groups) == len(opt_groups) and all(
                    len(a["params"]) == len(b["params"])
                    for a, b in zip(ckpt_groups, opt_groups)
                )
                if same_layout:
                    optimizer.load_state_dict(ckpt.optimizer_state)
                    scheduler.load_state_dict(ckpt.scheduler_state)
                else:
                    print("Skipping optimizer/scheduler restore due param-group mismatch")
            print(f"Resumed from step {start_step}  best_monitor={best_monitor:.4f}")

        # Compile after checkpoint loading so state dict keys match (uncompiled
        # submodules have plain keys; torch.compile wraps them under _orig_mod.*
        # prefix). We compile the T-loop hot sub-modules: backbone (8 GDN + 2
        # SWA + 8 FMoE) plus the per-step modules called around it. Only useful
        # with --pta-deterministic (stable shapes); stochastic PTA would force
        # recompilation each step.
        if cfg.compile:
            print("Compiling hot modules (one-time, ~60-120s)...")
            self.backbone = torch.compile(self.backbone)
            self.regime_detector = torch.compile(self.regime_detector)
            self.input_projection = torch.compile(self.input_projection)
            self.fusion = torch.compile(self.fusion)
            self.head_r = torch.compile(self.head_r)
            self.head_s = torch.compile(self.head_s)
            self.vol_head = torch.compile(self.vol_head)
            print("Compiled.")

        def _set_backbone_trainable(trainable: bool) -> None:
            for p in backbone_params:
                p.requires_grad_(trainable)

        _set_backbone_trainable(True)

        for m in (
            self.backbone,
            self.fusion,
            self.regime_detector,
            self.input_projection,
            self.head_r,
            self.head_s,
            self.vol_head,
        ):
            m.train()

        self._validate_regime_balance_config(
            batch_size=cfg.batch_size,
            n_regimes=self.spec.n_regimes,
            enforce_regime_balance=cfg.enforce_regime_balance,
        )

        if cfg.cross_asset_warmup_steps > 0 and start_step == 0:
            self._run_cross_asset_warmup(
                train_loader=warmup_train_loader,
                cfg=cfg,
                backbone_params=backbone_params,
                synthetic_loader=warmup_synthetic_loader,
            )
        elif cfg.cross_asset_warmup_steps > 0 and start_step > 0:
            print(
                f"Skipping cross-asset warm-up because training resumes at step {start_step}"
            )

        backbone_frozen = self._should_freeze_backbone(start_step, cfg)
        if backbone_frozen:
            _set_backbone_trainable(False)
            print(
                f"Backbone frozen until step {cfg.freeze_backbone_steps} "
                f"(then unfreeze at {cfg.backbone_lr_scale:.2f}x LR)"
            )
        else:
            _set_backbone_trainable(True)

        step = start_step
        data_iter = iter(train_loader)
        synth_iter = iter(synthetic_loader) if synthetic_loader is not None else None
        sector_ids_t: torch.Tensor | None = None

        def _pull_batch(source: str) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal data_iter, synth_iter
            if source == "synth":
                assert synthetic_loader is not None
                try:
                    return next(synth_iter)
                except StopIteration:
                    synth_iter = iter(synthetic_loader)
                    return next(synth_iter)
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                return next(data_iter)

        while step < cfg.total_steps:
            source_weight = 1.0
            use_synth = (
                synth_iter is not None
                and torch.rand(1).item() < cfg.fda_mix_prob
            )
            if backbone_frozen and not self._should_freeze_backbone(step, cfg):
                _set_backbone_trainable(True)
                backbone_frozen = False
                print(
                    f"Unfroze backbone at step {step} "
                    f"(backbone lr scale={cfg.backbone_lr_scale:.2f})"
                )

            source = "synth" if use_synth else "real"
            features, shifted_returns = _pull_batch(source)
            if sector_ids_t is None:
                sector_ids_t = self._resolve_asset_sector_ids(
                    cfg.asset_sector_ids,
                    n_assets=features.shape[1],
                    device=features.device,
                )
            if source == "synth":
                source_weight = cfg.fda_loss_weight

            if cfg.enforce_regime_balance:
                # Pool-based selection: accumulate a candidate pool large enough
                # to cover all 4 regimes under quantile-relative labelling,
                # then pick one item per regime (padding with random extras up
                # to cfg.batch_size). Robust to skewed absolute-threshold
                # labelling of the real data.
                target_batch = features.shape[0]
                pool_size = max(
                    cfg.regime_pool_size, target_batch * self.spec.n_regimes
                )
                feats_pool = [features]
                shifts_pool = [shifted_returns]
                while sum(f.shape[0] for f in feats_pool) < pool_size:
                    f, s = _pull_batch(source)
                    feats_pool.append(f)
                    shifts_pool.append(s)
                feats_cat = torch.cat(feats_pool, dim=0)
                shifts_cat = torch.cat(shifts_pool, dim=0)

                labels = self._quantile_regime_labels(feats_cat)
                selected_idx: list[int] = []
                for r in range(self.spec.n_regimes):
                    cand = (labels == r).nonzero(as_tuple=True)[0]
                    if cand.numel() == 0:
                        continue
                    pick = cand[torch.randint(0, cand.numel(), (1,)).item()].item()
                    selected_idx.append(int(pick))

                if len(selected_idx) < target_batch:
                    chosen = set(selected_idx)
                    remaining = [i for i in range(feats_cat.shape[0]) if i not in chosen]
                    fill = torch.randperm(len(remaining))[
                        : target_batch - len(selected_idx)
                    ]
                    for i in fill.tolist():
                        selected_idx.append(remaining[i])
                elif len(selected_idx) > target_batch:
                    perm = torch.randperm(len(selected_idx))[:target_batch]
                    selected_idx = [selected_idx[i] for i in perm.tolist()]

                idx_t = torch.tensor(selected_idx, dtype=torch.long)
                features = feats_cat.index_select(0, idx_t)
                shifted_returns = shifts_cat.index_select(0, idx_t)

            features, shifted_returns = expand_pta_training_batch(
                features,
                shifted_returns,
                sector_ids=sector_ids_t,
                cfg=cfg.pta_cfg,
            )

            # DataLoader collates item shape (N, T, F) into (B, N, T, F).
            optimizer.zero_grad(set_to_none=True)
            returns = prepare_return_targets(shifted_returns).to(self.device)

            # PTA-view gradient accumulation. Each PTA view is an independent
            # sample along the batch axis, so splitting the expanded batch into
            # micro-groups and accumulating grads is identical to one big batch.
            B_eff = features.shape[0]
            mbs = cfg.pta_micro_batch_size if cfg.pta_micro_batch_size > 0 else B_eff
            mbs = max(1, min(mbs, B_eff))
            num_groups = (B_eff + mbs - 1) // mbs

            mem_diag = cfg.debug and step - start_step < 3 and self.device.type == "cuda"
            if mem_diag:
                torch.cuda.reset_peak_memory_stats()
                logger.debug(
                    "[mem-diag] step=%d  B_eff=%d  micro_bs=%d  num_groups=%d  features.shape=%s",
                    step, B_eff, mbs, num_groups, tuple(features.shape),
                )

            log_parts: dict[str, float] = {}

            for g in range(num_groups):
                s = g * mbs
                e = min(s + mbs, B_eff)
                feats_g = features[s:e]
                rets_g = returns[s:e]
                # Weight by fraction of full batch so grads sum to mean across
                # all B_eff views (losses already reduce to per-group means).
                group_scale = float(e - s) / float(B_eff)

                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                ):
                    weights, pred_vol, _ = self._forward(feats_g, cfg.head, cfg)

                    if cfg.head == "return":
                        loss_dict = return_head_loss(
                            weights,
                            rets_g,
                            pred_vol=pred_vol,
                            lambda_ret=cfg.lambda_return,
                            lambda_turn=cfg.lambda_turn,
                            lambda_vol=cfg.lambda_vol,
                            dsr_eta=cfg.dsr_eta,
                        )
                        loss = loss_dict["total"]
                    else:
                        loss = dsr_loss(weights, rets_g, eta=cfg.dsr_eta)
                        loss_dict = {"dsr": loss, "total": loss}

                if mem_diag:
                    fwd_peak = torch.cuda.max_memory_allocated() / 1e9
                    fwd_curr = torch.cuda.memory_allocated() / 1e9
                    logger.debug(
                        "[mem-diag] step=%d group=%d/%d FWD_DONE micro_B=%d peak=%.1fGB current=%.1fGB",
                        step, g, num_groups, feats_g.shape[0], fwd_peak, fwd_curr,
                    )

                (source_weight * group_scale * loss).backward()

                if mem_diag:
                    bwd_peak = torch.cuda.max_memory_allocated() / 1e9
                    bwd_curr = torch.cuda.memory_allocated() / 1e9
                    logger.debug(
                        "[mem-diag] step=%d group=%d/%d BWD_DONE peak=%.1fGB current=%.1fGB",
                        step, g, num_groups, bwd_peak, bwd_curr,
                    )

                for k, v in loss_dict.items():
                    log_parts[k] = log_parts.get(k, 0.0) + group_scale * v.item()

            loss_dict = {
                k: torch.tensor(v, device=self.device) for k, v in log_parts.items()
            }
            grad_params = [p for p in trainable_params if p.requires_grad]
            nn.utils.clip_grad_norm_(grad_params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                parts = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
                print(f"step={step:>7d}  {parts}  lr={lr_now:.2e}")

            if val_loader is not None and step % cfg.eval_every == 0:
                val_metrics = self.evaluate(val_loader, cfg)
                monitor = val_metrics["dsr"]
                print(
                    f"  val total={val_metrics['total']:.4f}  "
                    f"val dsr={val_metrics['dsr']:.4f}"
                )
                if monitor < best_monitor - cfg.patience_min_delta:
                    best_monitor = monitor
                    last_improve_step = step
                    self._save(cfg.best_path, step, optimizer, scheduler, best_monitor)
                else:
                    # Still save best checkpoint on any absolute improvement, even
                    # if below min_delta — we just don't reset the patience counter.
                    if monitor < best_monitor:
                        best_monitor = monitor
                        self._save(cfg.best_path, step, optimizer, scheduler, best_monitor)
                    if self._should_stop_early(step, last_improve_step, cfg):
                        print(
                            f"Early stopping at step {step} "
                            f"(no improvement ≥{cfg.patience_min_delta} "
                            f"for {cfg.patience_steps} steps)"
                        )
                        break

            if step % cfg.save_every == 0:
                self._save(
                    cfg.step_path_fn(step), step, optimizer, scheduler, best_monitor
                )

        print(f"Training complete at step {step}. Best monitor: {best_monitor:.4f}")

    def _model_state(self) -> dict:
        # torch.compile wraps modules under _orig_mod; unwrap so checkpoints use
        # plain key names and can be loaded by an uncompiled model.
        def _unwrap(m):
            return getattr(m, "_orig_mod", m)
        return {
            "backbone": _unwrap(self.backbone).state_dict(),
            "fusion": _unwrap(self.fusion).state_dict(),
            "regime_detector": _unwrap(self.regime_detector).state_dict(),
            "input_projection": _unwrap(self.input_projection).state_dict(),
            "head_return": _unwrap(self.head_r).state_dict(),
            "head_sharpe": _unwrap(self.head_s).state_dict(),
            "aux_head": self.aux_head.state_dict(),
            "vol_head": _unwrap(self.vol_head).state_dict(),
        }

    def _save(self, path: Path, step: int, optimizer, scheduler, best: float) -> None:
        save_checkpoint(
            Checkpoint(
                step=step,
                model_state=self._model_state(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                best_val_loss=best,
            ),
            path,
        )


# ── Smoke data helpers ────────────────────────────────────────────────────────


def _write_smoke_shard(path: Path, n_rows: int, feature_dim: int = 280) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_struct = struct.Struct(f"<{feature_dim}f")
    rng = np.random.default_rng(seed=0)
    with open(path, "wb") as f:
        for _ in range(n_rows):
            row = rng.standard_normal(feature_dim).astype(np.float32)
            f.write(row_struct.pack(*row))


def _make_smoke_shards(tmp_dir: Path, n_assets: int, seq_len: int) -> list[Path]:
    paths = []
    for i in range(n_assets):
        for split in ("train", "validation"):
            p = tmp_dir / split / f"ASSET{i:02d}.f32bin"
            _write_smoke_shard(p, seq_len + 10)
            if split == "train":
                paths.append(p)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────


def _build_model(spec: MeridianSpec, device: torch.device):
    from train.pretrain import build_model

    backbone, fusion, regime_detector, input_projection = build_model(spec, device)
    head_r = MeridianR(spec.d_model).to(device)
    head_s = MeridianS(spec.d_model).to(device)
    aux_head = AuxHeadSurprise(spec.d_model).to(device)
    vol_head = PortfolioVolHead(spec.d_model).to(device)
    return (
        backbone,
        fusion,
        regime_detector,
        input_projection,
        head_r,
        head_s,
        aux_head,
        vol_head,
    )


def main() -> None:
    # Suppress noisy torch.compile / inductor warnings that obscure training logs
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._inductor")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo")

    parser = argparse.ArgumentParser(description="Portfolio head training")
    parser.add_argument("--head", choices=["return", "sharpe"], default="return")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("symbols.txt"),
        help="Universe symbols file used to derive sector IDs for PTA permutations",
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument(
        "--cross-asset-warmup-steps",
        dest="cross_asset_warmup_steps",
        type=int,
        default=-1,
        help="Cross-asset warm-up steps before head training "
        "(default: 1000 for return, 0 for sharpe)",
    )
    parser.add_argument(
        "--cross-asset-corr-window",
        dest="cross_asset_corr_window",
        type=int,
        default=20,
        help="Future return window size used for cross-asset warm-up correlation targets",
    )
    parser.add_argument(
        "--cross-asset-warmup-batch-size",
        dest="cross_asset_warmup_batch_size",
        type=int,
        default=8,
        help="Batch size used only for cross-asset warm-up (spec default: 8)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda-return", type=float, default=0.1)
    parser.add_argument("--lambda-turn", type=float, default=0.1)
    parser.add_argument("--lambda-vol", type=float, default=0.05)
    parser.add_argument("--fda-loss-weight", type=float, default=0.6)
    parser.add_argument("--fda-mix-prob", type=float, default=0.5)
    parser.add_argument(
        "--pta-prob",
        type=float,
        default=0.5,
        help="Probability of adding permutation PTA views for a sampled batch",
    )
    parser.add_argument(
        "--pta-permutations",
        type=int,
        default=4,
        help="Number of permutation views added when PTA permutation triggers",
    )
    parser.add_argument(
        "--pta-sign-flip-prob",
        type=float,
        default=0.5,
        help="Probability of adding a sign-flipped PTA batch view",
    )
    parser.add_argument(
        "--pta-vol-scale-prob",
        type=float,
        default=0.7,
        help="Probability of adding volatility-scaled PTA batch views",
    )
    parser.add_argument(
        "--pta-deterministic",
        action="store_true",
        help="Force all PTA view probabilities to 1.0. B_eff becomes constant "
        "(1 + permutations + 1 + len(vol_scales) views), enabling torch.compile "
        "with stable shapes. Strictly stronger augmentation than stochastic PTA.",
    )
    parser.add_argument(
        "--disable-regime-balance",
        action="store_true",
        help="Disable batch-level 4-regime coverage enforcement",
    )
    parser.add_argument(
        "--regime-resample-attempts",
        type=int,
        default=8,
        help="(Deprecated) Maximum resampling attempts — pool-based selection now handles coverage",
    )
    parser.add_argument(
        "--regime-pool-size",
        type=int,
        default=32,
        help="Candidate pool size for quantile-relative regime-balanced selection",
    )
    parser.add_argument(
        "--bptt-chunk",
        type=int,
        default=16,
        help="Truncated BPTT chunk length — shorter = less memory, shorter gradient horizon (0 = full seq)",
    )
    parser.add_argument(
        "--pta-micro-batch-size",
        type=int,
        default=0,
        help="Split PTA-expanded batch into micro-groups of this size, with "
        "gradient accumulation. Bit-exact identical to one big batch (every "
        "loss reduces independently along the batch axis). 0 = off.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/portfolio")
    )
    parser.add_argument(
        "--resume-backbone",
        type=Path,
        default=None,
        help="Pre-trained backbone checkpoint to load",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute activations during backward to save VRAM. "
        "Required on 24GB GPUs.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the backbone for faster training (~60s warmup, "
        "then 2-4x faster per step). Recommended on A100/H100.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug diagnostics (per-group GPU memory traces for "
        "the first 3 training steps). Off by default.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride between sequences (lower = more data, default 4)",
    )
    parser.add_argument("--patience-steps", type=int, default=1000)
    parser.add_argument("--patience-min-delta", type=float, default=1e-3)
    parser.add_argument("--patience-grace-steps", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument(
        "--freeze-backbone-steps",
        type=int,
        default=500,
        help="Freeze backbone for this many initial steps before fine-tuning",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="Backbone LR scale relative to head LR after unfreeze",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s",
        force=True,
    )
    if args.cross_asset_warmup_batch_size < 1:
        raise ValueError("--cross-asset-warmup-batch-size must be >= 1")

    if args.smoke:
        args.steps = args.steps if args.steps < 50_000 else 3
        args.batch_size = 2
        args.seq_len = max(args.seq_len, 8)
        args.workers = 0

    cross_asset_warmup_steps = (
        args.cross_asset_warmup_steps
        if args.cross_asset_warmup_steps >= 0
        else (1_000 if args.head == "return" else 0)
    )
    if args.smoke and args.cross_asset_warmup_steps < 0:
        cross_asset_warmup_steps = 0

    if args.pta_deterministic:
        pta_cfg = PTAAugmentationConfig(
            apply_prob=1.0,
            permutations_per_batch=args.pta_permutations,
            sign_flip_prob=1.0,
            vol_scale_prob=1.0,
        )
    else:
        pta_cfg = PTAAugmentationConfig(
            apply_prob=args.pta_prob,
            permutations_per_batch=args.pta_permutations,
            sign_flip_prob=args.pta_sign_flip_prob,
            vol_scale_prob=args.pta_vol_scale_prob,
        )
    pta_cfg.validate()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DEFAULT_SPEC
    spec.validate()

    (
        backbone,
        fusion,
        regime_detector,
        input_projection,
        head_r,
        head_s,
        aux_head,
        vol_head,
    ) = _build_model(spec, device)
    trainer = PortfolioTrainer(
        backbone,
        fusion,
        regime_detector,
        input_projection,
        head_r,
        head_s,
        aux_head,
        vol_head,
        spec,
        device,
    )

    if args.smoke:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n_smoke = 5  # fewer assets for smoke speed
            shard_paths = _make_smoke_shards(td, n_smoke, args.seq_len)
            val_paths = [
                td / "validation" / f"ASSET{i:02d}.f32bin" for i in range(n_smoke)
            ]
            train_loader = build_portfolio_loader(
                shard_paths, args.seq_len, args.batch_size, num_workers=0
            )
            cross_asset_warmup_train_loader = (
                build_portfolio_loader(
                    shard_paths,
                    args.seq_len,
                    args.cross_asset_warmup_batch_size,
                    num_workers=0,
                )
                if (
                    cross_asset_warmup_steps > 0
                    and args.cross_asset_warmup_batch_size != args.batch_size
                )
                else None
            )
            val_loader = build_portfolio_loader(
                val_paths, args.seq_len, args.batch_size, shuffle=False, num_workers=0
            )
            cfg = PortfolioTrainConfig(
                head=args.head,
                total_steps=args.steps,
                cross_asset_warmup_steps=cross_asset_warmup_steps,
                cross_asset_corr_window=args.cross_asset_corr_window,
                cross_asset_warmup_batch_size=args.cross_asset_warmup_batch_size,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                lr=args.lr,
                lambda_return=args.lambda_return,
                lambda_turn=args.lambda_turn,
                lambda_vol=args.lambda_vol,
                fda_loss_weight=args.fda_loss_weight,
                fda_mix_prob=args.fda_mix_prob,
                enforce_regime_balance=False,
                regime_resample_attempts=args.regime_resample_attempts,
                regime_pool_size=args.regime_pool_size,
                bptt_chunk=args.bptt_chunk,
                pta_micro_batch_size=args.pta_micro_batch_size,
                pta_cfg=pta_cfg,
                debug=args.debug,
                warmup_steps=0,
                log_every=min(args.log_every, 1),
                eval_every=min(args.eval_every, 2),
                save_every=args.save_every,
                checkpoint_dir=td / "checkpoints",
                freeze_backbone_steps=args.freeze_backbone_steps,
                backbone_lr_scale=args.backbone_lr_scale,
            )
            trainer.train(
                train_loader,
                cfg,
                val_loader,
                cross_asset_warmup_train_loader=cross_asset_warmup_train_loader,
                resume_from=args.resume_backbone,
            )
            print("Smoke test passed.")
    else:
        from market_data.dataset_ingestion import enumerate_shards

        all_train_paths = enumerate_shards(args.data_dir, "train")
        all_val_paths = enumerate_shards(args.data_dir, "validation")
        train_paths = select_portfolio_shards(
            all_train_paths,
            max_assets=spec.n_assets,
            required_assets=spec.n_assets,
        )
        synth_train_paths = sorted(
            p for p in all_train_paths if p.name.startswith(_SYNTH_PREFIX)
        )
        val_paths = select_portfolio_shards(
            all_val_paths,
            max_assets=spec.n_assets,
            required_assets=spec.n_assets,
        )
        excluded_train = len(all_train_paths) - len(train_paths)
        excluded_val = len(all_val_paths) - len(val_paths)
        if excluded_train or excluded_val:
            print(
                f"Excluded synthetic/non-asset shards from direct asset universe: "
                f"train={excluded_train} val={excluded_val}"
            )
        if not train_paths:
            raise RuntimeError(f"No train shards at {args.data_dir}/train")
        symbol_to_sector = load_symbol_sector_map(args.symbols_file)
        train_sector_ids = tuple(build_asset_sector_ids(train_paths, symbol_to_sector))
        mapped_assets = sum(1 for sid in train_sector_ids if sid >= 0)
        print(
            f"Loaded sector IDs for {mapped_assets}/{len(train_sector_ids)} train assets "
            f"from {args.symbols_file}"
        )
        train_loader = build_portfolio_loader(
            train_paths,
            args.seq_len,
            args.batch_size,
            num_workers=args.workers,
            stride=args.stride,
            augment=False,
        )
        synthetic_loader = (
            build_synthetic_portfolio_loader(
                synth_train_paths,
                n_assets=len(train_paths),
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                augment=False,
                num_workers=0,
            )
            if synth_train_paths
            else None
        )
        cross_asset_warmup_train_loader = (
            build_portfolio_loader(
                train_paths,
                args.seq_len,
                args.cross_asset_warmup_batch_size,
                num_workers=args.workers,
                stride=args.stride,
                augment=False,
            )
            if (
                cross_asset_warmup_steps > 0
                and args.cross_asset_warmup_batch_size != args.batch_size
            )
            else None
        )
        cross_asset_warmup_synthetic_loader = (
            build_synthetic_portfolio_loader(
                synth_train_paths,
                n_assets=len(train_paths),
                seq_len=args.seq_len,
                batch_size=args.cross_asset_warmup_batch_size,
                augment=False,
                num_workers=0,
            )
            if (
                synthetic_loader is not None
                and cross_asset_warmup_steps > 0
                and args.cross_asset_warmup_batch_size != args.batch_size
            )
            else None
        )
        if cross_asset_warmup_train_loader is not None:
            print(
                "Using dedicated cross-asset warm-up batch size "
                f"{args.cross_asset_warmup_batch_size} "
                f"(head training batch size {args.batch_size})"
            )
        if synthetic_loader is not None:
            print(
                f"Using {len(synth_train_paths)} FDA synthetic shards in {args.head}-head "
                "batch mixing"
            )
        val_loader = build_portfolio_loader(
            val_paths,
            args.seq_len,
            args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        cfg = PortfolioTrainConfig(
            head=args.head,
            total_steps=args.steps,
            cross_asset_warmup_steps=cross_asset_warmup_steps,
            cross_asset_corr_window=args.cross_asset_corr_window,
            cross_asset_warmup_batch_size=args.cross_asset_warmup_batch_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            lambda_return=args.lambda_return,
            lambda_turn=args.lambda_turn,
            lambda_vol=args.lambda_vol,
            fda_loss_weight=args.fda_loss_weight,
            fda_mix_prob=args.fda_mix_prob,
            enforce_regime_balance=not args.disable_regime_balance,
            regime_resample_attempts=args.regime_resample_attempts,
            regime_pool_size=args.regime_pool_size,
            bptt_chunk=args.bptt_chunk,
            pta_micro_batch_size=args.pta_micro_batch_size,
            pta_cfg=pta_cfg,
            debug=args.debug,
            asset_sector_ids=train_sector_ids,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_every=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
            patience_steps=args.patience_steps,
            patience_min_delta=args.patience_min_delta,
            patience_grace_steps=args.patience_grace_steps,
            freeze_backbone_steps=args.freeze_backbone_steps,
            backbone_lr_scale=args.backbone_lr_scale,
            gradient_checkpointing=args.gradient_checkpointing,
            compile=args.compile,
        )
        trainer.train(
            train_loader,
            cfg,
            val_loader,
            synthetic_loader=synthetic_loader,
            cross_asset_warmup_train_loader=cross_asset_warmup_train_loader,
            cross_asset_warmup_synthetic_loader=cross_asset_warmup_synthetic_loader,
            resume_from=args.resume_backbone,
        )


if __name__ == "__main__":
    main()
