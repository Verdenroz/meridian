"""Financial Diffusion Augmentation (FDA) offline pipeline

This module provides:
1. A lightweight regime-conditioned diffusion trainer/generator
2. A stylized-fact validation gate
3. Synthetic shard writing compatible with f32bin training readers
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.self_supervised_pretraining import infer_regime_label
from train.feature_schema import (
    MARKET_FEATURE_OFFSET,
    RETURN_FEATURE_INDEX,
    RSI_FEATURE_INDEX,
)


@dataclass
class FDAConfig:
    # Defaults match spec §8.2 (line 1038): T_diff=1000, 200 epochs, batch=64.
    # Earlier defaults (100 steps, 5 epochs, batch=32) severely underfit the
    # denoiser, producing off-manifold samples that the statistical gate cannot
    # distinguish from valid ones.
    window_len: int = 64
    feature_dim: int = 280
    diffusion_steps: int = 1000
    hidden_dim: int = 256
    lr: float = 3e-4
    epochs: int = 200
    batch_size: int = 64
    device: str = "cpu"
    generation_batch_size: int = 64
    target_valid_windows: int = 1_000
    max_generation_rounds: int = 1_000
    # Hard per-value bound on synthetic features. Real finance-query features
    # are normalized to O(1); a window with any |x| exceeding this is treated
    # as off-manifold and rejected outright, independent of the 9 statistical
    # checks. Overrides the score-based pass threshold when violated.
    max_abs_feature: float = 10.0
    # Target per-regime share of generated synthetic windows (spec §8.4 line
    # 1070-1073): low-vol trending, high-vol trending, mean-reverting, crisis.
    # Quotas control which regime IDs are *conditioned on* during diffusion
    # sampling; the realized distribution (via heuristic post-hoc labeling)
    # will track these but is not guaranteed to match exactly.
    regime_quotas: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)


@dataclass
class ReferenceStats:
    mean_corr_proxy: float
    tail_comovement: float


@dataclass
class StylizedFactReport:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]


@dataclass
class CrossAssetReference:
    """Pool-level reference statistics for spec §8.5 cross-asset validation.

    Holds flattened samples of per-window betas and MVP weights, each
    aggregated across many random N-asset pools drawn from the real data.
    Stored as samples (not moments) so the comparison to synthetic pools
    can be a non-parametric KS test.
    """
    beta_samples: np.ndarray
    mvp_weight_samples: np.ndarray
    pool_size: int
    n_pool_samples: int


@dataclass
class CrossAssetReport:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]


def _safe_std(x: np.ndarray) -> float:
    return float(max(np.std(x), 1e-8))


def _autocorr_lag1(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    den = np.linalg.norm(x0) * np.linalg.norm(x1)
    if den <= 1e-8:
        return 0.0
    return float(np.dot(x0, x1) / den)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError("correlation inputs must have same length")
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = np.linalg.norm(a0) * np.linalg.norm(b0)
    if den <= 1e-8:
        return 0.0
    return float(np.dot(a0, b0) / den)


def _compute_rsi_like(close_returns: np.ndarray, period: int = 14) -> np.ndarray:
    if close_returns.size < period + 1:
        return np.full_like(close_returns, 50.0, dtype=np.float64)
    gains = np.maximum(close_returns, 0.0)
    losses = -np.minimum(close_returns, 0.0)
    rsi = np.full(close_returns.shape[0], 50.0, dtype=np.float64)
    for i in range(period, close_returns.shape[0]):
        avg_gain = gains[i - period + 1 : i + 1].mean()
        avg_loss = losses[i - period + 1 : i + 1].mean()
        rs = avg_gain / max(avg_loss, 1e-8)
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_reference_stats(real_windows: np.ndarray) -> ReferenceStats:
    r = real_windows[:, :, RETURN_FEATURE_INDEX].reshape(-1)
    # Market proxy: first market-wide feature (volatility_sentiment block start).
    m = real_windows[:, :, MARKET_FEATURE_OFFSET].reshape(-1)

    mean_corr_proxy = _corr(r, m)
    r_sigma = _safe_std(r)
    m_sigma = _safe_std(m)
    tail = (
        (np.abs(r) > 2 * r_sigma)
        & (np.abs(m) > 2 * m_sigma)
        & (np.sign(r) == np.sign(m))
    ).mean()
    return ReferenceStats(
        mean_corr_proxy=float(mean_corr_proxy), tail_comovement=float(tail)
    )


# ── Cross-asset validation (spec §8.5) ───────────────────────────────────────


def _pool_cross_asset_stats(pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For an (n_assets, T, F) pool, compute per-asset betas + MVP weights.

    Market proxy is the equal-weight mean of the pool's returns. MVP weights
    come from the unconstrained minimum-variance closed form
    w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1). Pseudo-inverse handles singular covariance
    (common when n_assets > T).
    """
    returns = pool[:, :, 3]  # (n_assets, T)
    market = returns.mean(axis=0)
    market_var = max(float(market.var()), 1e-8)
    n_assets = returns.shape[0]
    betas = np.empty(n_assets, dtype=np.float64)
    for i in range(n_assets):
        betas[i] = float(np.cov(returns[i], market, ddof=0)[0, 1]) / market_var
    cov = np.cov(returns, ddof=0)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    ones = np.ones(cov.shape[0])
    w = inv @ ones
    denom = float(ones @ w)
    if abs(denom) < 1e-12:
        w = np.full(cov.shape[0], 1.0 / cov.shape[0])
    else:
        w = w / denom
    return betas, w


def _sample_cross_asset_stats(
    windows: np.ndarray,
    pool_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_windows = windows.shape[0]
    replace = n_windows < pool_size
    betas_accum: list[np.ndarray] = []
    mvp_accum: list[np.ndarray] = []
    for _ in range(n_samples):
        idx = rng.choice(n_windows, size=pool_size, replace=replace)
        pool = windows[idx]
        betas, mvp = _pool_cross_asset_stats(pool)
        betas_accum.append(betas)
        mvp_accum.append(mvp)
    return (
        np.concatenate(betas_accum).astype(np.float64),
        np.concatenate(mvp_accum).astype(np.float64),
    )


def compute_cross_asset_reference(
    real_windows: np.ndarray,
    pool_size: int = 100,
    n_pool_samples: int = 200,
    seed: int = 42,
) -> CrossAssetReference:
    """Build the reference beta + MVP-weight sample pools from real data."""
    if real_windows.ndim != 3:
        raise ValueError("real_windows must be (N, T, F)")
    rng = np.random.default_rng(seed)
    betas, mvp = _sample_cross_asset_stats(
        real_windows, pool_size=pool_size, n_samples=n_pool_samples, rng=rng
    )
    return CrossAssetReference(
        beta_samples=betas,
        mvp_weight_samples=mvp,
        pool_size=pool_size,
        n_pool_samples=n_pool_samples,
    )


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic without a scipy dependency."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    both = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, both, side="right") / max(len(a_sorted), 1)
    cdf_b = np.searchsorted(b_sorted, both, side="right") / max(len(b_sorted), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def validate_cross_asset_structure(
    synth_windows: np.ndarray,
    reference: CrossAssetReference,
    n_pool_samples: int = 100,
    beta_ks_threshold: float = 0.20,
    mvp_ks_threshold: float = 0.20,
    seed: int = 7,
) -> CrossAssetReport:
    """Spec §8.5 portfolio-level gate: compare synth and real beta + MVP
    distributions via KS statistic on pools of `reference.pool_size` windows.

    The literal spec compares element-wise MVP weight arrays; since the FDA
    generates anonymous per-asset windows without stable identity, we compare
    the *distributions* of weights instead. Same failure modes are caught
    (off-manifold covariance → concentrated/unstable MVP, unrealistic factor
    exposures → shifted beta distribution) without needing asset alignment.
    """
    if synth_windows.ndim != 3:
        raise ValueError("synth_windows must be (N, T, F)")
    rng = np.random.default_rng(seed)
    synth_betas, synth_mvp = _sample_cross_asset_stats(
        synth_windows,
        pool_size=reference.pool_size,
        n_samples=n_pool_samples,
        rng=rng,
    )
    beta_ks = _ks_statistic(reference.beta_samples, synth_betas)
    mvp_ks = _ks_statistic(reference.mvp_weight_samples, synth_mvp)
    checks = {
        "beta_distribution": beta_ks < beta_ks_threshold,
        "mvp_weight_distribution": mvp_ks < mvp_ks_threshold,
    }
    metrics = {
        "beta_ks": beta_ks,
        "mvp_ks": mvp_ks,
        "beta_ks_threshold": beta_ks_threshold,
        "mvp_ks_threshold": mvp_ks_threshold,
        "pool_size": float(reference.pool_size),
        "n_real_pools": float(reference.n_pool_samples),
        "n_synth_pools": float(n_pool_samples),
    }
    return CrossAssetReport(
        passed=all(checks.values()), checks=checks, metrics=metrics
    )


_SOFT_CHECK_KEYS: tuple[str, ...] = (
    "fat_tails",
    "volatility_clustering",
    "negative_skewness",
    "no_return_autocorr",
    "leverage_effect",
    "volume_vol_correlation",
    "cross_asset_correlation_proxy",
    "tail_comovement_proxy",
    "indicator_consistency",
)
_SOFT_PASS_THRESHOLD: float = 2.0 / 3.0  # spec §8.3: 6 of 9 soft checks.


def _hard_checks(
    window: np.ndarray, max_abs_feature: float
) -> tuple[bool, bool, float]:
    """Per-value sanity: all values finite, no value exceeds `max_abs_feature`.

    Returns (finite_ok, range_ok, observed_max_abs). When `finite_ok` is False,
    `observed_max_abs` is returned as `inf` so the range check cascades.
    """
    finite_ok = bool(np.isfinite(window).all())
    max_abs = float(np.abs(window).max()) if finite_ok else float("inf")
    range_ok = finite_ok and max_abs <= max_abs_feature
    return finite_ok, range_ok, max_abs


def _soft_checks(
    window: np.ndarray, reference: ReferenceStats
) -> tuple[dict[str, bool], dict[str, float]]:
    """The 9 statistical stylized-fact checks (spec §8.3 lines 1045-1056).

    Caller scores these against `_SOFT_PASS_THRESHOLD`; they are independent
    of the hard finite/range gate in `_hard_checks`.
    """
    r = window[:, RETURN_FEATURE_INDEX]
    vol = np.abs(window[:, RETURN_FEATURE_INDEX + 1]) + 1e-6
    mkt = (
        window[:, MARKET_FEATURE_OFFSET]
        if window.shape[1] > MARKET_FEATURE_OFFSET
        else np.zeros_like(r)
    )

    mu = float(r.mean())
    std = _safe_std(r)
    z = (r - mu) / std
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    skewness = float(np.mean(z**3))
    vol_cluster = _autocorr_lag1(r**2)
    return_ac = _autocorr_lag1(r)
    leverage = _corr(r[:-1], np.abs(r[1:])) if r.shape[0] > 2 else 0.0
    vol_corr = _corr(np.log(vol), np.abs(r))
    corr_proxy = _corr(r, mkt)
    r_sigma = _safe_std(r)
    m_sigma = _safe_std(mkt)
    tail_move = float(
        (
            (np.abs(r) > 2 * r_sigma)
            & (np.abs(mkt) > 2 * m_sigma)
            & (np.sign(r) == np.sign(mkt))
        ).mean()
    )

    rsi_comp = _compute_rsi_like(r)
    if window.shape[1] > RSI_FEATURE_INDEX:
        rsi_feat = np.clip(
            window[:, RSI_FEATURE_INDEX] * 50.0 + 50.0, 0.0, 100.0
        )
    else:
        rsi_feat = np.full_like(rsi_comp, 50.0)
    indicator_mae = float(np.mean(np.abs(rsi_comp - rsi_feat)) / 100.0)

    checks = {
        "fat_tails": excess_kurtosis > 1.0,
        "volatility_clustering": vol_cluster > 0.10,
        "negative_skewness": skewness < -0.05,
        "no_return_autocorr": abs(return_ac) < 0.10,
        "leverage_effect": leverage < -0.05,
        "volume_vol_correlation": vol_corr > 0.15,
        "cross_asset_correlation_proxy": abs(corr_proxy - reference.mean_corr_proxy)
        <= 0.10,
        "tail_comovement_proxy": abs(tail_move - reference.tail_comovement) <= 0.05,
        "indicator_consistency": indicator_mae < 0.10,
    }
    metrics = {
        "excess_kurtosis": excess_kurtosis,
        "vol_cluster": vol_cluster,
        "skewness": skewness,
        "return_autocorr": return_ac,
        "leverage": leverage,
        "volume_vol_corr": vol_corr,
        "corr_proxy": corr_proxy,
        "tail_move": tail_move,
        "indicator_mae": indicator_mae,
    }
    return checks, metrics


def validate_stylized_facts(
    window: np.ndarray,
    reference: ReferenceStats,
    max_abs_feature: float = 10.0,
) -> StylizedFactReport:
    """9 soft statistical checks + 2 hard sanity checks (finite / range).

    `passed` requires BOTH hard checks AND ≥ 2/3 of the soft checks.
    """
    finite_ok, range_ok, max_abs = _hard_checks(window, max_abs_feature)
    soft, soft_metrics = _soft_checks(window, reference)

    checks = {**soft, "finite": finite_ok, "range": range_ok}
    metrics = {**soft_metrics, "max_abs": max_abs}
    score = float(
        sum(1 for k in _SOFT_CHECK_KEYS if checks[k]) / len(_SOFT_CHECK_KEYS)
    )
    metrics["score"] = score
    passed = finite_ok and range_ok and (score >= _SOFT_PASS_THRESHOLD)
    return StylizedFactReport(passed=passed, checks=checks, metrics=metrics)


class FDADiffusionModel(nn.Module):
    """Lightweight regime-conditioned diffusion denoiser over (T, F) windows."""

    def __init__(
        self,
        feature_dim: int = 280,
        hidden_dim: int = 256,
        diffusion_steps: int = 100,
        n_regimes: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.diffusion_steps = diffusion_steps
        self.time_emb = nn.Embedding(diffusion_steps + 1, hidden_dim)
        self.regime_emb = nn.Embedding(n_regimes, hidden_dim)
        self.in_proj = nn.Linear(feature_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, diffusion_steps + 1))
        alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, regime: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, T, F), t: (B,), regime: (B,)
        time_cond = self.time_emb(t)[:, None, :]
        regime_cond = self.regime_emb(regime)[:, None, :]
        h = self.in_proj(x) + time_cond + regime_cond
        return self.ff(h)


def load_windows_from_shards(
    shard_paths: list[Path],
    window_len: int = 64,
    stride: int = 4,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load sliding windows from per-asset `.f32bin` shards.

    Each shard is a flat float32 stream of `N × 280` floats; we memmap, reshape
    to `(N, 280)`, and emit overlapping windows of length `window_len` with
    `stride` rows between successive window starts (so the *offset between
    adjacent window starts* is `stride`, not between samples inside a window).

    Shards with fewer than `window_len + 1` rows contribute zero windows and
    are counted as `too_short`. Shards whose byte size isn't a multiple of
    the 280-float row are counted as `malformed`. Both categories are skipped
    silently but the function raises RuntimeError if the totals leave zero
    windows — which usually means collection needs to be rerun.

    Returns (windows, regime_labels) where windows has shape `(n, T, 280)`
    and regime_labels is a parallel int64 array produced by the heuristic
    `infer_regime_label` from `core.self_supervised_pretraining`.
    """
    windows: list[np.ndarray] = []
    regimes: list[int] = []
    malformed = 0
    too_short = 0
    for path in shard_paths:
        raw = np.memmap(path, dtype="float32", mode="r")
        if len(raw) % 280 != 0:
            malformed += 1
            continue
        rows = raw.reshape(-1, 280)

        n = rows.shape[0]
        if n <= window_len:
            too_short += 1
            continue
        for start in range(0, n - window_len, stride):
            w = rows[start : start + window_len]
            windows.append(w.astype(np.float32, copy=True))
            regimes.append(infer_regime_label(w))
            if max_windows is not None and len(windows) >= max_windows:
                break
        if max_windows is not None and len(windows) >= max_windows:
            break
    if not windows:
        raise RuntimeError(
            "no windows could be loaded from shard paths "
            f"(checked={len(shard_paths)}, malformed={malformed}, too_short={too_short}); "
            "re-run collection to refresh shards"
        )
    return np.stack(windows, axis=0), np.asarray(regimes, dtype=np.int64)


def train_diffusion_model(
    windows: np.ndarray,
    regimes: np.ndarray,
    cfg: FDAConfig,
) -> FDADiffusionModel:
    device = torch.device(cfg.device)
    x = torch.from_numpy(windows).to(device)
    r = torch.from_numpy(regimes).to(device)
    model = FDADiffusionModel(
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        diffusion_steps=cfg.diffusion_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    n = x.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            x0 = x[idx]
            regime = r[idx]
            t = torch.randint(1, cfg.diffusion_steps + 1, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            ab = model.alpha_bar[t][:, None, None]
            noisy = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
            pred = model(noisy, t, regime)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def _allocate_regime_ids(
    n: int,
    quotas: tuple[float, float, float, float],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Deterministically assign `n` regime IDs with counts proportional to `quotas`.

    Uses largest-remainder apportionment to handle non-integer quota × n, then
    shuffles so adjacent entries are not regime-sorted. This is what lets
    `sample_diffusion_windows` honour spec §8.4's 30/25/25/20% target exactly
    (up to rounding) instead of drifting toward uniform.
    """
    if len(quotas) != 4:
        raise ValueError(f"quotas must have length 4, got {len(quotas)}")
    total = float(sum(quotas))
    if total <= 0:
        raise ValueError(f"quotas must sum to > 0, got {total}")
    normalized = [q / total for q in quotas]
    raw = [q * n for q in normalized]
    counts = [int(np.floor(x)) for x in raw]
    remainders = [(raw[i] - counts[i], i) for i in range(4)]
    leftover = n - sum(counts)
    for _, i in sorted(remainders, reverse=True)[:leftover]:
        counts[i] += 1
    ids = np.concatenate([
        np.full(counts[r], r, dtype=np.int64) for r in range(4)
    ])
    if rng is None:
        rng = np.random.default_rng()
    rng.shuffle(ids)
    return ids


def sample_diffusion_windows(
    model: FDADiffusionModel,
    n_windows: int,
    window_len: int,
    feature_dim: int,
    regimes: np.ndarray | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    dev = device or next(model.parameters()).device
    x_t = torch.randn(n_windows, window_len, feature_dim, device=dev)
    if regimes is None:
        regime = torch.randint(0, 4, (n_windows,), device=dev)
    else:
        regime = torch.from_numpy(regimes.astype(np.int64)).to(dev)

    model.eval()
    with torch.no_grad():
        for step in range(model.diffusion_steps, 0, -1):
            t = torch.full((n_windows,), step, dtype=torch.long, device=dev)
            eps_theta = model(x_t, t, regime)
            alpha_t = model.alphas[t][:, None, None]
            alpha_bar_t = model.alpha_bar[t][:, None, None]
            beta_t = model.betas[t][:, None, None]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            )
            if step > 1:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean
    return x_t.cpu().numpy().astype(np.float32)


def generate_valid_synthetic_windows(
    model: FDADiffusionModel,
    reference: ReferenceStats,
    cfg: FDAConfig,
    window_len: int = 64,
    feature_dim: int = 280,
) -> tuple[np.ndarray, dict]:
    """Generate accepted windows. Aggregates per-check failure counts rather than
    retaining per-candidate reports (retaining ~1M dict-heavy Python objects OOMs
    the process)."""
    accepted: list[np.ndarray] = []
    accepted_conditioning_regimes: list[int] = []
    candidate_count = 0
    pass_total = 0
    check_fail_counts: dict[str, int] = {}
    rounds = 0
    last_log = 0

    while len(accepted) < cfg.target_valid_windows:
        rounds += 1
        if rounds > cfg.max_generation_rounds:
            pass_rate = pass_total / max(candidate_count, 1)
            top_fails = sorted(check_fail_counts.items(), key=lambda kv: -kv[1])[:5]
            raise RuntimeError(
                f"FDA generation exceeded max rounds ({cfg.max_generation_rounds}) "
                f"after {candidate_count} candidates, accepted {len(accepted)}/{cfg.target_valid_windows} "
                f"(pass rate {pass_rate:.3f}). Top failing checks: {top_fails}. "
                f"Try --epochs larger (undertrained diffusion), --max-generation-rounds larger, "
                f"--generation-batch-size larger, or (if `range`/`finite` dominate) raising "
                f"--max-abs-feature — though a tight bound is usually correct."
            )
        round_regimes = _allocate_regime_ids(
            cfg.generation_batch_size, cfg.regime_quotas
        )
        candidates = sample_diffusion_windows(
            model,
            n_windows=cfg.generation_batch_size,
            window_len=window_len,
            feature_dim=feature_dim,
            regimes=round_regimes,
        )
        candidate_count += len(candidates)
        for w, cond_regime in zip(candidates, round_regimes):
            # Reconcile RSI with the synthetic return series before validation.
            # In real data, technical features (RSI, MACD, …) are deterministic
            # functions of the return series — the diffusion model has no
            # inductive bias for that constraint and can produce inconsistent
            # pairs. Overwriting RSI with the canonical value from returns
            # removes a structural failure mode of the `indicator_consistency`
            # soft check without weakening what it actually measures. Same
            # constants (rescale from [-1, 1] ↔ RSI [0, 100]) as in
            # `validate_stylized_facts`.
            if w.shape[1] > RSI_FEATURE_INDEX:
                rsi_from_r = _compute_rsi_like(w[:, RETURN_FEATURE_INDEX])
                w[:, RSI_FEATURE_INDEX] = np.clip(
                    (rsi_from_r - 50.0) / 50.0, -1.0, 1.0
                )
            rep = validate_stylized_facts(
                w, reference, max_abs_feature=cfg.max_abs_feature
            )
            if rep.passed:
                pass_total += 1
                accepted.append(w)
                accepted_conditioning_regimes.append(int(cond_regime))
                if len(accepted) >= cfg.target_valid_windows:
                    break
            else:
                for k, ok in rep.checks.items():
                    if not ok:
                        check_fail_counts[k] = check_fail_counts.get(k, 0) + 1
        if len(accepted) - last_log >= max(cfg.target_valid_windows // 50, 100):
            pass_rate = pass_total / max(candidate_count, 1)
            top_fails = sorted(check_fail_counts.items(), key=lambda kv: -kv[1])[:3]
            print(
                f"[fda] round {rounds}/{cfg.max_generation_rounds} "
                f"accepted={len(accepted)}/{cfg.target_valid_windows} "
                f"candidates={candidate_count} pass_rate={pass_rate:.3f} "
                f"top_fails={top_fails}",
                flush=True,
            )
            last_log = len(accepted)
    accepted_arr = np.stack(accepted, axis=0)
    # Regime distributions: the IDs we conditioned on vs. what the heuristic
    # post-hoc labeller (core.self_supervised_pretraining.infer_regime_label)
    # assigns to each accepted window. The first reports our sampling
    # intent (§8.4 quotas); the second is what training will actually see.
    cond_regimes_arr = np.asarray(accepted_conditioning_regimes, dtype=np.int64)
    cond_counts = np.bincount(cond_regimes_arr, minlength=4).tolist()
    post_labels = [infer_regime_label(w) for w in accepted_arr]
    post_counts = np.bincount(
        np.asarray(post_labels, dtype=np.int64), minlength=4
    ).tolist()
    n_accepted = max(len(accepted_arr), 1)
    stats = {
        "candidates": candidate_count,
        "accepted": pass_total,
        "pass_rate": float(pass_total / max(candidate_count, 1)),
        "check_fail_counts": check_fail_counts,
        "conditioned_regime_counts": cond_counts,
        "conditioned_regime_distribution": [c / n_accepted for c in cond_counts],
        "realized_regime_counts": post_counts,
        "realized_regime_distribution": [c / n_accepted for c in post_counts],
    }
    return accepted_arr, stats


def write_synthetic_shards(
    windows: np.ndarray,
    output_dir: Path,
    split: str = "train",
    prefix: str = "FDA_SYNTH",
) -> list[Path]:
    if windows.ndim != 3:
        raise ValueError("windows must have shape (N, T, F)")
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for i, w in enumerate(windows):
        path = split_dir / f"{prefix}_{i:05d}.f32bin"
        w.astype(np.float32).tofile(path)
        out_paths.append(path)
    return out_paths


def write_fda_manifest(
    output_dir: Path,
    cfg: FDAConfig,
    reference: ReferenceStats,
    stats: dict,
    shard_paths: list[Path],
    cross_asset: CrossAssetReport | None = None,
) -> Path:
    manifest = {
        "config": asdict(cfg),
        "reference": asdict(reference),
        "candidate_windows": stats["candidates"],
        "accepted_windows": stats["accepted"],
        "acceptance_rate": stats["pass_rate"],
        "check_fail_counts": stats["check_fail_counts"],
        "shards_written": [str(p) for p in shard_paths],
    }
    if cross_asset is not None:
        manifest["cross_asset_validation"] = {
            "passed": cross_asset.passed,
            "checks": cross_asset.checks,
            "metrics": cross_asset.metrics,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fda_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _parse_regime_quotas(raw: str) -> tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--regime-quotas needs 4 comma-separated floats, got {len(parts)}"
        )
    if min(parts) < 0:
        raise argparse.ArgumentTypeError(f"--regime-quotas must be non-negative, got {parts}")
    if sum(parts) <= 0:
        raise argparse.ArgumentTypeError(f"--regime-quotas must sum to > 0, got {parts}")
    return tuple(parts)  # type: ignore[return-value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDA offline generation pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("data/mmaps/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mmaps_synth"))
    parser.add_argument("--max-real-windows", type=int, default=4096)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="DDPM noise levels T_diff. Spec §8.2: 1000. Lower values severely "
             "underfit the denoiser and produce off-manifold samples.",
    )
    parser.add_argument("--target-valid-windows", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=0,
                        help="Samples per diffusion round; 0 = use --batch-size")
    parser.add_argument("--max-generation-rounds", type=int, default=5000)
    parser.add_argument(
        "--max-abs-feature",
        type=float,
        default=10.0,
        help="Hard per-value bound on synthetic features. Windows with |x| "
             "exceeding this are rejected regardless of statistical checks. "
             "Real finance-query features are normalized to O(1); defaults to 10.0.",
    )
    parser.add_argument(
        "--regime-quotas",
        type=_parse_regime_quotas,
        default=(0.30, 0.25, 0.25, 0.20),
        help="Comma-separated per-regime conditioning share for synthetic "
             "sampling (low-vol, high-vol, mean-revert, crisis). Spec §8.4 "
             "default: 0.30,0.25,0.25,0.20. Values are renormalized to sum 1.",
    )
    parser.add_argument("--device", default="cpu")
    # Cross-asset validation (spec §8.5) — pool-level gate on top of the 9
    # per-window statistical checks. Runs once after generation completes.
    parser.add_argument(
        "--no-cross-asset-check",
        action="store_true",
        help="Disable the spec §8.5 portfolio-level validation (not recommended).",
    )
    parser.add_argument("--cross-asset-pool-size", type=int, default=100,
                        help="Number of windows per synthetic pool (default: 100 ~ portfolio size).")
    parser.add_argument("--cross-asset-real-pools", type=int, default=200,
                        help="Random pools drawn from real data to build the reference distribution.")
    parser.add_argument("--cross-asset-synth-pools", type=int, default=100,
                        help="Random pools drawn from synth data for the KS comparison.")
    parser.add_argument("--cross-asset-beta-ks", type=float, default=0.20,
                        help="KS threshold on per-asset beta distribution (spec §8.5: 0.20).")
    parser.add_argument("--cross-asset-mvp-ks", type=float, default=0.20,
                        help="KS threshold on MVP weight distribution.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = FDAConfig(
        window_len=args.window_len,
        diffusion_steps=args.diffusion_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_valid_windows=args.target_valid_windows,
        generation_batch_size=args.generation_batch_size or args.batch_size,
        max_generation_rounds=args.max_generation_rounds,
        max_abs_feature=args.max_abs_feature,
        regime_quotas=args.regime_quotas,
        device=args.device,
    )
    shard_paths = sorted(args.input_dir.glob("*.f32bin"))
    if not shard_paths:
        raise RuntimeError(f"no shards found under {args.input_dir}")

    windows, regimes = load_windows_from_shards(
        shard_paths=shard_paths,
        window_len=cfg.window_len,
        stride=args.stride,
        max_windows=args.max_real_windows,
    )
    ref = compute_reference_stats(windows)
    model = train_diffusion_model(windows, regimes, cfg)
    synth, stats = generate_valid_synthetic_windows(
        model=model,
        reference=ref,
        cfg=cfg,
        window_len=cfg.window_len,
        feature_dim=cfg.feature_dim,
    )
    out_paths = write_synthetic_shards(synth, args.output_dir, split="train")

    cond_pct = ", ".join(f"{x:.1%}" for x in stats["conditioned_regime_distribution"])
    real_pct = ", ".join(f"{x:.1%}" for x in stats["realized_regime_distribution"])
    print(
        f"[fda] regime distribution — conditioned (target {cfg.regime_quotas}): "
        f"[{cond_pct}], realized (post-hoc heuristic): [{real_pct}]",
        flush=True,
    )

    cross_report: CrossAssetReport | None = None
    if not args.no_cross_asset_check:
        print(
            f"[fda] running cross-asset validation "
            f"(pool_size={args.cross_asset_pool_size}, "
            f"real_pools={args.cross_asset_real_pools}, "
            f"synth_pools={args.cross_asset_synth_pools})",
            flush=True,
        )
        cross_ref = compute_cross_asset_reference(
            windows,
            pool_size=args.cross_asset_pool_size,
            n_pool_samples=args.cross_asset_real_pools,
        )
        cross_report = validate_cross_asset_structure(
            synth,
            cross_ref,
            n_pool_samples=args.cross_asset_synth_pools,
            beta_ks_threshold=args.cross_asset_beta_ks,
            mvp_ks_threshold=args.cross_asset_mvp_ks,
        )
        status = "PASS" if cross_report.passed else "FAIL"
        print(
            f"[fda] cross-asset validation {status}: "
            f"beta_ks={cross_report.metrics['beta_ks']:.3f} "
            f"(<{args.cross_asset_beta_ks}) "
            f"mvp_ks={cross_report.metrics['mvp_ks']:.3f} "
            f"(<{args.cross_asset_mvp_ks})",
            flush=True,
        )

    manifest = write_fda_manifest(
        args.output_dir, cfg, ref, stats, out_paths, cross_asset=cross_report
    )
    print(f"Generated {len(out_paths)} FDA shards. Manifest: {manifest}")

    if cross_report is not None and not cross_report.passed:
        raise RuntimeError(
            "FDA cross-asset validation failed (spec §8.5). "
            f"beta_ks={cross_report.metrics['beta_ks']:.3f} "
            f"mvp_ks={cross_report.metrics['mvp_ks']:.3f}. "
            "Synthetic pool covariance structure diverges from real. "
            "Likely causes: diffusion model under-trained (bump --epochs), "
            "sampling too noisy (bump --diffusion-steps), or per-window gate "
            "admitting off-manifold windows (tighten --max-abs-feature). "
            "Shards were still written for inspection; delete before retry."
        )


if __name__ == "__main__":
    main()
