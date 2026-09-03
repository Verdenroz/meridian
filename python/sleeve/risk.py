"""Sleeve risk model: forecast vols on the diagonal, shrunk correlation off
it, and the risk-weighted sizing rules the charter permits (no return view)."""

from __future__ import annotations

import numpy as np
from spine.correlation import ledoit_wolf_constant_corr

CAP_ITERATIONS = 50


def capped_inverse_vol_weights(vols: np.ndarray, cap: float) -> np.ndarray:
    """Inverse-vol weights summing to 1 with no name above `cap`, excess
    redistributed to uncapped names in proportion. An infeasible cap
    (n*cap < 1) degrades to equal weight."""
    n = len(vols)
    if n * cap < 1.0:
        return np.full(n, 1.0 / n)
    raw = 1.0 / vols
    weights = raw / raw.sum()
    for _ in range(CAP_ITERATIONS):
        over = weights > cap + 1e-12
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights = np.where(over, cap, weights)
        free = ~over
        weights = np.where(free, weights + excess * weights / weights[free].sum(), weights)
    return weights


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def shrunk_correlation(window_returns: np.ndarray) -> np.ndarray:
    cov, _ = ledoit_wolf_constant_corr(window_returns)
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def basket_vol(weights: np.ndarray, vols: np.ndarray, corr: np.ndarray) -> float:
    scaled = weights * vols
    return float(np.sqrt(scaled @ corr @ scaled))
