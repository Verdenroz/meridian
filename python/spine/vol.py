"""Volatility: RiskMetrics EWMA incumbent and evaluation utilities.

All series here are causal: out[t] is computable at the close of day t.
Forward realized vol looks into the future by construction — it exists only to
score forecasts, never as a model input. Measured bar for challengers:
0.67 forecast/realized correlation (probing result); naive is 0.60.
"""

from __future__ import annotations

import math

import numpy as np

TRADING_DAYS_PER_YEAR = 252
EWMA_LAMBDA = 0.94
MIN_OBS = 60


def ewma_vol_series(
    returns: np.ndarray, lam: float = EWMA_LAMBDA, min_obs: int = MIN_OBS
) -> np.ndarray:
    """Annualized EWMA vol forecast; out[t] uses returns[..t] and is the
    forecast standing for day t+1. NaN until min_obs returns are seen
    (NaN returns before inception are passed through)."""
    n = len(returns)
    out = np.full(n, np.nan)
    variance = 0.0
    seen = 0
    for t in range(n):
        r = returns[t]
        if math.isnan(r):
            continue
        variance = r * r if seen == 0 else lam * variance + (1.0 - lam) * r * r
        seen += 1
        if seen >= min_obs:
            out[t] = math.sqrt(variance * TRADING_DAYS_PER_YEAR)
    return out


def forward_realized_vol(returns: np.ndarray, horizon: int = 21) -> np.ndarray:
    """Annualized realized vol over t+1..t+horizon — evaluation target only."""
    n = len(returns)
    out = np.full(n, np.nan)
    for t in range(n - horizon):
        window = returns[t + 1 : t + 1 + horizon]
        if np.isnan(window).any():
            continue
        out[t] = float(np.sqrt(np.mean(window**2) * TRADING_DAYS_PER_YEAR))
    return out


def forecast_correlation(forecast: np.ndarray, realized: np.ndarray) -> float:
    mask = ~np.isnan(forecast) & ~np.isnan(realized)
    if mask.sum() < 2:
        raise ValueError("not enough overlapping observations")
    return float(np.corrcoef(forecast[mask], realized[mask])[0, 1])
