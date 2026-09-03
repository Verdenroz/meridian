"""Per-name vol forecasters over a T×N return matrix, causal by construction.

Every forecast matrix follows the spine convention: out[t, j] uses returns
[..t] of name j and is the forecast standing for day t+1. NaN where a name
has no history yet or a trailing window is incomplete. The pooled ridge is
the first learned per-name challenger: one cross-sectional model fitted on
every (day, name) row of the training span, refit per walk-forward window.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from spine.vol import TRADING_DAYS_PER_YEAR, ewma_vol_series
from vol_race.forecasters import har_lite_vol_series

REALIZED_WINDOWS = (5, 22, 63)
MARKET_REALIZED_WINDOW = 22
RIDGE_MIN_ROWS = 500


class VolForecaster(Protocol):
    def fit(self, fit_end: int) -> np.ndarray:
        """Forecast matrix using only data up to `fit_end` for any fitting."""
        ...


class StaticForecast:
    """A parameter-free forecast matrix; fitting is a no-op."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def fit(self, fit_end: int) -> np.ndarray:
        return self.matrix


def returns_matrix(returns_by_symbol: dict[str, list[float | None]], symbols: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [
            np.array([np.nan if r is None else r for r in returns_by_symbol[s]], dtype=float)
            for s in symbols
        ]
    )


def ewma_vol_matrix(returns: np.ndarray) -> np.ndarray:
    return np.column_stack([ewma_vol_series(returns[:, j]) for j in range(returns.shape[1])])


def har_lite_vol_matrix(returns: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [har_lite_vol_series(returns[:, j]) for j in range(returns.shape[1])]
    )


def trailing_realized_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Annualized realized vol over t-window+1..t; NaN when the window has a gap."""
    t_obs = returns.shape[0]
    out = np.full_like(returns, np.nan, dtype=float)
    if t_obs < window:
        return out
    sq = sliding_window_view(np.square(returns), window, axis=0)
    out[window - 1 :] = np.sqrt(sq.mean(axis=-1) * TRADING_DAYS_PER_YEAR)
    return out


def forward_realized_vol_matrix(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Annualized realized vol over t+1..t+horizon — evaluation target only."""
    t_obs = returns.shape[0]
    out = np.full_like(returns, np.nan, dtype=float)
    if t_obs <= horizon:
        return out
    sq = sliding_window_view(np.square(returns[1:]), horizon, axis=0)
    out[: t_obs - horizon] = np.sqrt(sq.mean(axis=-1) * TRADING_DAYS_PER_YEAR)
    return out


def finite_run_lengths(returns: np.ndarray) -> np.ndarray:
    """out[t, j] = number of consecutive finite returns of name j ending at t."""
    out = np.zeros(returns.shape, dtype=int)
    run = np.zeros(returns.shape[1], dtype=int)
    finite = np.isfinite(returns)
    for t in range(returns.shape[0]):
        run = np.where(finite[t], run + 1, 0)
        out[t] = run
    return out


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(x), np.nan)


def ridge_features(returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
    """T×N×F causal log-vol features: own EWMA, own realized 5/22/63d,
    market EWMA, market realized 22d."""
    n = returns.shape[1]
    own = [ewma_vol_matrix(returns)] + [
        trailing_realized_vol(returns, w) for w in REALIZED_WINDOWS
    ]
    market_cols = [
        ewma_vol_series(market_returns),
        trailing_realized_vol(market_returns[:, None], MARKET_REALIZED_WINDOW)[:, 0],
    ]
    market = [np.repeat(m[:, None], n, axis=1) for m in market_cols]
    return np.stack([_safe_log(f) for f in own + market], axis=-1)


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge on standardized columns with an unpenalized intercept; returns
    (coefficients in standardized units, intercept)."""
    n_features = x.shape[1]
    design = np.column_stack([x, np.ones(len(x))])
    penalty = lam * np.eye(n_features + 1)
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta[:-1], float(beta[-1])


class PooledRidgeForecaster:
    """Cross-sectional ridge from log-vol features to log forward realized
    vol, pooled over every (day, name) training row whose forward window
    closes at or before the fit point. Forecasts are exp(prediction)."""

    def __init__(
        self, features: np.ndarray, forward_vol: np.ndarray, horizon: int, lam: float
    ) -> None:
        self.features = features
        self.log_target = _safe_log(forward_vol)
        self.horizon = horizon
        self.lam = lam
        self.coef: np.ndarray | None = None

    def fit(self, fit_end: int) -> np.ndarray:
        end = fit_end - self.horizon + 1
        n_features = self.features.shape[-1]
        x = self.features[:end].reshape(-1, n_features)
        y = self.log_target[:end].reshape(-1)
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if mask.sum() < RIDGE_MIN_ROWS:
            raise ValueError("not enough closed training rows to fit the pooled ridge")
        mu = x[mask].mean(axis=0)
        sd = np.maximum(x[mask].std(axis=0), 1e-12)
        coef, intercept = fit_ridge((x[mask] - mu) / sd, y[mask], self.lam)
        self.coef = coef
        x_all = (self.features.reshape(-1, n_features) - mu) / sd
        pred = np.exp(x_all @ coef + intercept)
        pred = np.where(np.isfinite(x_all).all(axis=1), pred, np.nan)
        return pred.reshape(self.features.shape[:2])


def walk_forward_forecasts(
    forecaster: VolForecaster, splits: list[tuple[int, int, int]], shape: tuple[int, int]
) -> np.ndarray:
    """Out-of-sample stitched forecast matrix: each test span is filled from
    a fit at its own fit boundary; everything outside the test spans is NaN."""
    out = np.full(shape, np.nan)
    for fit_end, test_start, test_end in splits:
        pred = forecaster.fit(fit_end)
        out[test_start : test_end + 1] = pred[test_start : test_end + 1]
    return out
