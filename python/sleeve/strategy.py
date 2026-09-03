"""SleeveVolTarget: the null's vol-target shell with a risk-weighted stock
basket in place of the index.

Eligibility is causal: a name joins once it carries `min_history`
consecutive finite returns and a finite forecast. Basket weights come from
the sizing rule (inverse forecast vol under a per-name cap, or equal), the
basket's forecast vol from forecast vols on the diagonal and a Ledoit–Wolf
shrunk correlation refit every `corr_refit_days` or whenever the eligible
set changes, and the equity fraction is target_vol / basket_vol capped at 1.
Below `min_names` eligible names the equity bucket falls back to the index
sized on its own EWMA vol, so early windows never trade a thin basket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from harness.panel import Panel
from harness.view import HistoryView
from spine.vol import ewma_vol_series

from sleeve.forecasters import VolForecaster, finite_run_lengths, returns_matrix
from sleeve.risk import (
    basket_vol,
    capped_inverse_vol_weights,
    equal_weights,
    shrunk_correlation,
)

SIZINGS = ("inverse_vol", "equal")


@dataclass(frozen=True)
class SleeveData:
    """Precomputed, window-independent inputs shared by every contender."""

    symbols: tuple[str, ...]
    market_symbol: str
    returns: np.ndarray
    run_lengths: np.ndarray
    market_returns: np.ndarray
    market_vol: np.ndarray


def sleeve_data(panel: Panel, market_symbol: str, universe: tuple[str, ...]) -> SleeveData:
    returns = returns_matrix(panel.returns, universe)
    market = returns_matrix(panel.returns, (market_symbol,))[:, 0]
    return SleeveData(
        universe, market_symbol, returns, finite_run_lengths(returns), market,
        ewma_vol_series(market),
    )


class SleeveVolTarget:
    def __init__(
        self,
        data: SleeveData,
        forecaster: VolForecaster,
        *,
        sizing: str,
        target_vol: float,
        band: float,
        max_name_weight: float,
        min_names: int,
        min_history: int,
        corr_window: int,
        corr_refit_days: int,
    ) -> None:
        if sizing not in SIZINGS:
            raise ValueError(f"unknown sizing {sizing}")
        self.data = data
        self.forecaster = forecaster
        self.sizing = sizing
        self.target_vol = target_vol
        self.band = band
        self.max_name_weight = max_name_weight
        self.min_names = min_names
        self.min_history = min_history
        self.corr_window = corr_window
        self.corr_refit_days = corr_refit_days
        self._forecast: np.ndarray | None = None
        self._corr: np.ndarray | None = None
        self._corr_key: tuple[int, ...] = ()
        self._corr_t = -1
        self._last_target: dict[str, float] | None = None

    def fit(self, view: HistoryView) -> None:
        self._forecast = self.forecaster.fit(view.t)
        self._corr = None
        self._corr_key = ()
        self._corr_t = -1
        self._last_target = None

    def _eligible(self, t: int) -> np.ndarray:
        assert self._forecast is not None
        vols = self._forecast[t]
        return np.flatnonzero(
            (self.data.run_lengths[t] >= self.min_history) & np.isfinite(vols) & (vols > 0)
        )

    def _correlation(self, t: int, idx: np.ndarray) -> np.ndarray:
        key = tuple(int(i) for i in idx)
        stale = t - self._corr_t >= self.corr_refit_days
        if self._corr is None or key != self._corr_key or stale:
            window = self.data.returns[t - self.corr_window + 1 : t + 1][:, idx]
            self._corr = shrunk_correlation(window)
            self._corr_key, self._corr_t = key, t
        return self._corr

    def _basket_target(self, t: int, idx: np.ndarray) -> dict[str, float]:
        assert self._forecast is not None
        vols = self._forecast[t, idx]
        weights = (
            capped_inverse_vol_weights(vols, self.max_name_weight)
            if self.sizing == "inverse_vol"
            else equal_weights(len(idx))
        )
        vol = basket_vol(weights, vols, self._correlation(t, idx))
        equity = min(1.0, self.target_vol / vol)
        return {self.data.symbols[i]: equity * w for i, w in zip(idx, weights)}

    def _index_target(self, t: int) -> dict[str, float] | None:
        vol = self.data.market_vol[t]
        if not np.isfinite(vol):
            return None
        return {self.data.market_symbol: min(1.0, self.target_vol / float(vol))}

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        t = view.t
        idx = self._eligible(t)
        target = self._basket_target(t, idx) if len(idx) >= self.min_names else self._index_target(t)
        if target is None:
            return None
        if self._last_target is not None:
            drift = max(
                abs(target.get(s, 0.0) - self._last_target.get(s, 0.0))
                for s in set(target) | set(self._last_target)
            )
            if drift < self.band:
                return None
        self._last_target = target
        return target
