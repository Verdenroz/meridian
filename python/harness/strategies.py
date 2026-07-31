"""Null baselines: buy-and-hold and SPY + vol-target. These are the bar.

Constants are classical conventions (RiskMetrics lambda, 15% target), fixed —
not tuned — so the null stays simple and unimpeachable.
"""

from __future__ import annotations

import math

from harness.panel import TRADING_DAYS_PER_YEAR
from harness.view import HistoryView


class BuyHold:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._entered = False

    def fit(self, view: HistoryView) -> None:
        self._entered = False

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        if self._entered:
            return None
        self._entered = True
        return {self.symbol: 1.0}


class VolTarget:
    """Scale exposure to target_vol / EWMA realized vol, capped at 1 (no
    leverage); remainder earns cash. Rebalances only when the target moves by
    more than `band` — daily micro-trades would just donate spread."""

    def __init__(
        self,
        symbol: str,
        target_vol: float = 0.15,
        ewma_lambda: float = 0.94,
        window: int = 512,
        min_obs: int = 60,
        band: float = 0.02,
    ) -> None:
        self.symbol = symbol
        self.target_vol = target_vol
        self.ewma_lambda = ewma_lambda
        self.window = window
        self.min_obs = min_obs
        self.band = band
        self._last_target: float | None = None

    def fit(self, view: HistoryView) -> None:
        self._last_target = None

    def _ewma_vol(self, returns: list[float]) -> float | None:
        if len(returns) < self.min_obs:
            return None
        variance = returns[0] * returns[0]
        for r in returns[1:]:
            variance = self.ewma_lambda * variance + (1.0 - self.ewma_lambda) * r * r
        return math.sqrt(variance * TRADING_DAYS_PER_YEAR)

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        vol = self._ewma_vol(view.returns_window(self.symbol, self.window))
        weight = 0.0 if vol is None else min(1.0, self.target_vol / vol)
        if self._last_target is not None and abs(weight - self._last_target) < self.band:
            return None
        self._last_target = weight
        return {self.symbol: weight}
