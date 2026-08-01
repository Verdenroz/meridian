"""Causal signal mechanics shared by the pre-registered races.

Frozen mechanics, not tunable knobs: trailing z-scores whose baseline
excludes the current value, burst-to-brake activation windows, and the
braked vol-target challenger shape. Races own their thresholds and pass
them in — nothing here defaults a pre-registered constant.
"""

from __future__ import annotations

import numpy as np

from harness.strategies import VolTarget
from harness.view import HistoryView


def causal_zscores(
    values: np.ndarray, window: int = 252, min_obs: int = 120
) -> np.ndarray:
    """z[t] compares values[t] to the trailing `window` observations before
    t — same convention as the event study's intensity z-scores."""
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        if not np.isfinite(values[t]):
            continue
        past = values[max(0, t - window) : t]
        past = past[np.isfinite(past)]
        if len(past) < min_obs:
            continue
        std = past.std()
        if std > 0:
            out[t] = (values[t] - past.mean()) / std
    return out


def brake_series(z: np.ndarray, burst_z: float, brake_days: int) -> np.ndarray:
    """active[t] iff any of z[t-brake_days+1 .. t] >= burst_z — causal."""
    burst = np.isfinite(z) & (z >= burst_z)
    active = np.zeros(len(z), dtype=bool)
    for t in np.flatnonzero(burst):
        active[t : t + brake_days] = True
    return active


class BrakeVolTarget(VolTarget):
    """The null vol-target with exposure scaled down while the brake is on."""

    def __init__(
        self, symbol: str, active: np.ndarray, brake_scale: float
    ) -> None:
        super().__init__(symbol)
        self.active = active
        self.brake_scale = brake_scale

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        vol = self._ewma_vol(view.returns_window(self.symbol, self.window))
        weight = 0.0 if vol is None else min(1.0, self.target_vol / vol)
        if self.active[view.t]:
            weight *= self.brake_scale
        if self._last_target is not None and abs(weight - self._last_target) < self.band:
            return None
        self._last_target = weight
        return {self.symbol: weight}
