"""Challenger vol forecasters: parameter-free, causal by construction.

Every series follows the incumbent's convention: out[t] uses data <= t and
is the forecast standing for day t+1. Candidates are deliberately fitted
with nothing — no tuned coefficients — so stage 1 compares information
content, not optimization effort. Scale bias (VIX's risk premium) is
irrelevant to stage 1 (correlation is affine-invariant) and handled in
stage 2 by train-window calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from spine.features import _ffill_series, _macro_by_date
from spine.vol import TRADING_DAYS_PER_YEAR, ewma_vol_series

HAR_WINDOWS = (1, 5, 22)
HAR_MIN_OBS = 22


def vix_vol_series(store_dir: Path, dates: list[str]) -> np.ndarray:
    """VIX close as an annualized vol forecast (forward-filled, /100)."""
    return _ffill_series(dates, _macro_by_date(store_dir, "VIXCLS", "value")) / 100.0


def har_lite_vol_series(returns: np.ndarray) -> np.ndarray:
    """Equal-weight HAR: mean of 1d/5d/22d trailing realized variances,
    annualized. Parameter-free stand-in for fitted HAR-RV."""
    n = len(returns)
    sq = np.square(returns)
    out = np.full(n, np.nan)
    for t in range(n):
        window = sq[max(0, t - HAR_MIN_OBS + 1) : t + 1]
        finite = window[np.isfinite(window)]
        if len(finite) < HAR_MIN_OBS:
            continue
        components = [np.mean(sq[t - w + 1 : t + 1]) for w in HAR_WINDOWS]
        out[t] = float(np.sqrt(np.mean(components) * TRADING_DAYS_PER_YEAR))
    return out


def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def build_candidates(
    store_dir: Path, dates: list[str], returns: np.ndarray
) -> dict[str, np.ndarray]:
    """The fixed candidate set. 'ewma' is the incumbent and the in-run bar."""
    ewma = ewma_vol_series(returns)
    vix = vix_vol_series(store_dir, dates)
    har = har_lite_vol_series(returns)
    return {
        "ewma": ewma,
        "vix": vix,
        "har_lite": har,
        "blend_ewma_vix": blend(ewma, vix),
        "blend_har_vix": blend(har, vix),
    }
