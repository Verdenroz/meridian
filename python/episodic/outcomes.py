"""What happened after an analog date: the reported nonparametric prior.

Forward-looking by construction — legal only because analogs sit at least an
exclusion window (≥ horizon) in the past, so their futures are known history
at query time. Never computed for the query date itself.
"""

from __future__ import annotations

import numpy as np
from harness.panel import Panel

TRADING_DAYS_PER_YEAR = 252
HORIZON = 21


def forward_outcomes(
    panel: Panel, index: int, horizon: int = HORIZON, symbol: str = "SPY"
) -> dict[str, float] | None:
    """Return, realized vol, and max drawdown over index+1..index+horizon;
    None when the window is incomplete or has missing returns."""
    window = panel.returns[symbol][index + 1 : index + 1 + horizon]
    if len(window) < horizon or any(r is None for r in window):
        return None
    rets = np.array(window, dtype=float)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    peak = np.maximum.accumulate(curve)
    return {
        "fwd_return": float(curve[-1] - 1.0),
        "fwd_vol": float(np.sqrt(np.mean(rets**2) * TRADING_DAYS_PER_YEAR)),
        "fwd_max_drawdown": float((curve / peak - 1.0).min()),
    }
