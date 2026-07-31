"""Episodic state vector: causal daily features for analog retrieval.

Row t is computable at the close of day t. v1 uses the observable drivers of
the regime posterior rather than the posterior itself — raw features keep the
matrix fit-free and point-in-time. News intensity joins once the archive has
enough history to matter; the correlation summary needs the full core
universe, so the searchable buffer starts once GLD has a correlation window
of data (early 2005) and grows every day Meridian lives.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from harness.panel import Panel
from spine.correlation import rolling_correlation
from spine.features import _ffill_series, _macro_by_date, build_features

STATE_SYMBOLS = ("SPY", "TLT", "GLD")
STATE_FEATURES = (
    "log_realized_vol",
    "log_vix",
    "curve_slope",
    "drawdown",
    "credit_spread",
    "mean_pairwise_corr",
)
CORR_WINDOW = 60


def _mean_pairwise_corr(panel: Panel) -> np.ndarray:
    series = {
        s: np.array([np.nan if r is None else r for r in panel.returns[s]])
        for s in STATE_SYMBOLS
    }
    # All pairs required: partial averages would change the feature's meaning
    # over time and break distance comparability across the buffer.
    stacked = np.vstack(
        [
            rolling_correlation(series[a], series[b], CORR_WINDOW)
            for a, b in combinations(STATE_SYMBOLS, 2)
        ]
    )
    return stacked.mean(axis=0)


def build_state_matrix(panel: Panel, store_dir: Path) -> np.ndarray:
    """n_days × 6 matrix aligned to panel.dates; NaN rows until warm-up."""
    base = build_features(panel, store_dir)
    spread = _ffill_series(panel.dates, _macro_by_date(store_dir, "BAA10Y", "value"))
    corr = _mean_pairwise_corr(panel)
    return np.column_stack([base, spread, corr])
