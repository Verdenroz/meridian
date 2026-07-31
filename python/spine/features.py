"""Regime features: causal by construction — row t uses only data ≤ t.

Feature 0 must stay a volatility measure: the HMM orders states by it, and the
allocator reads the last state as "stress".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from harness.panel import Panel
from market_data.store import read_snapshot

from spine.vol import ewma_vol_series

FEATURE_NAMES = ("log_realized_vol", "log_vix", "curve_slope", "drawdown")


def _macro_by_date(store_dir: Path, name: str, field: str) -> dict[str, float]:
    doc = read_snapshot(store_dir, "macro", name)
    if doc is None:
        raise FileNotFoundError(f"macro snapshot {name} missing; run market_data fetch")
    out: dict[str, float] = {}
    for row in doc["data"]["rows"]:
        value = row.get(field)
        if value is not None:
            out[row["date"]] = float(value)
    return out


def _ffill_series(dates: list[str], by_date: dict[str, float]) -> np.ndarray:
    out = np.full(len(dates), np.nan)
    last = np.nan
    for i, date in enumerate(dates):
        last = by_date.get(date, last)
        out[i] = last
    return out


def build_features(
    panel: Panel, store_dir: Path, market_symbol: str = "SPY"
) -> np.ndarray:
    """n_days × 4 matrix aligned to panel.dates; NaN rows until warm-up."""
    returns = np.array(
        [np.nan if r is None else r for r in panel.returns[market_symbol]]
    )
    realized = ewma_vol_series(returns)

    vix = _ffill_series(panel.dates, _macro_by_date(store_dir, "VIXCLS", "value"))
    # 10Y-3M spread straight from FRED — the treasury endpoint lacks y10 pre-2001.
    slope = _ffill_series(panel.dates, _macro_by_date(store_dir, "T10Y3M", "value"))

    prices = np.array(
        [np.nan if p is None else p for p in panel.adj_close[market_symbol]]
    )
    peak = np.fmax.accumulate(np.where(np.isnan(prices), -np.inf, prices))
    with np.errstate(invalid="ignore"):
        drawdown = np.where(peak > 0, prices / peak - 1.0, np.nan)

    with np.errstate(invalid="ignore"):
        features = np.column_stack([np.log(realized), np.log(vix), slope, drawdown])
    return features
