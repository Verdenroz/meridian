"""Pure window statistics for the behavior-match evaluation.

All functions operate on decision rows shaped like the shadow log
(`weights`, `cash_weight`, `meta`) so the same code measures both the
logged shadow window and the reference replay — comparability by
construction, not by convention.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def max_weight_diff(a: dict[str, float], b: dict[str, float]) -> float:
    symbols = set(a) | set(b)
    return max((abs(a.get(s, 0.0) - b.get(s, 0.0)) for s in symbols), default=0.0)


def mean_gross_exposure(rows: list[dict]) -> float:
    return float(np.mean([sum(r["weights"].values()) for r in rows]))


def turnover_annualized(rows: list[dict]) -> float:
    """One-way decision turnover: sum of |Δweight| between consecutive
    emitted targets, annualized. Non-cash legs only, like the engine."""
    if len(rows) < 2:
        return 0.0
    total = 0.0
    for prev, cur in pairwise(rows):
        symbols = set(prev["weights"]) | set(cur["weights"])
        total += sum(
            abs(cur["weights"].get(s, 0.0) - prev["weights"].get(s, 0.0))
            for s in symbols
        )
    return total * 252.0 / (len(rows) - 1)


def brake_active_frac(rows: list[dict]) -> float:
    return float(np.mean([bool(r["meta"].get("brake_active")) for r in rows]))


def missing_and_max_gap(
    logged_dates: set[str], trading_days: list[str]
) -> tuple[int, int]:
    """(count of trading days with no row, longest consecutive missing run)."""
    missing = 0
    max_gap = 0
    run = 0
    for day in trading_days:
        if day in logged_dates:
            run = 0
        else:
            missing += 1
            run += 1
            max_gap = max(max_gap, run)
    return missing, max_gap


def rolling_windows(rows: list[dict], length: int) -> list[list[dict]]:
    if length < 1 or length > len(rows):
        return []
    return [rows[i : i + length] for i in range(len(rows) - length + 1)]


def percentile_band(
    values: list[float], lo_pct: float, hi_pct: float
) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, lo_pct)), float(np.percentile(arr, hi_pct))
