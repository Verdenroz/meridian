"""Walk-forward evaluation: expanding origin, out-of-sample stitching.

Every window gets a freshly constructed strategy fitted only on data before
its test span; strategies may read all *past* history through the view but can
never see past the current day. Each window enters positions fresh, so entry
costs are counted per window — slightly conservative versus continuous
operation, never flattering.
"""

from __future__ import annotations

from collections.abc import Callable

from harness.engine import Strategy, run_backtest
from harness.panel import Panel

MIN_TRAIN_DAYS = 5 * 252
TEST_DAYS = 252


def walk_forward_splits(
    first_index: int,
    n_days: int,
    min_train_days: int = MIN_TRAIN_DAYS,
    test_days: int = TEST_DAYS,
) -> list[tuple[int, int, int]]:
    """(fit_end, test_start, test_end) index triples; test spans are disjoint
    and cover everything after the initial training period."""
    splits: list[tuple[int, int, int]] = []
    test_start = first_index + min_train_days
    while test_start < n_days:
        test_end = min(test_start + test_days - 1, n_days - 1)
        splits.append((test_start - 1, test_start, test_end))
        test_start = test_end + 1
    return splits


def run_walk_forward(
    panel: Panel,
    strategy_factory: Callable[[], Strategy],
    first_index: int,
    min_train_days: int = MIN_TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    cost_bps: dict[str, float] | None = None,
) -> dict:
    splits = walk_forward_splits(first_index, panel.n_days, min_train_days, test_days)
    if not splits:
        raise ValueError("not enough history for a single walk-forward window")
    dates: list[str] = []
    returns: list[float] = []
    windows: list[dict] = []
    for fit_end, test_start, test_end in splits:
        result = run_backtest(
            panel, strategy_factory(), start=fit_end, end=test_end, cost_bps=cost_bps
        )
        assert result["dates"][0] == panel.dates[test_start]
        dates.extend(result["dates"])
        returns.extend(result["returns"])
        windows.append(
            {
                "test_start": panel.dates[test_start],
                "test_end": panel.dates[test_end],
                "final_value": result["final_value"],
                "turnover_oneway": result["turnover_oneway"],
            }
        )
    return {"dates": dates, "returns": returns, "windows": windows}
