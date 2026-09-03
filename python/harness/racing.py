"""Race runner: walk-forward + metrics for a contender lineup, plus the
standard report table and JSON-safe verdict serialization.

Every pre-registered race is the same shape — a dict of labeled strategy
factories run through the one walk-forward engine, scored after costs on a
common span. This module owns that shape; each race module owns only its
hypothesis: features, challengers, and frozen constants.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import Panel
from harness.walkforward import run_walk_forward

METRIC_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day")


def run_contender(
    panel: Panel,
    factory: Callable[[], Strategy],
    first: int,
    oos_start: str | None = None,
    cost_bps: dict[str, float] | None = None,
) -> dict:
    """Walk-forward run + metrics; `oos_start` restricts the scored span
    (races whose panel opens long before their pre-registered OOS window).
    `cost_bps` overrides the shared cost table for races on instruments it
    does not list."""
    result = run_walk_forward(panel, factory, first, cost_bps=cost_bps)
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    kept = [
        (d, r)
        for d, r in zip(result["dates"], result["returns"])
        if oos_start is None or d >= oos_start
    ]
    metrics = compute_metrics(
        [r for _, r in kept], [cash_by_date[d] for d, _ in kept]
    )
    years = len(kept) / 252
    metrics["turnover_oneway"] = (
        sum(
            w["turnover_oneway"]
            for w in result["windows"]
            if oos_start is None or w["test_end"] >= oos_start
        )
        / years
    )
    return {"result": result, "metrics": metrics, "oos_start": oos_start}


def race_contenders(
    panel: Panel,
    contenders: Mapping[str, Callable[[], Strategy]],
    first: int,
    oos_start: str | None = None,
    cost_bps: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Run every labeled contender through the same engine and span."""
    return {
        label: run_contender(panel, factory, first, oos_start, cost_bps)
        for label, factory in contenders.items()
    }


def metrics_table(
    metrics: Mapping[str, Mapping[str, float]],
    columns: Sequence[str] = METRIC_COLUMNS,
    label_width: int = 28,
    col_width: int = 14,
) -> str:
    """Fixed-width comparison table, one row per contender."""
    lines = [
        f"{'strategy':<{label_width}}" + "".join(f"{c:>{col_width}}" for c in columns)
    ]
    for label, m in metrics.items():
        lines.append(
            f"{label:<{label_width}}"
            + "".join(f"{m[c]:>{col_width}.4f}" for c in columns)
        )
    return "\n".join(lines)


def plain(value: object) -> object:
    """Recursively coerce numpy scalars so verdicts serialize as JSON."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()  # type: ignore[attr-defined]
    return value
