"""Daily backtest engine with drift, costs, and model-external guardrails.

Timeline per day t: earn day-t returns on yesterday's weights, let weights
drift, then (after the close) ask the strategy for targets using a HistoryView
capped at t; the trade executes at day-t close and its cost nets out of day
t's recorded return, so the return series is fully after-cost.
"""

from __future__ import annotations

from typing import Protocol

from harness.costs import one_way_cost
from harness.panel import Panel
from harness.view import HistoryView

# Long-only, unlevered (charter: no margin, no short selling).
MAX_GROSS_EXPOSURE = 1.0 + 1e-9


class GuardrailViolation(Exception):
    """Strategy asked for weights the charter forbids. Never softened."""


class Strategy(Protocol):
    def fit(self, view: HistoryView) -> None: ...

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        """Targets as fractions of portfolio value; None keeps current
        weights. Unlisted symbols mean zero; remainder is cash."""
        ...


def _check_guardrails(targets: dict[str, float]) -> None:
    gross = 0.0
    for symbol, weight in targets.items():
        if weight < -1e-12:
            raise GuardrailViolation(f"negative weight {weight} for {symbol}")
        gross += weight
    if gross > MAX_GROSS_EXPOSURE:
        raise GuardrailViolation(f"gross exposure {gross:.4f} exceeds 1.0")


def _rebalance(
    weights: dict[str, float],
    targets: dict[str, float],
    cost_bps: dict[str, float] | None,
) -> tuple[dict[str, float], float, float]:
    """Returns (new weights, one-way turnover, cost as fraction of value)."""
    _check_guardrails(targets)
    traded = 0.0
    cost_frac = 0.0
    for symbol in set(weights) | set(targets):
        delta = abs(targets.get(symbol, 0.0) - weights.get(symbol, 0.0))
        traded += delta
        cost_frac += delta * one_way_cost(symbol, cost_bps)
    return dict(targets), traded, cost_frac


def run_backtest(
    panel: Panel,
    strategy: Strategy,
    start: int,
    end: int | None = None,
    cost_bps: dict[str, float] | None = None,
) -> dict:
    """Simulate from index `start` (first decision day) through `end`
    inclusive. The returned daily returns are after costs."""
    end = panel.n_days - 1 if end is None else end
    weights: dict[str, float] = {}
    values: list[float] = []
    returns: list[float] = []
    dates: list[str] = []
    value = 1.0
    total_turnover = 0.0
    total_cost = 0.0
    # Entry-day trade cost has no return day of its own; it nets out of the
    # first recorded day instead.
    carry_factor = 1.0

    strategy.fit(HistoryView(panel, start))

    for t in range(start, end + 1):
        gross_growth = None
        if t > start:
            day_return = 0.0
            drifted: dict[str, float] = {}
            for symbol, weight in weights.items():
                r = panel.returns[symbol][t]
                if r is None:
                    raise ValueError(f"{symbol} has no return on {panel.dates[t]}")
                day_return += weight * r
                drifted[symbol] = weight * (1.0 + r)
            day_return += (1.0 - sum(weights.values())) * panel.cash_returns[t]
            gross_growth = 1.0 + day_return
            weights = {s: w / gross_growth for s, w in drifted.items()}

        targets = strategy.target_weights(HistoryView(panel, t))
        cost_frac = 0.0
        if targets is not None:
            targets = {s: w for s, w in targets.items() if w != 0.0}
            weights, traded, cost_frac = _rebalance(weights, targets, cost_bps)
            total_turnover += traded
            total_cost += cost_frac * value * (gross_growth or 1.0)

        if gross_growth is None:
            carry_factor *= 1.0 - cost_frac
        else:
            net_growth = gross_growth * (1.0 - cost_frac) * carry_factor
            carry_factor = 1.0
            value *= net_growth
            returns.append(net_growth - 1.0)
            values.append(value)
            dates.append(panel.dates[t])

    return {
        "dates": dates,
        "returns": returns,
        "values": values,
        "final_value": value,
        "turnover_oneway": total_turnover,
        "total_cost": total_cost,
        "final_weights": weights,
    }
