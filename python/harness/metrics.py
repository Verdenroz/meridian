"""Performance metrics over after-cost daily return series."""

from __future__ import annotations

import math
import statistics

from harness.panel import TRADING_DAYS_PER_YEAR


def max_drawdown(returns: list[float]) -> float:
    """Largest peak-to-trough loss, as a positive fraction."""
    peak = 1.0
    value = 1.0
    worst = 0.0
    for r in returns:
        value *= 1.0 + r
        peak = max(peak, value)
        worst = max(worst, 1.0 - value / peak)
    return worst


def compute_metrics(
    returns: list[float], cash_returns: list[float] | None = None
) -> dict:
    n = len(returns)
    if n < 2:
        raise ValueError("need at least 2 daily returns")
    growth = 1.0
    for r in returns:
        growth *= 1.0 + r
    years = n / TRADING_DAYS_PER_YEAR
    ann_vol = statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
    excess = (
        [r - c for r, c in zip(returns, cash_returns)]
        if cash_returns is not None
        else returns
    )
    excess_std = statistics.stdev(excess)
    sharpe = (
        statistics.mean(excess) / excess_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if excess_std > 0
        else 0.0
    )
    return {
        "days": n,
        "total_return": growth - 1.0,
        "cagr": growth ** (1.0 / years) - 1.0,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "worst_day": min(returns),
        "best_day": max(returns),
    }
