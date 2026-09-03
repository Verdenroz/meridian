"""Sleeve panel on the market calendar.

Names are aligned to the market symbol's trading days. An isolated missing
candle inside a name's history (an upstream hole, not a holiday) carries the
previous adjusted close forward so the engine sees a zero return that day
and the true multi-day return the next; the count is reported with every
race as a known limitation. Gaps longer than MAX_GAP_DAYS are left as gaps.
"""

from __future__ import annotations

from pathlib import Path

from harness.panel import TRADING_DAYS_PER_YEAR, Panel, build_panel

MAX_GAP_DAYS = 3


def _fill_isolated_gaps(prices: list[float | None]) -> tuple[list[float | None], int]:
    out = list(prices)
    filled = 0
    last: float | None = None
    run_start: int | None = None
    for i, price in enumerate(prices):
        if price is not None:
            if run_start is not None and last is not None and i - run_start <= MAX_GAP_DAYS:
                for k in range(run_start, i):
                    out[k] = last
                    filled += 1
            run_start = None
            last = price
        elif last is not None and run_start is None:
            run_start = i
    return out, filled


def build_sleeve_panel(
    store_dir: Path, market_symbol: str, universe: tuple[str, ...]
) -> tuple[Panel, dict[str, int]]:
    """Panel on the market calendar with isolated holes filled; returns the
    panel and the per-name count of filled candles."""
    raw = build_panel(store_dir, (market_symbol, *universe))
    keep = [i for i, p in enumerate(raw.adj_close[market_symbol]) if p is not None]
    dates = [raw.dates[i] for i in keep]
    cash = [raw.cash_returns[i] * TRADING_DAYS_PER_YEAR * 100.0 for i in keep]
    adj_close: dict[str, list[float | None]] = {}
    filled: dict[str, int] = {}
    for symbol, prices in raw.adj_close.items():
        aligned = [prices[i] for i in keep]
        if symbol == market_symbol:
            adj_close[symbol] = aligned
            continue
        adj_close[symbol], filled[symbol] = _fill_isolated_gaps(aligned)
    return Panel(dates, adj_close, cash), {s: n for s, n in filled.items() if n}
