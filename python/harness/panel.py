"""Backtest panel: aligned total-return data plus a daily cash rate.

Cash earns the 3-month T-bill yield (treasury curve, forward-filled) — using a
rate series rather than BIL lets backtests before BIL's 2007 inception stay
honest about the opportunity cost of de-risking.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from market_data.macro import TREASURY_FIRST_YEAR
from market_data.panel import load_total_return_panel
from market_data.store import read_snapshot

TRADING_DAYS_PER_YEAR = 252
# Holiday gaps in the curve are a few days; anything longer means the store is
# missing treasury history, which would otherwise price cash at zero.
MAX_CASH_GAP_DAYS = 10


class Panel:
    """Immutable daily panel; returns[symbol][i] is the total return from
    dates[i-1] to dates[i] (None on day 0 or across missing data)."""

    def __init__(
        self,
        dates: list[str],
        adj_close: dict[str, list[float | None]],
        cash_rate_annual_pct: list[float],
    ) -> None:
        self.dates = list(dates)
        self.adj_close = {s: list(v) for s, v in adj_close.items()}
        self.cash_returns = [
            r / 100.0 / TRADING_DAYS_PER_YEAR for r in cash_rate_annual_pct
        ]
        self.returns: dict[str, list[float | None]] = {}
        for symbol, prices in self.adj_close.items():
            rets: list[float | None] = [None]
            for i in range(1, len(prices)):
                prev, cur = prices[i - 1], prices[i]
                rets.append(
                    cur / prev - 1.0 if prev is not None and cur is not None else None
                )
            self.returns[symbol] = rets

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def first_index_with_data(self, symbols: tuple[str, ...]) -> int:
        """First index at which every symbol has a defined daily return."""
        for i in range(self.n_days):
            if all(self.returns[s][i] is not None for s in symbols):
                return i
        raise ValueError(f"no common history for {symbols}")


def _cash_rate_by_date(store_dir: Path, dates: list[str]) -> list[float]:
    doc = read_snapshot(store_dir, "macro", "treasury_yields")
    by_date: dict[str, float] = {}
    if doc is not None:
        for row in doc["data"]["rows"]:
            rate = row.get("y3M") or row.get("y1M") or row.get("y1")
            if rate is not None:
                by_date[row["date"]] = float(rate)
    out: list[float] = []
    last, seen = 0.0, None
    for day in dates:
        if day in by_date:
            last, seen = by_date[day], day
        if day < str(TREASURY_FIRST_YEAR):
            out.append(0.0)
            continue
        if seen is None or (date.fromisoformat(day) - date.fromisoformat(seen)).days > MAX_CASH_GAP_DAYS:
            raise ValueError(
                f"no T-bill rate within {MAX_CASH_GAP_DAYS} days of {day}; "
                "the store is missing treasury history"
            )
        out.append(last)
    return out


def build_panel(store_dir: Path, symbols: tuple[str, ...]) -> Panel:
    raw = load_total_return_panel(store_dir, symbols)
    cash = _cash_rate_by_date(store_dir, raw["dates"])
    return Panel(raw["dates"], raw["adj_close"], cash)
