"""Frozen stock-sleeve universe.

Twenty-five liquid US large caps chosen for depth of clean daily history in
the point-in-time store, fixed before any sleeve result was computed. Names
enter a backtest only once they carry a full year of returns, so inception
is honored causally; nothing here is rebalanced by hindsight after that.

Known limitation, reported with every sleeve backtest: the list is today's
survivors. Yahoo-sourced history has no delisted names, so absolute equity
returns are inflated; the sleeve consumes vol and correlation, where the
bias is limited. Additions require a dated comment; removals never.
"""

from __future__ import annotations

MARKET_SYMBOL = "SPY"

SLEEVE_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "CAT",
    "CSCO",
    "CVX",
    "DIS",
    "GOOGL",
    "HD",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "META",
    "MRK",
    "MSFT",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "UNH",
    "WMT",
    "XOM",
)
