"""Append-only archive of intraday bars for point-in-time intraday research.

Upstream keeps 1m bars for 29 days and 5m bars for 58, then they are gone:
this archive is the only intraday history Meridian will ever have. One
gzipped JSONL file per (interval, session, symbol) under git-tracked
``archive/bars/``, written once after the session closes and never
rewritten. Bars are regular-session only (upstream omits pre/post), and
minutes with no trade are absent, so thin ETFs carry fewer than 390 bars.

The universe is frozen: add a symbol under a new constant, never edit one.
"""

DEFAULT_BASE_URL = "https://finance-query.com"

UNIVERSE: dict[str, tuple[str, ...]] = {
    "1m": ("SPY", "QQQ", "IWM", "DIA", "SH", "PSQ", "SPYM"),
    "5m": (
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
        "TLT", "GLD",
    ),
}
