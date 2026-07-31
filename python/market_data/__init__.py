"""Historical market data layer: loaders, adjClose cross-check, point-in-time store.

Consumes the finance-query v2 API (read-only). Upstream `adjClose` is the
canonical total-return input; every load is cross-checked against a local
dividends-based reconstruction and fails loudly on divergence, because silent
adjustment errors would poison every downstream estimate.
"""

from market_data.client import DEFAULT_BASE_URL
from market_data.crosscheck import CrossCheckError, verify_adj_close
from market_data.store import DEFAULT_STORE_DIR

# ETF core universe (spec: equities, treasuries, gold, T-bills, inverse hedge).
# The stock sleeve is added only after the ETF core beats the null baseline.
DEFAULT_SYMBOLS = ("SPY", "TLT", "GLD", "BIL", "SH")

# VIXCLS: implied vol; T10Y3M: 10Y-3M curve spread (treasury endpoint lacks
# y10 before 2001, so the spread comes straight from FRED).
DEFAULT_FRED_SERIES = ("VIXCLS", "T10Y3M")

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_FRED_SERIES",
    "DEFAULT_STORE_DIR",
    "DEFAULT_SYMBOLS",
    "CrossCheckError",
    "verify_adj_close",
]
