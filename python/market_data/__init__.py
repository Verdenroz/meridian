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
# y10 before 2001, so the spread comes straight from FRED). BAA10Y: Moody's
# Baa-10Y credit spread — ICE BofA OAS series are truncated to ~3y upstream.
# AAA10Y: Moody's Aaa-10Y, the low leg of the quality spread (quality_race).
# DGS10: 10Y constant-maturity yield (1962+) — rate-vol channel (rate_race).
# DCOILWTICO: WTI spot (1986+) — oil-vol channel (oilvol_race). DEXJPUS (1971+)
# and DEXUSEU (1999+): USD/JPY and USD/EUR — FX-vol channel (fxvol_race).
# ICNSA: weekly initial claims, NSA (1967+) — real-economy channel
# (claims_race); NSA so retrospective seasonal-factor rewrites never leak in.
DEFAULT_FRED_SERIES = (
    "VIXCLS",
    "T10Y3M",
    "BAA10Y",
    "AAA10Y",
    "DGS10",
    "DCOILWTICO",
    "DEXJPUS",
    "DEXUSEU",
    "ICNSA",
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_FRED_SERIES",
    "DEFAULT_STORE_DIR",
    "DEFAULT_SYMBOLS",
    "CrossCheckError",
    "verify_adj_close",
]
