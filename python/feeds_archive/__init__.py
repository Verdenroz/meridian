"""Append-only archive of finance-query news/feeds for point-in-time research.

JSONL day files (git-tracked) are the source of truth; SQLite is a derived
index. Rows are never mutated: headline revisions become new rows.
"""

DEFAULT_BASE_URL = "https://finance-query.com"

# Finance/macro-relevant slugs from the /v2/feeds source enum; crypto and
# startup-tech sources are deliberately excluded from the default sweep.
DEFAULT_SOURCES = (
    "federal-reserve",
    "sec",
    "sec-filings",
    "marketwatch",
    "cnbc",
    "bloomberg",
    "ft",
    "nyt",
    "guardian",
    "investing",
    "bea",
    "ecb",
    "cfpb",
    "wsj",
    "fortune",
    "businesswire",
    "oilprice",
    "calculated-risk",
    "scmp",
    "nikkei",
    "boe",
    "economist",
    "financial-post",
    "ft-lex",
    "ritholtz",
)
