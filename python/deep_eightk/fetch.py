"""Full per-CIK 8-K filing-date history from SEC's submissions API.

The finance-query submissions endpoint caps at ~1,000 recent filings —
that cap is exactly why the original occurrence race was stuck with a
9-name panel and 2022+ OOS. SEC's own submissions JSON serves the full
history in pages; we go to data.sec.gov directly for it, the same
precedent event_pipeline set for document text (a surface finance-query
does not expose). CIKs are resolved through finance-query.

Snapshots land in the point-in-time store (kind "edgar_deep"), stamped
`fetched_at` like every other snapshot.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from market_data.client import get_json
from market_data.store import read_snapshot, write_snapshot

SEC_USER_AGENT = "Meridian research harveytseng2@gmail.com"
HTTP_TIMEOUT = 60
# SEC fair-access ceiling is 10 req/s; stay far below it.
POLITE_DELAY_S = 0.2

# Corporate-history CIKs the ticker's current CIK does not reach: Alphabet
# (2015+) does not carry Google Inc's filings, and upstream maps XOM to a
# 2026-vintage entity whose history lives under Exxon Mobil's original CIK.
# Availability-driven, fixed before any race result was computed.
PREDECESSOR_CIKS: dict[str, tuple[str, ...]] = {
    "GOOGL": ("0001288776",),  # Google Inc
    "XOM": ("0000034088",),    # Exxon Mobil Corp
}


def _sec_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp)


def _extract_8k_dates(page: dict) -> list[str]:
    forms = page.get("form", [])
    dates = page.get("filingDate", [])
    return [d for f, d in zip(forms, dates) if f == "8-K" and d]


def _cik_8k_dates(cik: str) -> list[str]:
    root = _sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    dates = _extract_8k_dates(root.get("filings", {}).get("recent", {}))
    for older in root.get("filings", {}).get("files", []):
        time.sleep(POLITE_DELAY_S)
        page = _sec_json(f"https://data.sec.gov/submissions/{older['name']}")
        dates.extend(_extract_8k_dates(page))
    return dates


def fetch_full_8k_history(base_url: str, store_dir: Path, symbol: str) -> dict:
    """Fetch every 8-K filing date for `symbol` (current CIK plus any
    predecessor entities) and snapshot the union."""
    cik = get_json(base_url, f"/v2/edgar/submissions/{symbol}")["cik"]
    ciks = (cik, *PREDECESSOR_CIKS.get(symbol, ()))
    dates: list[str] = []
    for c in ciks:
        dates.extend(_cik_8k_dates(c))
        time.sleep(POLITE_DELAY_S)

    dates = sorted(set(dates))
    write_snapshot(store_dir, "edgar_deep", symbol, {"ciks": list(ciks), "dates": dates})
    return {"symbol": symbol, "cik": "+".join(ciks), "n_8k": len(dates),
            "span": (dates[0], dates[-1]) if dates else None}


def load_8k_dates(store_dir: Path, symbol: str) -> list[str] | None:
    doc = read_snapshot(store_dir, "edgar_deep", symbol)
    return None if doc is None else doc["data"]["dates"]
