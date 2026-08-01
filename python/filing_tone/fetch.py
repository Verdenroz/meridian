"""Per-symbol filing index (accession, form, date) from SEC submissions.

The deep-race fetch kept only 8-K *dates*; the filing-tone study needs
accessions to join filings to the legacy full-text store
(`data/edgar_text/`, keyed by dash-stripped accession). Same source,
same CIK resolution (predecessor entities included), snapshotted under
kind "edgar_index".
"""

from __future__ import annotations

import time
from pathlib import Path

from deep_eightk.fetch import POLITE_DELAY_S, PREDECESSOR_CIKS, _sec_json
from market_data.client import get_json
from market_data.store import read_snapshot, write_snapshot

EDGAR_TEXT_DIR = Path("data/edgar_text")


def _extract(page: dict) -> list[dict]:
    filings = []
    accession = page.get("accessionNumber", [])
    forms = page.get("form", [])
    dates = page.get("filingDate", [])
    for a, f, d in zip(accession, forms, dates):
        if a and f and d:
            filings.append({"accession": a, "form": f, "date": d})
    return filings


def _cik_filings(cik: str) -> list[dict]:
    root = _sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    filings = _extract(root.get("filings", {}).get("recent", {}))
    for older in root.get("filings", {}).get("files", []):
        time.sleep(POLITE_DELAY_S)
        page = _sec_json(f"https://data.sec.gov/submissions/{older['name']}")
        filings.extend(_extract(page))
    return filings


def fetch_filing_index(base_url: str, store_dir: Path, symbol: str) -> dict:
    cik = get_json(base_url, f"/v2/edgar/submissions/{symbol}")["cik"]
    ciks = (cik, *PREDECESSOR_CIKS.get(symbol, ()))
    filings: list[dict] = []
    for c in ciks:
        filings.extend(_cik_filings(c))
        time.sleep(POLITE_DELAY_S)
    seen = set()
    unique = []
    for row in sorted(filings, key=lambda r: (r["date"], r["accession"])):
        if row["accession"] not in seen:
            seen.add(row["accession"])
            unique.append(row)
    write_snapshot(
        store_dir, "edgar_index", symbol, {"ciks": list(ciks), "filings": unique}
    )
    return {"symbol": symbol, "n_filings": len(unique)}


def load_filing_index(store_dir: Path, symbol: str) -> list[dict] | None:
    doc = read_snapshot(store_dir, "edgar_index", symbol)
    return None if doc is None else doc["data"]["filings"]


def text_path(accession: str) -> Path:
    return EDGAR_TEXT_DIR / f"{accession.replace('-', '')}.txt"
