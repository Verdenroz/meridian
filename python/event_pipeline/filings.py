"""8-K filing metadata (via finance-query) and document text (via SEC EDGAR).

finance-query's own EDGAR adapter wraps the free public SEC API; its HTTP
surface does not serve raw filing documents, so document text is fetched from
the same public archive directly (data.sec.gov requires only a User-Agent
with contact info). Nothing here modifies finance-query.
"""

from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from market_data.client import get_json
from market_data.store import read_snapshot, write_snapshot

SEC_USER_AGENT = "meridian-research harveytseng2@gmail.com"
HTTP_TIMEOUT = 60
# 8-K bodies are short; inline-XBRL wrappers are not. Plenty for an LLM read.
MAX_DOC_CHARS = 16000


def refresh_8k_metadata(base_url: str, store_dir: Path, symbol: str) -> dict:
    data = get_json(base_url, f"/v2/edgar/submissions/{symbol}")
    cik = str(int(data["cik"]))
    filings = [
        {
            "accession": f["accessionNumber"],
            "date": f["filingDate"],
            "document": f.get("primaryDocument") or "",
        }
        for f in data.get("filings", [])
        if f.get("form") == "8-K" and f.get("filingDate") and f.get("accessionNumber")
    ]
    filings.sort(key=lambda f: f["date"], reverse=True)
    doc = {"cik": cik, "filings": filings}
    write_snapshot(store_dir, "edgar_8k", symbol, doc)
    return doc


def load_8k_metadata(store_dir: Path, symbol: str) -> dict | None:
    doc = read_snapshot(store_dir, "edgar_8k", symbol)
    return None if doc is None else doc["data"]


def document_url(cik: str, accession: str, document: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession.replace('-', '')}/{document}"
    )


class _TextExtractor(HTMLParser):
    _SKIP: ClassVar[set[str]] = {"script", "style", "head", "title"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def fetch_document_text(
    cik: str, accession: str, document: str, max_chars: int = MAX_DOC_CHARS
) -> str:
    url = document_url(cik, accession, document)
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return html_to_text(html)[:max_chars]
