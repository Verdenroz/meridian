"""Event sources: EDGAR filings (via finance-query) and the FNSPID news cache.

Availability clocks: a filing is known at the close of its filingDate; a
headline at the close of its timestamp's UTC date. The FNSPID cache under
data/.fnspid_cache/ is read-only historical input — never written here.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from market_data.client import get_json
from market_data.store import read_snapshot, write_snapshot

FNSPID_CACHE = Path("data/.fnspid_cache")
STUDY_FORMS = ("8-K", "10-K", "10-Q")


def fetch_filings(base_url: str, store_dir: Path, symbol: str) -> list[dict]:
    """Fetch and snapshot EDGAR submissions; return study-relevant filings."""
    data = get_json(base_url, f"/v2/edgar/submissions/{symbol}")
    filings = [
        {"date": f["filingDate"], "form": f["form"]}
        for f in data.get("filings", [])
        if f.get("form") in STUDY_FORMS and f.get("filingDate")
    ]
    filings.sort(key=lambda f: f["date"])
    write_snapshot(store_dir, "edgar", symbol, {"filings": filings})
    return filings


def load_filings(store_dir: Path, symbol: str) -> list[dict] | None:
    doc = read_snapshot(store_dir, "edgar", symbol)
    return None if doc is None else doc["data"]["filings"]


def news_daily_counts(symbol: str, cache_dir: Path = FNSPID_CACHE) -> dict[str, int]:
    """UTC-date → headline count for one symbol; {} if not covered."""
    path = cache_dir / f"{symbol}.jsonl"
    if not path.is_file():
        return {}
    counts: Counter[str] = Counter()
    with path.open() as fh:
        for line in fh:
            try:
                ts = json.loads(line)["ts"]
            except (ValueError, KeyError):
                continue
            counts[datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")] += 1
    return dict(counts)


def market_daily_counts(cache_dir: Path = FNSPID_CACHE) -> dict[str, int]:
    """Aggregate headline count per UTC date across every cached symbol."""
    counts: Counter[str] = Counter()
    for path in sorted(cache_dir.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                try:
                    ts = json.loads(line)["ts"]
                except (ValueError, KeyError):
                    continue
                counts[datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")] += 1
    return dict(counts)


def intensity_zscores(
    dates: list[str], counts: dict[str, int], window: int = 252, min_obs: int = 120
) -> np.ndarray:
    """Causal z-score of the news count aligned to trading dates: z[t] compares
    day t's count to the trailing `window` trading days *before* t."""
    raw = np.array([float(counts.get(d, 0)) for d in dates])
    out = np.full(len(dates), np.nan)
    for t in range(len(dates)):
        lo = max(0, t - window)
        past = raw[lo:t]
        if len(past) < min_obs:
            continue
        std = past.std()
        if std > 0:
            out[t] = (raw[t] - past.mean()) / std
    return out
