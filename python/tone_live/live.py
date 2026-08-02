"""Live headline-tone enrichment: the tone channel's continuation onto the
feeds archive, frozen before the first score was written.

The tone-feature race's held candidate (8f-turn-tone 3-state) reads a
feature that is zero after FNSPID's 2024-01 end. This module gives the
channel a live continuation so the candidate can accrue true out-of-sample
evidence: every archived headline from the frozen source whitelist is
FinBERT-scored once and appended to a git-tracked, append-only store.

Frozen conventions (changing any after scores exist invalidates the
continuation, exactly like editing a race's GATE):

- Scorer: ProsusAI/finbert, the sentiment cache's standing contract —
  signed P(pos) − P(neg), text keyed as sha256(model\\x00text). Headline
  text is title.strip()[:2000], the tone study's preprocessing.
- Source whitelist: financial/market news outlets only — the FNSPID-like
  corpus. General-news outlets, official/regulator feeds (SEC, central
  banks, statistical agencies) and commentary blogs are excluded a priori:
  the tone signal was measured on market-news headlines, and officials/
  blogs are different text populations.
- Point-in-time: rows are keyed by the archive item's `fetched_at` (when
  we could have known); `date` is its UTC calendar date. Downstream
  session alignment inherits the tone race's convention (a UTC date's
  headlines belong to the first session on or after it).
- Append-only: one row per archive item id, ever; re-runs skip scored
  ids. Never rescore, never edit rows — a model upgrade is a *new* store
  and a new registration, not an overwrite.

The store is data/tone_live/scores.jsonl (git-tracked). Wiring the live
series into any race or the shadow roster is future work through the
normal pre-registration discipline; this module only keeps the record.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from filing_tone.score import MAX_CHARS, read_cached_tone, score_missing

__all__ = ["SOURCE_WHITELIST", "STORE_PATH", "daily_tone", "score_new", "status"]

ARCHIVE_DIR = Path("archive/feeds")
STORE_PATH = Path("data/tone_live/scores.jsonl")
SENTIMENT_MODEL = "ProsusAI/finbert"

# Frozen a priori. Criterion: financial/market news outlets (the corpus the
# tone signal was measured on). Spelling variants both appear upstream.
SOURCE_WHITELIST = frozenset(
    {
        "Barrons",
        "Bloomberg",
        "Business Wire",
        "CNBC",
        "ETF Trends",
        "FXEmpire",
        "Financial Post",
        "Financial Times",
        "Financial Times Lex",
        "Forbes",
        "Fortune",
        "Fox Business",
        "Investing.com",
        "Investopedia",
        "Invezz",
        "Market Watch",
        "MarketWatch",
        "Nikkei Asia",
        "OilPrice.com",
        "Reuters",
        "TipRanks",
        "WSJ",
        "Wall Street Journal",
    }
)


def _iter_archive_items(archive_dir: Path) -> Iterator[dict]:
    for path in sorted(archive_dir.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _scored_ids(store_path: Path) -> set[str]:
    if not store_path.is_file():
        return set()
    with store_path.open() as fh:
        return {json.loads(line)["id"] for line in fh if line.strip()}


def _eligible(row: dict) -> bool:
    title = row.get("title")
    return bool(title and str(title).strip()) and row.get("source") in SOURCE_WHITELIST


def score_new(
    archive_dir: Path = ARCHIVE_DIR,
    store_path: Path = STORE_PATH,
    limit: int | None = None,
) -> int:
    """Score whitelisted archive items not yet in the store; append rows in
    (fetched_at, id) order. Returns rows written."""
    seen = _scored_ids(store_path)
    todo = [
        row
        for row in _iter_archive_items(archive_dir)
        if _eligible(row) and row["id"] not in seen
    ]
    todo.sort(key=lambda r: (r.get("fetched_at") or "", r["id"]))
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return 0

    texts = [str(row["title"]).strip()[:MAX_CHARS] for row in todo]
    score_missing(texts)

    store_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with store_path.open("a") as fh:
        for row, text in zip(todo, texts):
            tone = read_cached_tone(text)
            if tone is None:
                continue
            fetched = str(row.get("fetched_at") or "")
            record = {
                "id": row["id"],
                "fetched_at": fetched,
                "date": fetched[:10],
                "source": row["source"],
                "tone": round(float(tone), 6),
                "model": SENTIMENT_MODEL,
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
    return written


def daily_tone(store_path: Path = STORE_PATH) -> dict[str, tuple[int, float]]:
    """UTC fetched date -> (headline count, mean signed tone); the live
    counterpart of the tone race's market_daily_tone aggregate."""
    totals: dict[str, list[float]] = {}
    if not store_path.is_file():
        return {}
    with store_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            box = totals.setdefault(row["date"], [0.0, 0.0])
            box[0] += 1
            box[1] += float(row["tone"])
    return {d: (int(n), s / n) for d, (n, s) in totals.items() if n > 0}


def status(store_path: Path = STORE_PATH) -> dict:
    days = daily_tone(store_path)
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    recent = sorted(days.items())[-5:]
    return {
        "rows": sum(n for n, _ in days.values()),
        "days": len(days),
        "span": (min(days), max(days)) if days else None,
        "today_utc": now,
        "recent": [
            {"date": d, "headlines": n, "mean_tone": round(m, 4)}
            for d, (n, m) in recent
        ],
    }
