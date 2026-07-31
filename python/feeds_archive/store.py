"""Append-only JSONL storage plus derived SQLite index."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

# Feeds roll over within hours; two weeks of look-back is far more than any
# source's window, so older ids can never reappear as "new".
DEDUPE_LOOKBACK_DAYS = 14

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    kind       TEXT NOT NULL,
    source     TEXT NOT NULL,
    symbol     TEXT,
    title      TEXT,
    url        TEXT,
    published  TEXT,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_fetched ON items(fetched_at);
CREATE INDEX IF NOT EXISTS idx_items_source ON items(source, fetched_at);
CREATE TABLE IF NOT EXISTS heartbeats (
    ts        TEXT NOT NULL,
    new_items INTEGER NOT NULL,
    stats     TEXT NOT NULL
);
"""


def _utc_today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _day_files(jsonl_dir: Path, limit_days: int | None = None) -> list[Path]:
    files = sorted(jsonl_dir.glob("????-??-??.jsonl"))
    return files[-limit_days:] if limit_days else files


def load_seen_ids(jsonl_dir: Path) -> set[str]:
    seen: set[str] = set()
    for path in _day_files(jsonl_dir, DEDUPE_LOOKBACK_DAYS):
        with path.open() as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    continue
    return seen


def append_rows(jsonl_dir: Path, rows: list[dict]) -> Path:
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    path = jsonl_dir / f"{_utc_today()}.jsonl"
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return path


def append_heartbeat(jsonl_dir: Path, new_items: int, stats: dict) -> None:
    hb_dir = jsonl_dir / "heartbeats"
    hb_dir.mkdir(parents=True, exist_ok=True)
    beat = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "new_items": new_items,
        "stats": stats,
    }
    with (hb_dir / f"{_utc_today()}.jsonl").open("a") as fh:
        fh.write(json.dumps(beat, separators=(",", ":"), sort_keys=True) + "\n")


def build_db(jsonl_dir: Path, db_path: Path) -> dict:
    """Idempotently compile all JSONL day files (and heartbeats) into SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCHEMA)
        inserted = 0
        for path in _day_files(jsonl_dir):
            with path.open() as fh:
                for line in fh:
                    row = json.loads(line)
                    cur = con.execute(
                        "INSERT OR IGNORE INTO items VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            row["id"],
                            row["fetched_at"],
                            row["kind"],
                            row["source"],
                            row["symbol"],
                            row["title"],
                            row["url"],
                            row["published"],
                            json.dumps(row["payload"], sort_keys=True),
                        ),
                    )
                    inserted += cur.rowcount
        con.execute("DELETE FROM heartbeats")
        hb_dir = jsonl_dir / "heartbeats"
        if hb_dir.is_dir():
            for path in sorted(hb_dir.glob("????-??-??.jsonl")):
                with path.open() as fh:
                    for line in fh:
                        beat = json.loads(line)
                        con.execute(
                            "INSERT INTO heartbeats VALUES (?,?,?)",
                            (beat["ts"], beat["new_items"], json.dumps(beat["stats"])),
                        )
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        beats = con.execute("SELECT COUNT(*) FROM heartbeats").fetchone()[0]
        return {"inserted": inserted, "total_items": total, "heartbeats": beats}
    finally:
        con.close()
