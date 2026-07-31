"""Append-only score store: JSONL keyed by (accession, scorer_version).

Lives beside — never inside — the raw data (spec §3.4). Deleting the file
loses paid LLM inference, so rows are only ever appended; re-scoring under a
new prompt/model writes new rows under the new version.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_SCORES_PATH = Path("data/scores/eightk_scores.jsonl")


def load_scored_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.is_file():
        return keys
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                keys.add((row["accession"], row["scorer_version"]))
            except (ValueError, KeyError):
                continue
    return keys


def append_score(
    path: Path,
    symbol: str,
    accession: str,
    filing_date: str,
    scorer_version: str,
    score: dict,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "accession": accession,
        "symbol": symbol,
        "filing_date": filing_date,
        "scorer_version": scorer_version,
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **score,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return row


def load_scores(path: Path, scorer_version: str | None = None) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if scorer_version is None or row.get("scorer_version") == scorer_version:
                rows.append(row)
    return rows
