"""Append-only decisions log: the paper-trading record.

Rows are point-in-time evidence — what each contender would have held,
written the day it was decided, keyed by `fetched_at` for when we could
have known. Unlike everything else under data/, this is NOT rebuildable:
recomputing a past decision later would use today's revised data and
today's code. Never mutate or regenerate rows; the file is git-tracked.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_LOG_PATH = Path("data/shadow/decisions.jsonl")
JOB_VERSION = "shadow-v1"


def logged_keys(path: Path) -> set[tuple[str, str]]:
    """(decision_date, strategy) pairs already on record."""
    keys: set[tuple[str, str]] = set()
    if not path.is_file():
        return keys
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                keys.add((row["decision_date"], row["strategy"]))
            except (ValueError, KeyError):
                continue
    return keys


def append_decision(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def load_decisions(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows
