"""Write-once gzipped JSONL session files."""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path


def session_path(root: Path, interval: str, session: date, symbol: str) -> Path:
    return root / interval / session.isoformat() / f"{symbol}.jsonl.gz"


def write_session(path: Path, rows: list[dict]) -> bool:
    """Exclusive create: an archived session is never rewritten. False when it
    already exists."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in rows)
    with path.open("xb") as fh, gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(body.encode())
    return True


def read_session(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def coverage(root: Path, interval: str) -> dict[str, set[str]]:
    base = root / interval
    if not base.is_dir():
        return {}
    return {
        d.name: {p.name.removesuffix(".jsonl.gz") for p in d.glob("*.jsonl.gz")}
        for d in sorted(base.iterdir())
        if d.is_dir()
    }
