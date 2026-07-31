"""Point-in-time snapshot store under data/market/ (derived, rebuildable).

Every snapshot records `fetched_at` — when Meridian could have known the data.
Files are written atomically so a crash never leaves a torn snapshot.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_STORE_DIR = Path("data/market")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_snapshot(store_dir: Path, kind: str, name: str, data: dict) -> Path:
    path = store_dir / kind / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"kind": kind, "name": name, "fetched_at": _now_iso(), "data": data}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def read_snapshot(store_dir: Path, kind: str, name: str) -> dict | None:
    path = store_dir / kind / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def list_snapshots(store_dir: Path) -> list[dict]:
    out: list[dict] = []
    if not store_dir.is_dir():
        return out
    for path in sorted(store_dir.glob("*/*.json")):
        try:
            doc = json.loads(path.read_text())
            out.append(
                {
                    "kind": doc["kind"],
                    "name": doc["name"],
                    "fetched_at": doc["fetched_at"],
                    "path": str(path),
                }
            )
        except (ValueError, KeyError):
            out.append({"kind": path.parent.name, "name": path.stem, "error": "unreadable"})
    return out
