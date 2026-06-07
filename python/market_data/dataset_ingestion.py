"""JSONL dataset ingestion — raw rows → memory-mapped f32 shards.

Input JSONL schema (one row per asset-minute):
  {
    "timestamp": "2024-06-30T10:00:00",
    "symbol":    "AAPL",
    "asset_class": "equity" | "etf" | "crypto",
    "price_volume":        float[10],
    "technical":           float[42],
    "options":             float[12] | null,   # null → zeroed
    "fundamental":         float[30] | null,
    "risk_analytics":      float[8]  | null,
    "event_state":         float[10] | null,
    "cross_asset_context": float[8]  | null,
    "text":                float[64] | null
  }

Output: binary shard files at data/mmaps/{split}/{symbol}.f32bin
  Format: raw little-endian f32, (PER_ASSET_FEATURE_DIM + MARKET_FEATURE_DIM)
  floats per row (184 + 96 = 280), no header.

Split assignment is delegated to train.splits.
Ingestion manifest written to reports/ingestion/{symbol}.json.
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

from train.feature_schema import (
    MARKET_FEATURE_DIM,
    PER_ASSET_BLOCKS,
    PER_ASSET_FEATURE_DIM,
    PER_ASSET_TEXT_DIM,
)
from train.splits import Split, assign_split

# ── Constants ─────────────────────────────────────────────────────────────────

ROW_DIM = PER_ASSET_FEATURE_DIM + MARKET_FEATURE_DIM  # 280 floats per row
ROW_STRUCT = struct.Struct(f"<{ROW_DIM}f")

# Block sizes derived from the canonical schema (feature_schema.py). The
# import-time assert guards against schema drift without needing a second
# copy of the numbers here.
_BLOCK_DIMS: dict[str, int] = {b.name: b.dim for b in PER_ASSET_BLOCKS} | {
    "text": PER_ASSET_TEXT_DIM,
}
assert sum(_BLOCK_DIMS.values()) == PER_ASSET_FEATURE_DIM, (
    f"Block dims sum to {sum(_BLOCK_DIMS.values())}, expected {PER_ASSET_FEATURE_DIM}"
)
_MARKET_DIM = MARKET_FEATURE_DIM


# ── Manifest ──────────────────────────────────────────────────────────────────

@dataclass
class SymbolStats:
    symbol: str
    rows_by_split: dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0, "test": 0})
    skipped_rows: int = 0
    first_timestamp: str = ""
    last_timestamp: str = ""


def _write_manifest(stats: SymbolStats, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{stats.symbol}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "symbol":          stats.symbol,
        "rows_by_split":   stats.rows_by_split,
        "skipped_rows":    stats.skipped_rows,
        "first_timestamp": stats.first_timestamp,
        "last_timestamp":  stats.last_timestamp,
        "total_rows":      sum(stats.rows_by_split.values()),
    }, indent=2))
    tmp.rename(path)


# ── Row packing ───────────────────────────────────────────────────────────────

def _parse_timestamp(raw: str) -> datetime:
    """Parse ISO-8601 timestamp to a timezone-aware datetime."""
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _coerce_block(data: list | None, expected: int) -> list[float]:
    """Return a float list of `expected` length, zeroing missing/null blocks."""
    if data is None:
        return [0.0] * expected
    if len(data) != expected:
        raise ValueError(f"expected {expected} values, got {len(data)}")
    return [float(v) for v in data]


def _pack_row(row: dict) -> tuple[str, datetime, bytes]:
    """Validate and pack one JSONL row to a packed struct bytes object.

    Returns (symbol, timestamp, packed_bytes).
    Raises ValueError for malformed rows.
    """
    symbol = row.get("symbol")
    if not symbol:
        raise ValueError("row missing 'symbol'")
    ts = _parse_timestamp(row["timestamp"])

    floats: list[float] = []
    for block, dim in _BLOCK_DIMS.items():
        floats.extend(_coerce_block(row.get(block), dim))
    floats.extend(_coerce_block(row.get("market"), _MARKET_DIM))

    return symbol, ts, ROW_STRUCT.pack(*floats)


# ── Shard writing ─────────────────────────────────────────────────────────────

class ShardWriter:
    """Atomic binary shard writer, one file per (symbol, split).

    Writes to {symbol}.f32bin.tmp and renames to {symbol}.f32bin on close.
    An interrupted run leaves only .tmp files, so --skip-existing correctly
    re-collects the symbol on resume.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._handles: dict[tuple[str, str], IO[bytes]] = {}

    def _handle(self, symbol: str, split: str) -> IO[bytes]:
        key = (symbol, split)
        if key not in self._handles:
            shard_dir = self._data_dir / split
            shard_dir.mkdir(parents=True, exist_ok=True)
            path = shard_dir / f"{symbol}.f32bin.tmp"
            self._handles[key] = open(path, "wb")  # noqa: SIM115
        return self._handles[key]

    def write(self, symbol: str, split: str, packed: bytes) -> None:
        self._handle(symbol, split).write(packed)

    def flush(self) -> None:
        for fh in self._handles.values():
            fh.flush()

    def close(self) -> None:
        for (symbol, split), fh in self._handles.items():
            fh.close()
            tmp_path = self._data_dir / split / f"{symbol}.f32bin.tmp"
            final_path = self._data_dir / split / f"{symbol}.f32bin"
            tmp_path.rename(final_path)
        self._handles.clear()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_jsonl(
    stream: Iterator[str],
    data_dir: Path,
    reports_dir: Path | None = None,
    log_every: int = 10_000,
) -> dict[str, SymbolStats]:
    """Ingest JSONL rows from `stream` into split-aware f32bin shards.

    Args:
        stream:      Line iterator (e.g. sys.stdin or open file).
        data_dir:    Root shard directory; shards written to {data_dir}/{split}/{symbol}.f32bin
        reports_dir: Where to write per-symbol manifests. Defaults to data_dir/../reports/ingestion.
        log_every:   Log progress every N rows processed.

    Returns:
        Dict mapping symbol → SymbolStats.
    """
    if reports_dir is None:
        reports_dir = data_dir.parent / "reports" / "ingestion"

    stats: dict[str, SymbolStats] = {}
    total_rows = 0

    with ShardWriter(data_dir) as writer:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                symbol, ts, packed = _pack_row(row)
                split: Split = assign_split(ts)

                if symbol not in stats:
                    stats[symbol] = SymbolStats(symbol=symbol)
                s = stats[symbol]
                s.rows_by_split[split] += 1
                if not s.first_timestamp:
                    s.first_timestamp = ts.isoformat()
                s.last_timestamp = ts.isoformat()

                writer.write(symbol, split, packed)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                sym = row.get("symbol", "?") if "row" in dir() else "?"
                if sym not in stats:
                    stats[sym] = SymbolStats(symbol=sym)
                stats[sym].skipped_rows += 1
                print(f"[ingest] skip row: {exc}", file=sys.stderr)
                continue

            total_rows += 1
            if log_every > 0 and total_rows % log_every == 0:
                print(f"[ingest] {total_rows} rows processed", file=sys.stderr)

        writer.flush()

    # Write manifests
    for s in stats.values():
        _write_manifest(s, reports_dir)
        print(
            f"[ingest] {s.symbol}: "
            f"train={s.rows_by_split['train']}  "
            f"val={s.rows_by_split['validation']}  "
            f"test={s.rows_by_split['test']}  "
            f"skip={s.skipped_rows}",
            file=sys.stderr,
        )

    return stats


def enumerate_shards(data_dir: Path, split: Split) -> list[Path]:
    """Return all f32bin shard paths under data_dir/split, sorted by symbol name.

    Used by SelfSupervisedPretrainer to build split-aware DataLoaders:

        train_paths = enumerate_shards(data_dir, "train")
        val_paths   = enumerate_shards(data_dir, "validation")
        train_loader = build_dataloader(train_paths, ...)
        val_loader   = build_dataloader(val_paths, ...)
    """
    shard_dir = data_dir / split
    if not shard_dir.exists():
        return []
    return sorted(shard_dir.glob("*.f32bin"))


def shard_row_count(path: Path) -> int:
    """Return the number of 280-dim feature rows in a shard file."""
    size = path.stat().st_size
    row_bytes = ROW_STRUCT.size
    if size % row_bytes != 0:
        raise ValueError(
            f"shard {path} size {size} is not a multiple of canonical row size {row_bytes}"
        )
    return size // row_bytes


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    """Read JSONL from stdin, write shards to data/mmaps."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest JSONL rows into f32bin shards")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mmaps"),
        help="Root shard directory (default: data/mmaps)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Manifest output dir (default: data/../reports/ingestion)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Log progress every N rows (default: 10000)",
    )
    args = parser.parse_args()
    ingest_jsonl(sys.stdin, args.data_dir, args.reports_dir, args.log_every)


if __name__ == "__main__":
    _cli()
