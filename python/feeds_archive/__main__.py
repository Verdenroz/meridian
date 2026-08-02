"""CLI: python -m feeds_archive {fetch|loop|build-db|status}"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from . import DEFAULT_BASE_URL, DEFAULT_SOURCES
from .fetch import fetch_all, fetch_symbols
from .store import append_heartbeat, append_rows, build_db, load_seen_ids

DEFAULT_JSONL_DIR = Path("archive/feeds")
DEFAULT_DB = Path("data/feeds_archive.db")
DEFAULT_PERNAME_DIR = Path("archive/pername")
DEFAULT_PERNAME_UNIVERSE = Path("deploy/pername-universe.txt")
# Per-symbol feeds return the ~25 newest items regardless of age, so sleepy
# names span weeks; the dedupe window must be far longer than the market
# sweep's hours-scale rollover.
PERNAME_DEDUPE_LOOKBACK_DAYS = 60
# Upstream admits ~150 requests per multi-minute window, then 404s until a
# ~90s cooldown (request spacing alone does not help): sweep in chunks under
# the budget with a cooldown between them, then re-sweep any leftovers so
# late-alphabet names are never systematically under-covered.
PERNAME_REQUEST_PAUSE = 1.0
PERNAME_CHUNK_SIZE = 120
# Recovery is ~90s in isolation but marginal right after a full chunk;
# 180s keeps retry rounds out of the still-hot window.
PERNAME_COOLDOWN = 180
PERNAME_RETRY_ROUNDS = 3


def _read_symbols(path: str | None) -> tuple[str, ...]:
    if not path:
        return ()
    lines = Path(path).read_text().splitlines()
    return tuple(s.strip().upper() for s in lines if s.strip() and not s.startswith("#"))


def _archive_new(jsonl_dir: Path, seen: set[str], rows: list[dict], stats: dict) -> int:
    fresh_by_id = {r["id"]: r for r in rows if r["id"] not in seen}
    fresh = sorted(fresh_by_id.values(), key=lambda r: r["id"])
    if fresh:
        append_rows(jsonl_dir, fresh)
    append_heartbeat(jsonl_dir, len(fresh), stats)
    print(
        f"fetched={sum(stats['fetched'].values())} new={len(fresh)} "
        f"errors={len(stats['errors'])}"
    )
    if stats["errors"]:
        print(json.dumps(stats["errors"], indent=2), file=sys.stderr)
    # Errors are non-fatal by design; a partial sweep is still worth committing.
    return 0


def run_fetch(args: argparse.Namespace) -> int:
    jsonl_dir = Path(args.jsonl_dir)
    seen = load_seen_ids(jsonl_dir)
    rows, stats = fetch_all(
        args.base_url, tuple(args.sources.split(",")), _read_symbols(args.symbols_file)
    )
    return _archive_new(jsonl_dir, seen, rows, stats)


def _chunked_sweep(base_url: str, symbols: tuple[str, ...]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats: dict = {"fetched": {}, "errors": {}}
    for start in range(0, len(symbols), PERNAME_CHUNK_SIZE):
        if start:
            time.sleep(PERNAME_COOLDOWN)
        chunk_rows, chunk_stats = fetch_symbols(
            base_url, symbols[start : start + PERNAME_CHUNK_SIZE], PERNAME_REQUEST_PAUSE
        )
        rows.extend(chunk_rows)
        stats["fetched"].update(chunk_stats["fetched"])
        stats["errors"].update(chunk_stats["errors"])
    return rows, stats


def run_fetch_pername(args: argparse.Namespace) -> int:
    symbols = _read_symbols(args.symbols_file)
    if not symbols:
        print(f"no symbols in {args.symbols_file}", file=sys.stderr)
        return 1
    jsonl_dir = Path(args.dir)
    seen = load_seen_ids(jsonl_dir, PERNAME_DEDUPE_LOOKBACK_DAYS)
    rows, stats = _chunked_sweep(args.base_url, symbols)
    for _ in range(PERNAME_RETRY_ROUNDS):
        failed = tuple(k.split(":", 1)[1] for k in stats["errors"])
        if not failed:
            break
        time.sleep(PERNAME_COOLDOWN)
        retry_rows, retry_stats = _chunked_sweep(args.base_url, failed)
        rows.extend(retry_rows)
        stats["fetched"].update(retry_stats["fetched"])
        stats["errors"] = retry_stats["errors"]
    return _archive_new(jsonl_dir, seen, rows, stats)


def run_loop(args: argparse.Namespace) -> int:
    while True:
        run_fetch(args)
        if args.db:
            build_db(Path(args.jsonl_dir), Path(args.db))
        time.sleep(args.interval + random.uniform(0, args.interval * 0.1))


def run_build_db(args: argparse.Namespace) -> int:
    result = build_db(Path(args.jsonl_dir), Path(args.db))
    print(json.dumps(result))
    return 0


def run_status(args: argparse.Namespace) -> int:
    hb_dir = Path(args.jsonl_dir) / "heartbeats"
    beats = sorted(hb_dir.glob("????-??-??.jsonl")) if hb_dir.is_dir() else []
    if not beats:
        print("no heartbeats recorded")
        return 1
    last = json.loads(beats[-1].read_text().splitlines()[-1])
    day_files = len(list(Path(args.jsonl_dir).glob("????-??-??.jsonl")))
    print(f"day_files={day_files} last_heartbeat={last['ts']} new_items={last['new_items']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="feeds_archive")
    parser.add_argument("--jsonl-dir", default=str(DEFAULT_JSONL_DIR))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="one-shot poll, append new items")
    p_loop = sub.add_parser("loop", help="poll forever (appliance mode)")
    for p in (p_fetch, p_loop):
        p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
        p.add_argument("--symbols-file", default=None)
    p_loop.add_argument("--interval", type=int, default=300)
    p_loop.add_argument("--db", default=str(DEFAULT_DB))
    p_fetch.set_defaults(func=run_fetch)
    p_loop.set_defaults(func=run_loop)

    p_pn = sub.add_parser(
        "fetch-pername", help="daily per-name news sweep into its own archive dir"
    )
    p_pn.add_argument("--dir", default=str(DEFAULT_PERNAME_DIR))
    p_pn.add_argument("--symbols-file", default=str(DEFAULT_PERNAME_UNIVERSE))
    p_pn.set_defaults(func=run_fetch_pername)

    p_db = sub.add_parser("build-db", help="compile JSONL into the SQLite index")
    p_db.add_argument("--db", default=str(DEFAULT_DB))
    p_db.set_defaults(func=run_build_db)

    p_status = sub.add_parser("status", help="show archive freshness")
    p_status.set_defaults(func=run_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
