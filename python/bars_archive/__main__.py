"""CLI: python -m bars_archive {fetch|status}"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from feeds_archive.store import append_heartbeat

from . import DEFAULT_BASE_URL, UNIVERSE
from .fetch import by_session, completed, fetch_bars
from .store import coverage, session_path, write_session

DEFAULT_ROOT = Path("archive/bars")


def archive_symbol(
    root: Path, base_url: str, symbol: str, interval: str, now: datetime
) -> int:
    fetched_at = now.isoformat(timespec="seconds")
    written = 0
    for session, bars in by_session(fetch_bars(base_url, symbol, interval, now)).items():
        if not completed(session, now):
            continue
        rows = [{"symbol": symbol, **bar, "fetched_at": fetched_at} for bar in bars]
        written += write_session(session_path(root, interval, session, symbol), rows)
    return written


def run_fetch(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    now = datetime.now(UTC)
    stats: dict = {"written": {}, "errors": {}}
    for interval, symbols in UNIVERSE.items():
        for symbol in symbols:
            key = f"{interval}:{symbol}"
            try:
                stats["written"][key] = archive_symbol(root, args.base_url, symbol, interval, now)
            except Exception as exc:  # noqa: BLE001 - one symbol's failure must not sink the sweep
                stats["errors"][key] = str(exc)[:300]
    written = sum(stats["written"].values())
    append_heartbeat(root, written, stats)
    print(f"sessions_written={written} errors={len(stats['errors'])}")
    if stats["errors"]:
        print(json.dumps(stats["errors"], indent=2), file=sys.stderr)
        return 1
    return 0


def run_status(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    gaps = 0
    for interval, symbols in UNIVERSE.items():
        sessions = coverage(root, interval)
        missing = {s: sorted(set(symbols) - have) for s, have in sessions.items()}
        missing = {s: m for s, m in missing.items() if m}
        gaps += len(missing)
        span = f"{min(sessions)}..{max(sessions)}" if sessions else "empty"
        print(f"{interval}: sessions={len(sessions)} span={span} incomplete={len(missing)}")
        for session, names in missing.items():
            print(f"  {session} missing {','.join(names)}")
    return 1 if gaps else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bars_archive")
    parser.add_argument("--dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="archive every completed session upstream still holds").set_defaults(
        func=run_fetch
    )
    sub.add_parser("status", help="sessions per interval and symbol gaps").set_defaults(
        func=run_status
    )
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
