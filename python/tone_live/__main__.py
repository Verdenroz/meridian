"""CLI: score archived headlines and inspect the live tone store.

Scoring needs torch + transformers (not project deps):

    uv run --with torch --with transformers python -m tone_live score
    uv run python -m tone_live status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tone_live.live import ARCHIVE_DIR, STORE_PATH, score_new, status


def cmd_score(args: argparse.Namespace) -> int:
    written = score_new(args.archive_dir, args.store, args.limit)
    print(f"scored {written} new headlines -> {args.store}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(status(args.store), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tone_live")
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score new whitelisted archive headlines")
    score.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    score.add_argument("--limit", type=int, default=None)
    score.set_defaults(func=cmd_score)
    stat = sub.add_parser("status", help="store summary")
    stat.set_defaults(func=cmd_status)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
