"""CLI: the pre-registered content-scored news tone study.

    uv run python -m tone_study run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_data.store import DEFAULT_STORE_DIR

from tone_study.study import run_study


def main() -> int:
    parser = argparse.ArgumentParser(prog="tone_study")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    args = parser.parse_args()
    report = run_study(args.store_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
