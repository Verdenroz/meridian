"""CLI: pre-registered behavior-match evaluation of the shadow log.

    uv run python -m behavior_match criteria   # print the frozen constants
    uv run python -m behavior_match run        # evaluate (refuses before opening date)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_data.store import DEFAULT_STORE_DIR
from shadow.log import DEFAULT_LOG_PATH

from behavior_match.criteria import CRITERIA
from behavior_match.evaluate import evaluate


def cmd_criteria(args: argparse.Namespace) -> int:
    print(json.dumps(CRITERIA, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    report = evaluate(args.log_path, args.store_dir)
    print(json.dumps(report, indent=2))
    return 0 if all(s["verdict"] == "PASS" for s in report["strategies"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="behavior_match")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    crit = sub.add_parser("criteria", help="print the frozen pre-registration")
    crit.set_defaults(func=cmd_criteria)
    run = sub.add_parser("run", help="evaluate the closed shadow window")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
