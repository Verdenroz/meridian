"""CLI: run the pre-registered compound race.

    uv run python -m compound_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from compound_race.race import CHALLENGER, run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    lo, hi = report["oos_span"]
    print(
        "pre-registered race: 6f regime model + rate-aware TLT veto "
        f"(both parents route-(a) passes) vs untouched vol-target null; "
        f"walk-forward OOS {lo} → {hi}, after costs\n"
    )
    print(metrics_table(report["metrics"], label_width=32))
    v = report["verdict"]["challenger"]
    status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
    print(f"\n{CHALLENGER}: {status}")
    print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if report['verdict']['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="compound_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered compound race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
