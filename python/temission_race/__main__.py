"""CLI: run the pre-registered Student-t emission race.

    uv run python -m temission_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from temission_race.race import run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    lo, hi = report["oos_span"]
    print(
        "pre-registered race: Student-t emission HMM (nu=5) on the 6f "
        f"features vs untouched vol-target null; walk-forward OOS {lo} → "
        f"{hi}, after costs\n"
    )
    print(metrics_table(report["metrics"], label_width=32))
    print()
    for label, v in report["verdict"]["per_challenger"].items():
        status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
        print(f"{label}: {status}")
        print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if report['verdict']['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="temission_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered Student-t emission race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
