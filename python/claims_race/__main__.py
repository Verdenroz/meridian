"""CLI: run the pre-registered initial-claims race.

    uv run python -m claims_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from claims_race.race import run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    lo, hi = report["oos_span"]
    print(
        "pre-registered race: initial-claims real-economy deterioration as "
        f"a regime feature vs untouched vol-target null; walk-forward OOS "
        f"{lo} → {hi}, after costs\n"
    )
    print(metrics_table(report["metrics"], label_width=34))
    print()
    for label, v in report["verdict"]["per_challenger"].items():
        status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
        print(f"{label}: {status}")
        print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if report['verdict']['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="claims_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered claims race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
