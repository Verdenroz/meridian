"""CLI: run the pre-registered tone regime-feature race (local-only:
needs the sentiment cache).

    uv run python -m tonefeat_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from tonefeat_race.race import run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    lo, hi = report["oos_span"]
    print(
        "pre-registered race: negative market tone as a regime feature vs "
        f"untouched vol-target null; clipped OOS {lo} → {hi}, after costs\n"
    )
    print(metrics_table(report["metrics"], label_width=36))
    print()
    for label, v in report["verdict"]["per_challenger"].items():
        status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
        print(f"{label}: {status}")
        print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if report['verdict']['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tonefeat_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered tone-feature race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
