"""CLI: run the pre-registered stock-bond correlation race.

    uv run python -m corr_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from corr_race.race import CHALLENGERS, run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    lo, hi = report["oos_span"]
    print(
        "pre-registered race: stock-bond correlation regime feature "
        f"(60-session SPY vs −ΔDGS10) vs untouched vol-target null; "
        f"walk-forward OOS {lo} → {hi}, after costs\n"
    )
    print(metrics_table(report["metrics"], label_width=30))
    verdict = report["verdict"]
    for label in CHALLENGERS:
        v = verdict["per_challenger"][label]
        status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
        print(f"\n{label}: {status}")
        print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if verdict['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="corr_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="run the pre-registered stock-bond correlation race"
    )
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
