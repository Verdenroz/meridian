"""CLI: run the pre-registered hedge-veto race.

    uv run python -m hedge_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from hedge_race.race import CHALLENGER, run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    gate = report["gate"]
    lo, hi = report["oos_span"]
    print(
        f"pre-registered race: TLT vetoed from the defensive sleeve for "
        f"{gate['veto_days']} sessions after a rate-vol burst (z ≥ "
        f"{gate['burst_z']}); walk-forward OOS {lo} → {hi}, after costs"
    )
    print(
        f"veto active {report['veto_active_days']} of {report['n_days']} "
        "panel days\n"
    )
    print(metrics_table(report["metrics"], label_width=30))
    v = report["verdict"]["challenger"]
    status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
    print(f"\n{CHALLENGER}: {status}")
    print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if report['verdict']['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="hedge_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered hedge-veto race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
