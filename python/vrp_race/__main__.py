"""CLI: run the pre-registered VRP (premium-adjusted VIX sizing) race.

    uv run python -m vrp_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table
from market_data.store import DEFAULT_STORE_DIR

from vrp_race.race import run_race

_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "turnover_oneway")


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    print("VRP race: premium-adjusted VIX sizing, after costs, walk-forward")
    print(f"OOS {report['oos_span'][0]}..{report['oos_span'][1]}")
    print(metrics_table(report["metrics"], _COLUMNS, label_width=30, col_width=16))
    print()
    print(json.dumps(report["verdicts"], indent=2))
    print(f"race_pass: {report['race_pass']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="vrp_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    args = parser.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
