"""CLI: run the pre-registered quality-spread race.

    uv run python -m quality_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table
from market_data.store import DEFAULT_STORE_DIR

from quality_race.race import run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    print(f"quality-spread race ({report['series']}), after costs, walk-forward")
    print(
        f"OOS {report['oos_span'][0]}..{report['oos_span'][1]}, "
        f"brake active {report['brake_active_days']} days"
    )
    print(metrics_table(report["metrics"]))
    print()
    print(json.dumps(report["verdicts"], indent=2))
    print(f"race_pass: {report['race_pass']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="quality_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    args = parser.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
