"""CLI: the pre-registered deep-history 8-K occurrence race.

    uv run python -m deep_eightk fetch    # full per-CIK 8-K histories
    uv run python -m deep_eightk run      # the race
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_study.study import PANEL
from market_data.client import DEFAULT_BASE_URL
from market_data.store import DEFAULT_STORE_DIR

from deep_eightk.fetch import fetch_full_8k_history
from deep_eightk.race import run_race

_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "turnover_oneway")


def cmd_fetch(args: argparse.Namespace) -> int:
    for symbol in PANEL:
        info = fetch_full_8k_history(args.base_url, args.store_dir, symbol)
        span = info["span"] or ("-", "-")
        print(f"  {symbol:<6} CIK {info['cik']}  {info['n_8k']:>4} 8-Ks  {span[0]}..{span[1]}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    print("deep 8-K occurrence race, after costs, walk-forward")
    print(
        f"OOS {report['oos_span'][0]}..{report['oos_span'][1]}, "
        f"panel {len(report['panel'])} names, "
        f"brake active {report['brake_active_days_oos']} OOS days"
    )
    header = f"{'strategy':<30}" + "".join(f"{c:>16}" for c in _COLUMNS)
    print(header)
    for label, m in report["metrics"].items():
        print(f"{label:<30}" + "".join(f"{m[c]:>16.4f}" for c in _COLUMNS))
    print()
    print(json.dumps({"verdict": report["verdict"],
                      "diagnostics": report["diagnostics"]}, indent=2))
    print(f"race_pass: {report['race_pass']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="deep_eightk")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("run")
    args = parser.parse_args()
    return cmd_fetch(args) if args.cmd == "fetch" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
