"""CLI: the pre-registered stock-sleeve race.

    uv run python -m sleeve fetch   # ensure the frozen universe is in the store
    uv run python -m sleeve run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.client import DEFAULT_BASE_URL
from market_data.pipeline import ensure_symbol
from market_data.store import DEFAULT_STORE_DIR

from sleeve.race import run_race
from sleeve.universe import MARKET_SYMBOL, SLEEVE_UNIVERSE


def cmd_fetch(args: argparse.Namespace) -> int:
    for symbol in (MARKET_SYMBOL, *SLEEVE_UNIVERSE):
        ensure_symbol(args.base_url, args.store_dir, symbol)
        print(f"  {symbol}: stored")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    lo, hi = report["oos_span"]
    print(
        "pre-registered race: risk-weighted stock sleeve inside the null's "
        f"vol-target shell vs untouched null; walk-forward OOS {lo} → {hi}, "
        "after costs\n"
    )
    stage1 = report["stage1"]
    print("stage 1 — OOS forecast quality (per-name pearson / cross-sectional spearman):")
    for name, corr in stage1["primary_per_name_pearson"].items():
        xs = stage1["secondary_cross_sectional_spearman"][name]
        print(f"  {name:<10} {corr:.3f} / {xs:.3f}")
    print(f"  graduates: {stage1['graduates'] or 'none'}\n")
    print(metrics_table(report["metrics"], label_width=32))
    print()
    for label, v in report["verdict"]["per_challenger"].items():
        status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
        print(f"{label}: {status}")
        print(f"  {json.dumps(plain(v))}")
    if report["filled_gaps"]:
        print(f"\nfilled upstream candle holes: {report['filled_gaps']}")
    print("known limitation: survivorship-biased universe (today's names)")
    print(f"\nrace verdict: {'PASS' if report['verdict']['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="sleeve")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="store charts for the frozen sleeve universe")
    fetch.set_defaults(func=cmd_fetch)
    run = sub.add_parser("run", help="run the pre-registered stock-sleeve race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
