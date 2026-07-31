"""CLI: run the pre-registered event study.

    uv run python -m event_study run [--refresh]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.store import DEFAULT_STORE_DIR

from event_study.study import run_study


def _fmt_uplift(result: dict | None) -> str:
    if result is None:
        return "insufficient events"
    return (
        f"n={result['n_events']:>5}  uplift={result['median_uplift']*100:+6.1f}%  "
        f"effect={result['effect_size']:+.3f}  p={result['p_value']:.2e}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    report = run_study(args.store_dir, refresh=args.refresh)
    gate = report["gate"]
    print(
        f"pre-registered gate: uplift ≥ {gate['min_median_uplift']*100:.0f}% "
        f"@ p ≤ {gate['max_p_value']} (h={gate['horizon_days']}d), or market "
        f"Spearman ≥ {gate['market_min_spearman']} @ same p"
    )
    for horizon, legs in report["horizons"].items():
        print(f"\nhorizon {horizon}d:")
        for form, result in legs["filings"].items():
            print(f"  filings {form:<5} {_fmt_uplift(result)}")
        print(f"  news bursts   {_fmt_uplift(legs['news_bursts'])}")
        market = legs["market_intensity"]
        print(
            f"  market intensity  n={market['n_days']}  "
            f"spearman={market['spearman']:+.3f}  p={market['p_value']:.2e}"
        )
    verdict = report["verdict"]
    print(
        f"\nverdict: filings={'PASS' if verdict['filings_pass'] else 'fail'}, "
        f"news bursts={'PASS' if verdict['news_bursts_pass'] else 'fail'}, "
        f"market intensity={'PASS' if verdict['market_intensity_pass'] else 'fail'}"
    )
    print(f"overall: {'PASS — text pipeline unlocked' if verdict['overall_pass'] else 'FAIL — pipeline stays shelved'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="event_study")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered event study")
    run.add_argument("--refresh", action="store_true", help="refetch EDGAR filings")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
