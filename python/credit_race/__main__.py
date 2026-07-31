"""CLI: run the pre-registered credit-spread race.

    uv run python -m credit_race run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.store import DEFAULT_STORE_DIR

from credit_race.race import run_race

_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day")


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    gate = report["gate"]
    print(
        f"pre-registered race on {report['series']}: 5f spine and widening "
        f"brake (z ≥ {gate['burst_z']} of {gate['change_days']}d change, "
        f"scale {gate['brake_scale']}, {gate['brake_days']}d) vs the null"
    )
    print(
        f"brake active {report['brake_active_days']} of {report['n_days']} days; "
        f"OOS {report['oos_span'][0]}..{report['oos_span'][1]}, after costs\n"
    )
    header = "".join(f"{c:>10}" for c in ("cagr%", "vol%", "sharpe", "maxdd%", "worst%"))
    print(f"{'strategy':<28}{header}{'to/yr':>10}")
    for label, metrics in report["metrics"].items():
        cells = "".join(
            f"{metrics[c] * (100 if c != 'sharpe' else 1):>10.2f}" for c in _COLUMNS
        )
        print(f"{label:<28}{cells}{report['turnover'][label]:>10.2f}")
    verdict = report["verdict"]
    print(
        f"\nverdict: null sharpe {verdict['null_sharpe']:.2f}; "
        + (
            f"PASS — {', '.join(verdict['winners'])} beat(s) the null"
            if verdict["race_pass"]
            else "FAIL — the null stands"
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="credit_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered credit-spread race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
