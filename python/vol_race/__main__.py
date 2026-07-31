"""CLI: run the pre-registered vol-forecaster race.

    uv run python -m vol_race run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.store import DEFAULT_STORE_DIR

from vol_race.race import run_race

_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day")


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    gate = report["gate"]
    print(
        f"stage 1 — forecast/realized correlation (common sample; graduate if "
        f"> ewma + {gate['min_corr_margin']:.2f} at {gate['primary_horizon']}d):"
    )
    horizons = sorted(report["correlations"])
    print(f"{'candidate':<18}" + "".join(f"{h:>8}d" for h in horizons))
    for name in report["correlations"][gate["primary_horizon"]]:
        cells = "".join(
            f"{report['correlations'][h][name]:>9.3f}" for h in horizons
        )
        marker = "  <- graduates" if name in report["graduates"] else ""
        print(f"{name:<18}{cells}{marker}")
    if not report["graduates"]:
        print("\nno candidate clears the stage-1 bar — race not run")
    print(
        f"\nstage 2 — walk-forward OOS race, after costs "
        f"({report['oos_span'][0]}..{report['oos_span'][1]}):"
    )
    header = "".join(f"{c:>10}" for c in ("cagr%", "vol%", "sharpe", "maxdd%", "worst%"))
    print(f"{'strategy':<28}{header}")
    for label, metrics in report["metrics"].items():
        cells = "".join(
            f"{metrics[c] * (100 if c != 'sharpe' else 1):>10.2f}" for c in _COLUMNS
        )
        print(f"{label:<28}{cells}")
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
    parser = argparse.ArgumentParser(prog="vol_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered vol-forecaster race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
