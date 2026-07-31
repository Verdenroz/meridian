"""CLI: run the pre-registered occurrence-flag race.

    uv run python -m eightk_race run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.store import DEFAULT_STORE_DIR

from eightk_race.race import run_race

_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day")


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    gate = report["gate"]
    print(
        f"pre-registered race: brake (z ≥ {gate['burst_z']}, "
        f"scale {gate['brake_scale']}, {gate['brake_days']}d) vs untouched "
        f"vol-target null; OOS ≥ {report['oos_start']}, after costs"
    )
    print(
        f"brake active {report['brake_active_days']} of {report['n_days']} "
        "panel days (incl. pre-OOS warmup)\n"
    )
    header = "".join(f"{c:>10}" for c in ("cagr%", "vol%", "sharpe", "maxdd%", "worst%"))
    print(f"{'strategy':<28}{header}")
    for label, metrics in report["metrics"].items():
        cells = "".join(
            f"{metrics[c] * (100 if c != 'sharpe' else 1):>10.2f}" for c in _COLUMNS
        )
        print(f"{label:<28}{cells}")
    signal = report["signal"]
    print(
        f"\nsignal diagnostic (non-gating): n={signal['n_days']}  "
        f"spearman={signal['spearman']:+.3f}  p={signal['p_value']:.2e}"
    )
    verdict = report["verdict"]
    print(
        f"\nverdict: challenger sharpe {verdict['challenger_sharpe']:.2f} vs "
        f"null {verdict['null_sharpe']:.2f} — "
        + (
            "PASS — occurrence brake beats the null"
            if verdict["race_pass"]
            else "FAIL — the null stands"
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="eightk_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered occurrence-flag race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
