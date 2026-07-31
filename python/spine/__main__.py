"""CLI: race the classical spine against the null baselines.

    uv run python -m spine race       # walk-forward OOS, same engine as nulls
    uv run python -m spine vol-eval   # EWMA forecast quality vs probing baseline
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import build_panel
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from market_data.store import DEFAULT_STORE_DIR

from spine.features import build_features
from spine.strategy import SpineStrategy, TunedSpineStrategy
from spine.vol import ewma_vol_series, forecast_correlation, forward_realized_vol

RACE_SYMBOLS = ("SPY", "TLT", "GLD")
_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day")


def cmd_race(args: argparse.Namespace) -> int:
    panel = build_panel(args.store_dir, RACE_SYMBOLS)
    features = build_features(panel, args.store_dir)
    first = panel.first_index_with_data(("SPY",))

    tuned_instances: list[TunedSpineStrategy] = []

    def tuned_factory() -> Strategy:
        strategy = TunedSpineStrategy(panel, features)
        tuned_instances.append(strategy)
        return strategy

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15%": lambda: VolTarget("SPY"),
        "spine v0 (2-state HMM)": lambda: SpineStrategy(panel, features),
        "spine tuned (train-select)": tuned_factory,
    }

    print(f"walk-forward OOS race, after costs (universe {', '.join(RACE_SYMBOLS)} + cash)")
    header = "".join(f"{c:>10}" for c in ("cagr%", "vol%", "sharpe", "maxdd%", "worst%"))
    print(f"{'strategy':<28}{header}{'to/yr':>10}")
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    for label, factory in contenders.items():
        result = run_walk_forward(panel, factory, first_index=first)
        years = len(result["returns"]) / 252
        turnover = sum(w["turnover_oneway"] for w in result["windows"]) / years
        metrics = compute_metrics(
            result["returns"], [cash_by_date[d] for d in result["dates"]]
        )
        cells = "".join(
            f"{metrics[c] * (100 if c != 'sharpe' else 1):>10.2f}" for c in _COLUMNS
        )
        print(f"{label:<28}{cells}{turnover:>10.2f}")
    print(
        f"\nOOS span: {result['dates'][0]}..{result['dates'][-1]} "
        f"({len(result['windows'])} windows). The spine ships only if it beats "
        "the nulls here."
    )
    if tuned_instances:
        counts: dict[str, int] = {}
        for strategy in tuned_instances:
            if strategy.chosen is not None:
                label = strategy.chosen.label()
                counts[label] = counts.get(label, 0) + 1
        picks = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
        print(f"tuner picks per window (train-data selection): {picks}")
    return 0


def cmd_vol_eval(args: argparse.Namespace) -> int:
    panel = build_panel(args.store_dir, ("SPY",))
    returns = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
    corr = forecast_correlation(
        ewma_vol_series(returns), forward_realized_vol(returns)
    )
    print(f"EWMA(0.94) forecast vs 21d forward realized vol corr: {corr:.3f}")
    print("baselines — naive: 0.60, probing incumbent bar: 0.67")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="spine")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    race = sub.add_parser("race", help="walk-forward race: spine vs null baselines")
    race.set_defaults(func=cmd_race)
    vol_eval = sub.add_parser("vol-eval", help="EWMA vol forecast quality")
    vol_eval.set_defaults(func=cmd_vol_eval)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
