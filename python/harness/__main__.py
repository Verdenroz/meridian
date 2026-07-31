"""CLI: run the null baseline race.

    PYTHONPATH=python python -m harness null

Prints after-cost metrics for buy-and-hold SPY and SPY + vol-target, full
sample and stitched walk-forward out-of-sample. All future Meridian claims are
measured against these numbers in this same engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.strategies import BuyHold, VolTarget
from harness.engine import run_backtest
from harness.walkforward import run_walk_forward
from market_data.store import DEFAULT_STORE_DIR

_COLUMNS = ("cagr", "ann_vol", "sharpe", "max_drawdown", "worst_day")


def _row(label: str, metrics: dict, turnover_py: float) -> str:
    cells = "".join(
        f"{metrics[c] * (100 if c != 'sharpe' else 1):>10.2f}" for c in _COLUMNS
    )
    return f"{label:<28}{cells}{turnover_py:>10.2f}"


def _print_table(title: str, rows: list[str]) -> None:
    header = "".join(f"{c:>10}" for c in ("cagr%", "vol%", "sharpe", "maxdd%", "worst%"))
    print(f"\n{title}")
    print(f"{'strategy':<28}{header}{'to/yr':>10}")
    for row in rows:
        print(row)


def _cash_slice(panel: Panel, dates: list[str]) -> list[float]:
    by_date = dict(zip(panel.dates, panel.cash_returns))
    return [by_date[d] for d in dates]


def cmd_null(args: argparse.Namespace) -> int:
    panel = build_panel(args.store_dir, (args.symbol,))
    first = panel.first_index_with_data((args.symbol,))
    strategies: dict[str, Callable[[], Strategy]] = {
        f"buy-hold {args.symbol}": lambda: BuyHold(args.symbol),
        f"vol-target {args.symbol} 15%": lambda: VolTarget(args.symbol),
    }

    full_rows = []
    for label, factory in strategies.items():
        result = run_backtest(panel, factory(), start=first)
        years = len(result["returns"]) / 252
        metrics = compute_metrics(
            result["returns"], _cash_slice(panel, result["dates"])
        )
        full_rows.append(_row(label, metrics, result["turnover_oneway"] / years))
    _print_table(
        f"full sample ({panel.dates[first]}..{panel.dates[-1]}, after costs)",
        full_rows,
    )

    oos_rows = []
    for label, factory in strategies.items():
        result = run_walk_forward(panel, factory, first_index=first)
        years = len(result["returns"]) / 252
        turnover = sum(w["turnover_oneway"] for w in result["windows"])
        metrics = compute_metrics(
            result["returns"], _cash_slice(panel, result["dates"])
        )
        oos_rows.append(_row(label, metrics, turnover / years))
    _print_table(
        f"walk-forward OOS ({result['dates'][0]}..{result['dates'][-1]}, "
        f"{len(result['windows'])} windows, after costs)",
        oos_rows,
    )
    print(
        "\nnote: survivorship bias is a known limitation of Yahoo-sourced "
        "history; SPY/ETF series are unaffected, the stock sleeve will be."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    null = sub.add_parser("null", help="run the null baselines (SPY, vol-target)")
    null.add_argument("--symbol", default="SPY")
    null.set_defaults(func=cmd_null)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
