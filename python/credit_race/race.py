"""Credit-spread race: the step-5 diagnostic's named lane, pre-registered.

Credit stress is a different information channel from equity vol — the one
thing the dead 2-state HMM's all-equity-derived features lacked. Feature:
Moody's Baa−10Y spread (FRED BAA10Y, daily since 1986; the ICE BofA OAS
series are truncated to ~3y upstream and unusable). Two challengers, both
fixed a priori; the race passes if either beats the untouched null:

1. Regime route — the tuned spine with a 5th feature, the spread level,
   through the frozen tuner candidate set. The 4-feature tuned spine runs
   as a non-gating reference so the credit contribution is attributable.
2. Brake route — the null vol-target with equity halved for 5 sessions
   after a credit-widening burst: causal trailing z of the 5-day spread
   change >= 2. Mechanics and constants inherited frozen from the
   occurrence-flag race (scale 0.5, 5 sessions, z window 252/min 120).

GATE below was fixed before any result was computed; changing it after
seeing results invalidates the race. Gate: either challenger's after-cost
walk-forward OOS Sharpe > the null vol-target's, same span, same engine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.signals import BrakeVolTarget, brake_series, causal_zscores
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from spine.features import _ffill_series, _macro_by_date, build_features
from spine.strategy import TunedSpineStrategy

__all__ = ["causal_zscores", "run_race", "widening_brake"]

SERIES = "BAA10Y"
GATE = {
    "change_days": 5,
    "burst_z": 2.0,
    "brake_scale": 0.5,
    "brake_days": 5,
    "z_window": 252,
    "z_min_obs": 120,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")


def credit_spread_series(store_dir: Path, dates: list[str]) -> np.ndarray:
    return _ffill_series(dates, _macro_by_date(store_dir, SERIES, "value"))


def widening_brake(spread: np.ndarray) -> np.ndarray:
    """Brake-active series from the causal z of the 5-day spread change."""
    delta = np.full(len(spread), np.nan)
    d = int(GATE["change_days"])
    delta[d:] = spread[d:] - spread[:-d]
    z = causal_zscores(delta, int(GATE["z_window"]), int(GATE["z_min_obs"]))
    return brake_series(z, GATE["burst_z"], int(GATE["brake_days"]))


def build_features5(panel: Panel, store_dir: Path) -> np.ndarray:
    """The spine's 4 regime features plus the credit spread level."""
    spread = credit_spread_series(store_dir, panel.dates)
    return np.column_stack([build_features(panel, store_dir), spread])


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    features5 = build_features5(panel, store_dir)
    spread = credit_spread_series(store_dir, panel.dates)
    active = widening_brake(spread)
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15% (null)": lambda: VolTarget("SPY"),
        "vol-target + credit brake": lambda: BrakeVolTarget(
            "SPY", active, GATE["brake_scale"]
        ),
        "spine tuned 4f (reference)": lambda: TunedSpineStrategy(panel, features4),
        "spine tuned 5f (+credit)": lambda: TunedSpineStrategy(panel, features5),
    }
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    metrics: dict[str, dict] = {}
    turnover: dict[str, float] = {}
    oos_span = ("", "")
    for label, factory in contenders.items():
        result = run_walk_forward(panel, factory, first)
        years = len(result["returns"]) / 252
        turnover[label] = sum(w["turnover_oneway"] for w in result["windows"]) / years
        metrics[label] = compute_metrics(
            result["returns"], [cash_by_date[d] for d in result["dates"]]
        )
        oos_span = (result["dates"][0], result["dates"][-1])

    null_sharpe = metrics["vol-target SPY 15% (null)"]["sharpe"]
    challengers = ("vol-target + credit brake", "spine tuned 5f (+credit)")
    winners = [c for c in challengers if metrics[c]["sharpe"] > null_sharpe]
    return {
        "gate": dict(GATE),
        "series": SERIES,
        "brake_active_days": int(active.sum()),
        "n_days": panel.n_days,
        "metrics": metrics,
        "turnover": turnover,
        "oos_span": oos_span,
        "verdict": {
            "null_sharpe": null_sharpe,
            "winners": winners,
            "race_pass": bool(winners),
        },
    }
