"""The shadow roster: exactly the configurations that were raced.

Each entry pairs a strategy factory with a meta extractor so the log
records why the weights are what they are. Contenders may be added or
retired freely (it's a shadow), but never edit a logged contender's
behavior in place — retire the name and add a new one, or the log's
history stops meaning anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from credit_race.race import (
    GATE as CREDIT_GATE,
)
from credit_race.race import (
    build_features5,
    credit_spread_series,
    widening_brake,
)
from eightk_race.race import BrakeVolTarget
from harness.panel import Panel
from harness.strategies import VolTarget
from harness.view import HistoryView
from spine.strategy import TunedSpineStrategy


def build_contenders(panel: Panel, store_dir: Path) -> dict[str, object]:
    features5 = build_features5(panel, store_dir)
    active = widening_brake(credit_spread_series(store_dir, panel.dates))
    return {
        "null_voltarget": VolTarget("SPY"),
        "credit_brake": BrakeVolTarget("SPY", active, CREDIT_GATE["brake_scale"]),
        "spine5f_credit": TunedSpineStrategy(panel, features5),
    }


def extract_meta(name: str, strategy: object, view: HistoryView) -> dict:
    """Small, stable explanation of the decision; never model internals."""
    if isinstance(strategy, BrakeVolTarget):
        return {"brake_active": bool(strategy.active[view.t])}
    if isinstance(strategy, VolTarget):
        vol = strategy._ewma_vol(view.returns_window(strategy.symbol, strategy.window))
        return {"vol_forecast": None if vol is None else round(float(vol), 6)}
    if isinstance(strategy, TunedSpineStrategy) and strategy.chosen is not None:
        return {"chosen_mapping": strategy.chosen.label()}
    return {}


def round_weights(weights: dict[str, float]) -> dict[str, float]:
    kept = {s: round(float(w), 6) for s, w in weights.items() if np.isfinite(w)}
    return {s: w for s, w in kept.items() if w != 0.0}
