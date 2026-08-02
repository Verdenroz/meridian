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
from hedge_race.race import rate_veto_series
from rate_race.race import rate_vol_feature
from spine.strategy import TunedSpineStrategy
from tone_live.live import merged_tone_feature
from turnover_race.race import turnover_feature


def build_contenders(panel: Panel, store_dir: Path) -> dict[str, object]:
    features5 = build_features5(panel, store_dir)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    features5r = np.column_stack([features5[:, :4], ratevol])
    features6 = np.column_stack([features5, ratevol])
    features7t = np.column_stack(
        [features6, turnover_feature(store_dir, panel.dates)]
    )
    active = widening_brake(credit_spread_series(store_dir, panel.dates))
    veto = {"TLT": rate_veto_series(store_dir, panel.dates)}
    return {
        "null_voltarget": VolTarget("SPY"),
        "credit_brake": BrakeVolTarget("SPY", active, CREDIT_GATE["brake_scale"]),
        "spine5f_credit": TunedSpineStrategy(panel, features5),
        # Added 2026-08-01 with the rate/hedge race passes (held candidates).
        # spine5f_rate logs the parsimonious book: credit's marginal next to
        # rate vol bootstraps to nothing, so its seat is adjudicated live.
        "spine6f_rate": TunedSpineStrategy(panel, features6),
        "spine5f_rate": TunedSpineStrategy(panel, features5r),
        "spine5fc_veto": TunedSpineStrategy(panel, features5, safe_veto=veto),
        # Added 2026-08-01 with the regime-granularity pass (largest margin
        # yet; held). Three states separate stress kinds — see nstate_race.
        "spine6f_3state": TunedSpineStrategy(panel, features6, n_states=3),
        # Added 2026-08-01 with the turnover-race pass: dollar-volume
        # intensity compounds on every parent; seat adjudicated vs 6f_3state.
        "spine7f_turn_3state": TunedSpineStrategy(panel, features7t, n_states=3),
        # Added 2026-08-01 with the tone-feature pass (clipped-span leader).
        # Live tone is zero until its causal z warms (~120 covered sessions),
        # so early decisions track the 7f parent — logged anyway so the
        # record starts now, on the merged FNSPID+archive series.
        "spine8f_turn_tone_3state": TunedSpineStrategy(
            panel,
            np.column_stack([features7t, merged_tone_feature(panel.dates)]),
            n_states=3,
        ),
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
