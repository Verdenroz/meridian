"""Regime-granularity race: does a third HMM state help the 6f spine?
Pre-registered before any result was computed.

Derived from the 2026-08-01 combination diagnostics: the three proven
stress channels fire on largely *disjoint* day sets (credit∩rate
Jaccard 0.15, credit∩tone 0.13, rate∩tone 0.17 on the OOS span) —
stress comes in kinds (equity-led, credit-led, rate-led), and a 2-state
model must compress them into one stress state with one blended
response. Hypothesis: with the 6f features carrying multiple channels,
a third state lets the filter separate stress kinds and the allocator
respond to the right one. The risk, equally registered: 3 Gaussian
states on 6 features is ~3x the parameters on the same data — the
quality-race lesson (more capacity can dilute) applies to states as
much as features.

Construction, fixed a priori: the rate_race 6f feature matrix unchanged;
TunedSpineStrategy with n_states=3 — the allocator already reads the
*last* state (states are ordered by the vol feature) as stress, so
p_stress remains the top-vol state's posterior; the tuner's candidate
set is untouched. Nothing else moves.

Challenger: spine 6f 3-state. References (non-gating): buy-hold, the
untouched null, spine 6f 2-state (the leader).

Gate: the challenger clears the charter's amended bar (§1.3) against
the untouched vol-target null — outright after-cost Sharpe win, or held
route (b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's +
beat of the matched-vol placebo. Same engine, full walk-forward span,
same panel (SPY/TLT/GLD). Changing any constant after seeing results
invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy

__all__ = ["run_race"]

GATE = {
    "n_states": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "spine 6f 3-state"


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    f6 = np.column_stack(
        [
            build_features(panel, store_dir),
            credit_spread_series(store_dir, panel.dates),
            rate_vol_feature(store_dir, panel.dates),
        ]
    )
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 2-state (leader ref)": lambda: TunedSpineStrategy(panel, f6),
        CHALLENGER: lambda: TunedSpineStrategy(
            panel, f6, n_states=int(GATE["n_states"])
        ),
    }
    runs = race_contenders(panel, contenders, first)
    verdict = amended_bar(runs[CHALLENGER], runs[NULL], panel, first, GATE)
    return {
        "gate": dict(GATE),
        "n_days": panel.n_days,
        "oos_span": (
            runs[NULL]["result"]["dates"][0],
            runs[NULL]["result"]["dates"][-1],
        ),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {"challenger": verdict, "race_pass": bool(verdict["passes"])},
    }
