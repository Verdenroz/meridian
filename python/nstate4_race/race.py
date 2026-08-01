"""Four-state regime race: one more state than the leader, pre-registered
before any result was computed.

The 3-state race passed route (a) with the project's largest margin —
granularity, not features, was the binding constraint on the 6f spine.
This race asks the obvious next question once: does a *fourth* state
keep paying, or was three the sweet spot? The registered risk scales
with the answer: 4 Gaussian states on 6 features is ~4x the 2-state
parameter count on the same data, and the quality-race lesson (capacity
can dilute) applies. One race, one increment — this is not a scan over
n; whatever the verdict, further granularity work goes through learned
regime challengers (§1.7), not n+1 registrations.

Construction, fixed a priori: the rate_race 6f feature matrix unchanged;
TunedSpineStrategy with n_states=4 (states stay ordered by the vol
feature; the allocator reads the top-vol state as stress; the tuner's
candidate set is untouched). Nothing else moves.

Challenger: spine 6f 4-state. References (non-gating): buy-hold, the
untouched null, spine 6f 2-state, spine 6f 3-state (the leader).

Gate: the challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held
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
    "n_states": 4,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "spine 6f 4-state"


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
        "spine 6f 2-state (ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 6f 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f6, n_states=3
        ),
        CHALLENGER: lambda: TunedSpineStrategy(panel, f6, n_states=4),
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
