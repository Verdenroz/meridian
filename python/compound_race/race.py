"""Compound race: the 6f regime model plus the rate-aware hedge veto,
pre-registered before any result was computed.

2026-08-01 produced the project's first two outright route-(a) passes,
and they act on *different levers*: the rate-vol regime feature changes
how the HMM sees the world (equity sizing via the posterior), while the
TLT veto changes only defensive-sleeve membership. Composing them is a
genuine compound hypothesis — if the mechanisms are independent, the
composition should retain both improvements; if they were two views of
the same episodes, the compound will add nothing over the better single.

Construction, fixed a priori — every constant inherited frozen from its
source race, nothing new to tune:
- Regime features: the rate_race 6f matrix (4f + BAA10Y level + log EWMA
  vol of daily DGS10 changes).
- Hedge veto: the hedge_race TLT veto (causal z of the rate-vol level
  >= 2 → veto 21 sessions).

Challenger: spine 6f + TLT veto. References (non-gating): buy-hold, the
untouched null, spine 6f alone, spine 5f-credit + veto — the two parents,
so the compound's delta over each is attributable.

Gate: the challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of
the matched-vol placebo. Same engine, full walk-forward span, same panel
(SPY/TLT/GLD). All passes today remain *held* pending the monthly
re-race and the step-9 shadow behavior-match; this race decides the
leading candidate, not a shipping decision. Changing any constant after
seeing results invalidates the race.
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
from hedge_race.race import rate_veto_series
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy

__all__ = ["run_race"]

GATE = {
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "spine 6f + TLT veto"


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    f5_credit = np.column_stack([features4, credit])
    f6 = np.column_stack([features4, credit, ratevol])
    veto = {"TLT": rate_veto_series(store_dir, panel.dates)}
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f (parent ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 5f-credit + veto (parent)": lambda: TunedSpineStrategy(
            panel, f5_credit, safe_veto=veto
        ),
        CHALLENGER: lambda: TunedSpineStrategy(panel, f6, safe_veto=veto),
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
