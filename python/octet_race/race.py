"""Octet consolidation race: do turnover and salience coexist on the
3-state leader? Pre-registered before any result was computed.

The turnover race made 7f-turn 3-state the leading candidate (best-resolved
bootstrap margin yet); the champion race showed the salience feature and the
3-state axis compose additively on the salience-clipped span. The compound
race taught us composition is an empirical question — a helper on one parent
can subtract on another. This race asks it once for the full stack: with
the participation channel aboard, does salience still earn a seat, or is
the news-composition information redundant with (or diluted by) turnover?

Construction, fixed a priori — everything inherited frozen: the rate_race
6f matrix, the turnover_race feature, the salience_race feature, n_states=3
from the nstate race, the tuner untouched. Scored on the salience race's
availability-clipped span (salience is zero outside FNSPID coverage, so an
unclipped span would mostly measure the turnover effect it inherits).
Local-only adjudication like the salience race: the feature depends on the
local embed cache, so this race is excluded from the CI monthly re-race.

Challenger: spine 8f 3-state (6f + turnover + salience). References
(non-gating): buy-hold, the untouched null, spine 6f 3-state, spine 7f-turn
3-state (the leader, on the same clipped span), spine 7f-sal 3-state (the
salience-era variant) — so both marginal seats are attributable.

Gate: the challenger clears the amended bar (§1.3) against the untouched
null on the clipped span. Changing any constant after seeing results
invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from rate_race.race import rate_vol_feature
from salience_race.race import salience_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from turnover_race.race import turnover_feature

__all__ = ["run_race"]

GATE = {
    "n_states": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "spine 8f 3-state"


def _clipped_runner(lo: str, hi: str) -> Callable[..., dict]:
    def run(
        panel: Panel,
        factory: Callable[[], Strategy],
        first: int,
        oos_start: str | None = None,
    ) -> dict:
        result = run_walk_forward(panel, factory, first)
        cash_by_date = dict(zip(panel.dates, panel.cash_returns))
        kept = [
            (r, cash_by_date[d])
            for d, r in zip(result["dates"], result["returns"])
            if lo <= d <= hi
        ]
        metrics = compute_metrics([r for r, _ in kept], [c for _, c in kept])
        return {"result": result, "metrics": metrics, "oos_start": lo}

    return run


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    f6 = np.column_stack(
        [
            build_features(panel, store_dir),
            credit_spread_series(store_dir, panel.dates),
            rate_vol_feature(store_dir, panel.dates),
        ]
    )
    turn = turnover_feature(store_dir, panel.dates)
    sal = salience_feature(panel.dates)
    f7_turn = np.column_stack([f6, turn])
    f7_sal = np.column_stack([f6, sal])
    f8 = np.column_stack([f6, turn, sal])
    first = panel.first_index_with_data(("SPY",))
    nonzero = np.flatnonzero(sal != 0.0)
    lo, hi = panel.dates[nonzero[0]], panel.dates[nonzero[-1]]
    run = _clipped_runner(lo, hi)

    n3 = int(GATE["n_states"])
    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 3-state (ref)": lambda: TunedSpineStrategy(panel, f6, n_states=n3),
        "spine 7f-turn 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f7_turn, n_states=n3
        ),
        "spine 7f-sal 3-state (ref)": lambda: TunedSpineStrategy(
            panel, f7_sal, n_states=n3
        ),
        CHALLENGER: lambda: TunedSpineStrategy(panel, f8, n_states=n3),
    }
    runs = {label: run(panel, factory, first) for label, factory in contenders.items()}
    verdict = amended_bar(runs[CHALLENGER], runs[NULL], panel, first, GATE, run=run)
    return {
        "gate": dict(GATE),
        "oos_span": (lo, hi),
        "n_days": panel.n_days,
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {"challenger": verdict, "race_pass": bool(verdict["passes"])},
    }
