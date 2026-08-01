"""Champion consolidation race: both passing axes at once, pre-registered
before any result was computed.

2026-08-01 produced two independent passing axes on the spine: the 7f
feature set (6f + topic salience; weak pass, held) and the 3-state
regime filter (largest margin of any race). The compound race taught us
composition is an empirical question, not an assumption — the veto
helped one parent and hurt the other. This race asks it once for the
two new axes: does the 3-state filter still lead with the salience
feature aboard, or do the axes interfere?

Construction, fixed a priori — everything inherited frozen: the
salience_race 7f matrix, n_states=3 from the nstate race, the tuner
untouched. Scored on the salience race's availability-clipped span
(salience is zero outside FNSPID coverage, so an unclipped span would
mostly measure the 3-state effect it inherits).

Challenger: spine 7f 3-state. References (non-gating): buy-hold, the
untouched null, spine 6f 3-state (the leader, on the same clipped
span), spine 7f 2-state (the salience parent).

Gate: the challenger clears the amended bar (§1.3) against the
untouched null on the clipped span. Changing any constant after seeing
results invalidates the race.
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

__all__ = ["run_race"]

GATE = {
    "n_states": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "spine 7f 3-state"


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
    sal = salience_feature(panel.dates)
    f7 = np.column_stack([f6, sal])
    first = panel.first_index_with_data(("SPY",))
    nonzero = np.flatnonzero(sal != 0.0)
    lo, hi = panel.dates[nonzero[0]], panel.dates[nonzero[-1]]
    run = _clipped_runner(lo, hi)

    n3 = int(GATE["n_states"])
    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f6, n_states=n3
        ),
        "spine 7f 2-state (parent ref)": lambda: TunedSpineStrategy(panel, f7),
        CHALLENGER: lambda: TunedSpineStrategy(panel, f7, n_states=n3),
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
