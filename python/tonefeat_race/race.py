"""Tone-feature race: headline tone through the regime model at last,
pre-registered before any result was computed.

This race closes the loop the record itself opened. The tone study PASSED
(the only text-content signal in the program: negative-tone days +6.5% vol
uplift, market tone spearman 0.128 on 5d residual vol); the tone-*brake*
race then FAILED — and its postmortem, consolidated across three brake
failures, says orthogonal information adds value through the *regime
model*, never as binary exposure brakes on the null. The strongest signal
diagnostic ever measured (spearman 0.144) has therefore never been tested
in the one deployment shape the record says works. That is this race.

Feature, fixed a priori — every constant inherited frozen from the tone
race's registration: session-aligned market negative-tone z
(min 30 headlines/session, causal z window 252 / min obs 120, the FNSPID
sentiment cache at the frozen scorer version), zero-filled where undefined
(the salience-race pattern). Scored on the tone-coverage-clipped span
(zero-filled tails would dilute an unclipped race toward the inherited
parents). Local-only adjudication: the sentiment cache lives on this
machine, so the race is excluded from the CI monthly re-race.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + tone (5f-tone) — does the channel carry value alone?
2. spine 6f + tone (7f-tone) — does it compound on the proven channels?
3. 8f-turn-tone 3-state (6f + turnover + tone, n=3) — the compounding test
   on the leading candidate (the octet race showed stacks can interfere;
   that is the question, not an assumption).
References (non-gating): buy-hold, the untouched null, spine 6f 2-state,
spine 6f 3-state, spine 7f-turn 3-state (the leading candidate).

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null on the clipped span — outright after-cost Sharpe
win, or held route (b): non-inferiority 0.05 + max drawdown <= 2/3 of the
null's + beat of the matched-vol placebo. Same engine, same panel
(SPY/TLT/GLD). Changing any constant after seeing results invalidates the
race.
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
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from tone_race.race import negative_tone_z, tone_by_session
from turnover_race.race import turnover_feature

__all__ = ["run_race", "tone_feature"]

GATE = {
    # Inherited frozen from the tone race's registration.
    "min_headlines": 30,
    "z_window": 252,
    "z_min_obs": 120,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-tone (+tone)",
    "spine 7f-tone (6f +tone)",
    "spine 8f-turn-tone 3-state",
)


def tone_feature(dates: list[str]) -> np.ndarray:
    """Causal z of session-aligned negative market tone; 0.0 where
    undefined (outside FNSPID coverage and z warm-up)."""
    tone = tone_by_session(dates, int(GATE["min_headlines"]))
    z = negative_tone_z(tone)
    return np.where(np.isfinite(z), z, 0.0)


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
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    tone = tone_feature(panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_tone = np.column_stack([features4, tone])
    f7_tone = np.column_stack([f6, tone])
    f7_turn = np.column_stack([f6, turn])
    f8 = np.column_stack([f6, turn, tone])
    first = panel.first_index_with_data(("SPY",))
    n3 = int(GATE["n_states_stack"])

    nonzero = np.flatnonzero(tone != 0.0)
    if not len(nonzero):
        raise RuntimeError("tone feature never warms; check sentiment cache")
    lo, hi = panel.dates[nonzero[0]], panel.dates[nonzero[-1]]
    run = _clipped_runner(lo, hi)

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 2-state (ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 6f 3-state (ref)": lambda: TunedSpineStrategy(panel, f6, n_states=n3),
        "spine 7f-turn 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f7_turn, n_states=n3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_tone),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_tone),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f8, n_states=n3),
    }
    runs = {label: run(panel, factory, first) for label, factory in contenders.items()}
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE, run=run)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "oos_span": (lo, hi),
        "n_days": panel.n_days,
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {
            "per_challenger": verdicts,
            "race_pass": any(v["passes"] for v in verdicts.values()),
        },
    }
