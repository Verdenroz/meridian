"""Per-name text race: negative-tone *breadth* as a regime feature,
pre-registered before any result was computed.

Motivation, from the accumulated record: per-name text is the last open
text axis. The tone study's per-name content leg is the only per-name
text signal that ever passed (firm-level negative-tone days +6.5% vol
uplift over a matched news-active control, p=0.008) — yet every deployed
tone shape so far is *market-level intensity* (one mean over all
headlines, dominated by market-wide coverage). Breadth is the per-name
quantity intensity cannot see: the fraction of covered names having a
firm-specific negative-tone day at once — systemic text stress (many
names in distress together) vs idiosyncratic (one name's scandal at the
same market-wide mean). Registered risks, honestly: the dispersion race
closed price-space localization with "the third state already extracts
it" — text-space breadth may replay that; and 9 features under 3
Gaussian states sits past the octet race's capacity edge, so the 9f
stack may interfere the way 8f-turn-sal did. Those are the questions
the race adjudicates, not assumptions.

Feature, fixed a priori: per (symbol, UTC date) over the full FNSPID
corpus (482 symbols, the market leg's universe), a firm negative-tone
day is >= 3 scored headlines with mean signed FinBERT tone <= -0.3 —
both constants inherited frozen from the tone study; a covered day is
the same >= 3-headline floor regardless of tone. UTC days belong to the
first session on or after them (the tone race's availability clock).
Session breadth = negative / covered, defined on sessions with >= 30
covered names; then causal z (window 252, min obs 120 — the inherited
convention), zero-filled where undefined, scored on the coverage-clipped
span (the tonefeat pattern). Causal by construction. Known caveats
inherited from the tone study: FNSPID carries publisher timestamps, not
fetched_at; FinBERT is a 2019-era model scoring older text. Local-only
adjudication: the sentiment and FNSPID caches live on this machine, so
the race is excluded from the CI monthly re-race; breadth is zero after
2024-01 until archive headlines gain per-name attribution (future work
under its own registration).

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + breadth (5f-br) — does the channel carry value alone?
2. spine 6f + breadth (7f-br) — does it compound on the macro channels,
   or dilute (the quality-race failure mode)?
3. spine 6f + turnover + tone + breadth 3-state (9f-turn-tone-br) — the
   seat *beyond market-tone intensity* on the clipped-span leader
   (8f-turn-tone 3-state), the claim this race actually makes.
References (non-gating): buy-hold, the untouched null, spine 6f 2-state,
spine 6f 3-state, spine 7f-turn 3-state (full-span leading candidate),
spine 8f-turn-tone 3-state (clipped-span leader).

Gate: each challenger clears the charter's amended bar (§1.3) against
the untouched vol-target null on the clipped span — outright after-cost
Sharpe win, or held route (b): non-inferiority 0.05 + max drawdown <=
2/3 of the null's + beat of the matched-vol placebo. Same engine, same
panel (SPY/TLT/GLD). Changing any constant after seeing results
invalidates the race.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from event_study.events import FNSPID_CACHE
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.signals import causal_zscores
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from tone_study.study import daily_tone
from tonefeat_race.race import tone_feature
from turnover_race.race import turnover_feature

__all__ = ["breadth_feature", "run_race"]

GATE = {
    # Inherited frozen from the tone study's registration.
    "min_headlines": 3,
    "negative_tone": -0.3,
    # Breadth's own constants, fixed a priori.
    "min_names": 30,
    "z_window": 252,
    "z_min_obs": 120,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-br (+breadth)",
    "spine 7f-br (6f +breadth)",
    "spine 9f-turn-tone-br 3-state",
)


def breadth_feature(dates: list[str]) -> np.ndarray:
    """Causal z of negative-tone breadth (negative / covered names per
    session); 0.0 where undefined (under-coverage and z warm-up)."""
    covered = np.zeros(len(dates))
    negative = np.zeros(len(dates))
    for path in sorted(FNSPID_CACHE.glob("*.jsonl")):
        for date, (n, mean) in daily_tone(path.stem).items():
            if n < int(GATE["min_headlines"]):
                continue
            i = bisect_left(dates, date)
            if i < len(dates):
                covered[i] += 1
                if mean <= float(GATE["negative_tone"]):
                    negative[i] += 1

    breadth = np.full(len(dates), np.nan)
    mask = covered >= int(GATE["min_names"])
    breadth[mask] = negative[mask] / covered[mask]
    z = causal_zscores(breadth, int(GATE["z_window"]), int(GATE["z_min_obs"]))
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
    br = breadth_feature(panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_br = np.column_stack([features4, br])
    f7_br = np.column_stack([f6, br])
    f7_turn = np.column_stack([f6, turn])
    f8_tone = np.column_stack([f6, turn, tone])
    f9 = np.column_stack([f6, turn, tone, br])
    first = panel.first_index_with_data(("SPY",))
    n3 = int(GATE["n_states_stack"])

    nonzero = np.flatnonzero(br != 0.0)
    if not len(nonzero):
        raise RuntimeError("breadth feature never warms; check FNSPID/sentiment caches")
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
        "spine 8f-turn-tone 3-state (ref)": lambda: TunedSpineStrategy(
            panel, f8_tone, n_states=n3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_br),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_br),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f9, n_states=n3),
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
