"""Range-vol race: Parkinson intraday-range volatility as the spine's vol
feature, pre-registered before any result was computed.

The last untouched chart fields are high/low. The gap race closed
*overnight* gaps as an additional channel; this race asks a different,
narrower question: measurement, not information. The spine's feature 0 —
the vol measure that orders HMM states and defines "stress" — is close-to-
close EWMA vol, which is noisy at a fixed window; the Parkinson range
estimator extracts intraday extremes and is ~5x more efficient per day on
diffusive price paths. Cleaner state separation from a better-measured
feature 0 is the hypothesis; the registered risk is the mirror of the
vol-forecaster lesson — a better estimator need not make a better book.

Feature, fixed a priori: feature 0 becomes log EWMA Parkinson vol — the
incumbent estimator applied verbatim to x_t = ln(high_t/low_t)/(2*sqrt(ln 2))
(so EWMA(x^2) is the Parkinson variance; lambda 0.94, min 60 obs,
annualized). Sessions with high <= low or missing extremes stay NaN.
Allocator sizing vol is untouched — this changes only what the regime
model *sees*. Everything else inherited frozen.

Challengers, both fixed a priori:
1. spine 4f-park — the base feature set with feature 0 swapped (does the
   measurement matter at all?).
2. spine 7f-turn-park 3-state — the leading candidate with feature 0
   swapped (the substitution that would matter).
References (non-gating): buy-hold, the untouched null, spine 4f (the
close-close base), spine 6f 3-state, spine 7f-turn 3-state (incumbent).

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of
the matched-vol placebo. Full walk-forward span, same engine, same panel
(SPY/TLT/GLD). Changing any constant after seeing results invalidates the
race.
"""

from __future__ import annotations

from collections.abc import Callable
from math import log, sqrt
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from market_data.store import read_snapshot
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from spine.vol import ewma_vol_series
from turnover_race.race import turnover_feature

__all__ = ["parkinson_feature", "run_race"]

GATE = {
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
MARKET_SYMBOL = "SPY"
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = ("spine 4f-park", "spine 7f-turn-park 3-state")

_PARKINSON_SCALE = 2.0 * sqrt(log(2.0))


def parkinson_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """log EWMA Parkinson vol of SPY, aligned to panel sessions."""
    doc = read_snapshot(store_dir, "charts", MARKET_SYMBOL)
    if doc is None:
        raise FileNotFoundError("SPY chart snapshot missing; run market_data fetch")
    hl = {
        r["date"]: (float(r["high"]), float(r["low"]))
        for r in doc["data"]["rows"]
        if r.get("high") and r.get("low")
    }
    x = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        pair = hl.get(d)
        if pair and pair[0] > pair[1] > 0:
            x[i] = np.log(pair[0] / pair[1]) / _PARKINSON_SCALE
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(ewma_vol_series(x))


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    park = parkinson_feature(store_dir, panel.dates)
    f4_park = np.column_stack([park, features4[:, 1:]])
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f7_turn = np.column_stack([f6, turn])
    f7_turn_park = np.column_stack([f4_park, credit, ratevol, turn])
    first = panel.first_index_with_data(("SPY",))
    n3 = int(GATE["n_states_stack"])

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 4f (ref)": lambda: TunedSpineStrategy(panel, features4),
        "spine 6f 3-state (ref)": lambda: TunedSpineStrategy(panel, f6, n_states=n3),
        "spine 7f-turn 3-state (incumbent ref)": lambda: TunedSpineStrategy(
            panel, f7_turn, n_states=n3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f4_park),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_turn_park, n_states=n3),
    }
    runs = race_contenders(panel, contenders, first)
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "n_days": panel.n_days,
        "oos_span": (
            runs[NULL]["result"]["dates"][0],
            runs[NULL]["result"]["dates"][-1],
        ),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {
            "per_challenger": verdicts,
            "race_pass": any(v["passes"] for v in verdicts.values()),
        },
    }
