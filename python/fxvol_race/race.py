"""FX-vol race: currency turbulence (carry/funding stress) as a regime
feature, pre-registered before any result was computed.

Motivation, from the accumulated record: the working pattern is handing
the regime model an orthogonal stress channel with a named economic
meaning, and the learned-race postmortems prescribed new *data*. Currency
turbulence is the funding-stress channel no current feature sees: carry
unwinds and safe-haven FX moves front-run or amplify equity stress (1992
ERM, 1997-98 Asia/LTCM yen unwind, 2008, 2015 CHF/China, the 2022 yen
slide), and the yen is the classic funding currency. FRED DEXJPUS (1971+,
the deepest daily series in the store — pre-dates SPY) and DEXUSEU
(1999+) are revision-free market prices, point-in-time clean, untouched
by any race. Distinct from the cross-asset race's channel: country-ETF
vol measures foreign *equity* turbulence in USD terms (an equity/FX
mixture); this is the FX leg isolated. Registered risk, honestly: that
race also showed a real orthogonal channel can be *substitutable* with
the leader's third state — the 8f challenger exists to adjudicate exactly
that, not to be assumed.

Feature, fixed a priori: per-pair daily log returns of the forward-filled
rate; per-pair log EWMA vol (incumbent estimator unchanged — lambda 0.94,
min 60 obs, annualized); per-pair causal z (window 252, min obs 120 — the
event-study convention), so the euro's 1999 entry shifts no levels; then
the mean over pairs defined at t (>= 1), zero-filled where none is
defined so early walk-forward windows stay fittable (the salience-race
pattern for staggered coverage). Causal by construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + fxvol (5f-fx) — does the channel carry value alone?
2. spine 6f + fxvol (7f-fx) — does it compound on the macro channels, or
   dilute (the quality-race failure mode)?
3. spine 6f + turnover + fxvol with the validated 3-state regime model
   (8f-turn-fx 3-state) — the compounding test on the leading candidate
   itself (n=3 inherited from the nstate race, not scanned).
References (non-gating): buy-hold, the untouched null, spine 6f 2-state,
spine 6f 3-state, spine 7f-turn 3-state (the leading candidate).

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of
the matched-vol placebo. Full walk-forward span, same engine, same panel
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
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.signals import causal_zscores
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.features import _ffill_series, _macro_by_date, build_features
from spine.strategy import TunedSpineStrategy
from spine.vol import ewma_vol_series
from turnover_race.race import turnover_feature

__all__ = ["fx_vol_feature", "run_race"]

FX_PAIRS = ("DEXJPUS", "DEXUSEU")
GATE = {
    "min_pairs": 1,
    "z_window": 252,
    "z_min_obs": 120,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-fx (+fxvol)",
    "spine 7f-fx (6f +fxvol)",
    "spine 8f-turn-fx 3-state",
)


def fx_vol_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Mean per-pair causal z of log EWMA FX vol; 0.0 where undefined
    (z warm-up; per-pair z so staggered inceptions shift no levels)."""
    n = len(dates)
    zs = np.full((n, len(FX_PAIRS)), np.nan)
    for j, series_id in enumerate(FX_PAIRS):
        level = _ffill_series(dates, _macro_by_date(store_dir, series_id, "value"))
        returns = np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            returns[1:] = np.log(level[1:] / level[:-1])
            log_vol = np.log(ewma_vol_series(returns))
        zs[:, j] = causal_zscores(log_vol, int(GATE["z_window"]), int(GATE["z_min_obs"]))

    feature = np.zeros(n)
    defined = np.isfinite(zs).sum(axis=1) >= int(GATE["min_pairs"])
    for t in np.flatnonzero(defined):
        row = zs[t][np.isfinite(zs[t])]
        feature[t] = row.mean()
    return feature


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    fx = fx_vol_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_fx = np.column_stack([features4, fx])
    f7_fx = np.column_stack([f6, fx])
    f7_turn = np.column_stack([f6, turn])
    f8 = np.column_stack([f6, turn, fx])
    first = panel.first_index_with_data(("SPY",))
    n3 = int(GATE["n_states_stack"])

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 2-state (ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 6f 3-state (ref)": lambda: TunedSpineStrategy(panel, f6, n_states=n3),
        "spine 7f-turn 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f7_turn, n_states=n3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_fx),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_fx),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f8, n_states=n3),
    }
    runs = race_contenders(panel, contenders, first)
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "fx_pairs": FX_PAIRS,
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
