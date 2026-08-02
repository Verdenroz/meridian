"""Oil-vol race: commodity/energy turbulence as a regime feature,
pre-registered before any result was computed.

Motivation, from the accumulated record: the only pattern that has ever
added OOS value is handing the regime model an orthogonal stress channel
with a named economic meaning (credit level, rate turbulence, turnover,
tone), and both learned-race postmortems concluded the next axis is new
*data*. Energy is a stress *kind* no current channel names: oil-led
regimes (1990 Gulf, the 2008 spike, 2014-16 collapse, the 2020 negative
settlement, 2022 invasion) transmit to equities through inflation and
margins rather than through the US price/credit/rate channels the spine
already reads. FRED DCOILWTICO (WTI spot, 1986+, 10,210 obs, revision-free
market prices — point-in-time clean) has never been touched by any race.
Registered risk, honestly: the cross-asset race showed a real orthogonal
stress channel can be *substitutable* with the 3-state leader's third
state (real information, seat negative on the leader); oil may replay
that. That is exactly what the race adjudicates — the 8f challenger exists
to answer it, not to be assumed.

Feature, fixed a priori: WTI spot level forward-filled to panel sessions
with non-positive prints treated as missing (the 2020-04-20 negative
settlement is a delivery artifact, not a tradable level); daily log
returns; then the incumbent vol estimator unchanged (log EWMA vol —
lambda 0.94, min 60 obs, annualized), matching the rate-vol channel's
shape. Causal by construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + oilvol (5f-oil) — does the channel carry value alone?
2. spine 6f + oilvol (7f-oil) — does it compound on the macro channels,
   or dilute (the quality-race failure mode)?
3. spine 6f + turnover + oilvol with the validated 3-state regime model
   (8f-turn-oil 3-state) — the compounding test on the leading candidate
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
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.features import _ffill_series, _macro_by_date, build_features
from spine.strategy import TunedSpineStrategy
from spine.vol import ewma_vol_series
from turnover_race.race import turnover_feature

__all__ = ["oil_vol_feature", "run_race"]

SERIES = "DCOILWTICO"
GATE = {
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-oil (+oilvol)",
    "spine 7f-oil (6f +oilvol)",
    "spine 8f-turn-oil 3-state",
)


def oil_vol_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """log EWMA vol of daily WTI log returns, aligned to panel sessions;
    non-positive prints treated as missing."""
    by_date = {
        d: v for d, v in _macro_by_date(store_dir, SERIES, "value").items() if v > 0
    }
    level = _ffill_series(dates, by_date)
    returns = np.full(len(level), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns[1:] = np.log(level[1:] / level[:-1])
        return np.log(ewma_vol_series(returns))


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    oil = oil_vol_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_oil = np.column_stack([features4, oil])
    f7_oil = np.column_stack([f6, oil])
    f7_turn = np.column_stack([f6, turn])
    f8 = np.column_stack([f6, turn, oil])
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
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_oil),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_oil),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f8, n_states=n3),
    }
    runs = race_contenders(panel, contenders, first)
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "series": SERIES,
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
