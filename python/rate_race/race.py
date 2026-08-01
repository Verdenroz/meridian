"""Rate-vol race: the bond-market stress channel as a regime feature,
pre-registered before any result was computed.

Motivation, from the accumulated race record: every binary brake on the
null has failed (8-K, credit, tone) — EWMA sizing already de-levers
through the same episodes — while the only OOS value ever added came from
giving the regime model an orthogonal macro channel (credit: spine 4f
0.44 → 5f 0.50 at a third of the null's drawdown). This race extends that
one working pattern with the channel the current features cannot see:
*rate* volatility. 2022 was bond-led equity stress — realized equity vol,
VIX, curve slope and drawdown all lag when the shock originates in rates,
and the credit spread is a level, not a turbulence measure.

Feature, fixed a priori: log EWMA vol of daily changes in the 10Y
constant-maturity yield (FRED DGS10, 1962+; forward-filled to panel
sessions, first differences, then the incumbent vol estimator unchanged —
lambda 0.94, min 60 obs, annualized). Log to match the spine's other vol
features. Causal by construction.

Challengers, both fixed a priori; references make the contribution
attributable:
1. spine 4f + rate vol (5f-rate) — does the channel carry value alone?
2. spine 4f + credit + rate vol (6f) — does it compound on the held
   credit candidate, or dilute it (the quality race's failure mode)?
References (non-gating): buy-hold, the 4f spine, the held 5f credit spine.

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of
the matched-vol placebo. A (b)-route pass is held, never shipped on this
backtest alone. Same engine, full walk-forward span, same panel as the
credit race (SPY/TLT/GLD). Changing any constant after seeing results
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
from spine.features import _ffill_series, _macro_by_date, build_features
from spine.strategy import TunedSpineStrategy
from spine.vol import ewma_vol_series

__all__ = ["rate_vol_feature", "run_race"]

SERIES = "DGS10"
GATE = {
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = ("spine 5f-rate (+ratevol)", "spine 6f (+credit+ratevol)")


def rate_vol_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """log EWMA vol of daily 10Y yield changes, aligned to panel sessions."""
    level = _ffill_series(dates, _macro_by_date(store_dir, SERIES, "value"))
    delta = np.full(len(level), np.nan)
    delta[1:] = level[1:] - level[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(ewma_vol_series(delta))


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    f5_rate = np.column_stack([features4, ratevol])
    f5_credit = np.column_stack([features4, credit])
    f6 = np.column_stack([features4, credit, ratevol])
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine tuned 4f (reference)": lambda: TunedSpineStrategy(panel, features4),
        "spine 5f-credit (held ref)": lambda: TunedSpineStrategy(panel, f5_credit),
        "spine 5f-rate (+ratevol)": lambda: TunedSpineStrategy(panel, f5_rate),
        "spine 6f (+credit+ratevol)": lambda: TunedSpineStrategy(panel, f6),
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
