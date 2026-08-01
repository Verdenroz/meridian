"""Stock-bond correlation race: the co-movement regime as a feature,
pre-registered before any result was computed.

The charter's mandate is volatility, correlation, and regime — and
correlation is the one leg no race has touched. The macro state the
current features cannot see directly is the *sign regime* of stock-bond
co-movement: growth-driven markets (bonds hedge equities) vs
inflation-driven markets (they fall together, 2022). Rate vol (the first
outright route-(a) pass) measures bond turbulence but not co-movement;
the credit spread is a level. The episodic-memory state vector already
carries a correlation summary — explanation-only; this race is the
gate-honest path into the allocator.

Feature, fixed a priori: rolling 60-session correlation of SPY daily
returns with the *negated* daily DGS10 yield change — a bond-return
proxy, so the series reads as stock-bond return correlation (negative =
bonds currently hedge). 60 sessions is the spine's standing safe-asset
correlation window, inherited, not tuned; DGS10 keeps the history deep
(TLT's 2002 inception would truncate the panel). Causal by construction
(`rolling_correlation`).

Challengers, both fixed a priori; references make the contribution
attributable:
1. spine 4f + corr (5f-corr) — does the channel carry value alone?
2. spine 4f + credit + ratevol + corr (7f) — does it compound on the
   leading 6f candidate, or dilute it (the quality race's failure mode)?
References (non-gating): buy-hold, the untouched null, the 6f leader.

Gate: each challenger clears the charter's amended bar (§1.3) against
the untouched vol-target null — outright after-cost Sharpe win, or held
route (b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's +
beat of the matched-vol placebo. Same engine, full walk-forward span,
same panel as the credit race (SPY/TLT/GLD). Changing any constant after
seeing results invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import Panel, build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.correlation import rolling_correlation
from spine.features import _ffill_series, _macro_by_date, build_features
from spine.strategy import TunedSpineStrategy

__all__ = ["run_race", "stock_bond_corr"]

GATE = {
    "corr_window": 60,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = ("spine 5f-corr (+sbcorr)", "spine 7f (6f +sbcorr)")


def stock_bond_corr(panel: Panel, store_dir: Path) -> np.ndarray:
    """Rolling correlation of SPY returns with the negated 10Y yield
    change — stock-bond return correlation on a deep-history proxy."""
    level = _ffill_series(panel.dates, _macro_by_date(store_dir, "DGS10", "value"))
    bond_proxy = np.full(len(level), np.nan)
    bond_proxy[1:] = -(level[1:] - level[:-1])
    spy = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
    return rolling_correlation(spy, bond_proxy, int(GATE["corr_window"]))


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    corr = stock_bond_corr(panel, store_dir)
    f5_corr = np.column_stack([features4, corr])
    f6 = np.column_stack([features4, credit, ratevol])
    f7 = np.column_stack([features4, credit, ratevol, corr])
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f (leader ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 5f-corr (+sbcorr)": lambda: TunedSpineStrategy(panel, f5_corr),
        "spine 7f (6f +sbcorr)": lambda: TunedSpineStrategy(panel, f7),
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
