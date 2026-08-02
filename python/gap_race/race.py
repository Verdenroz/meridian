"""Overnight-gap race: gap turbulence as a regime feature, pre-registered
before any result was computed.

Motivation, from the accumulated record: the working pattern is handing the
regime model an orthogonal stress channel (credit level, rate turbulence,
dollar-volume participation — each compounds because their episodes are
disjoint). The chart store's open field has never been touched by any race,
and it carries a channel the close-to-close features integrate away: how
much of volatility arrives *overnight*. Gap-dominated vol is a distinct
stress signature — news lands while the market cannot trade (2020-03 limit
moves, macro releases, weekend risk) — versus intraday churn at the same
close-close vol. SPY opens verify clean over the full 1993→present span
(zero missing, max |gap| 11% on 2020-03-16, the COVID limit-down open).

Feature, fixed a priori: log EWMA vol of overnight log-gaps
ln(open_t / close_{t-1}) on raw prices (the incumbent estimator unchanged —
lambda 0.94, min 60 obs, annualized; the rate-race pattern). Registered
limitation: raw prices put quarterly ex-dividend opens (~0.3%) into the
gaps; the EWMA integrates over them and they are an order below gap vol.
Causal by construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + gap (5f-gap) — does the channel carry value alone?
2. spine 6f + gap (7f-gap) — does it compound on the proven channels?
3. 8f-turn-gap 3-state (6f + turnover + gap, n=3) — the compounding test
   on the leading candidate (the octet race showed stacks can interfere;
   that is the question, not an assumption).
References (non-gating): buy-hold, the untouched null, spine 6f 2-state,
spine 6f 3-state, spine 7f-turn 3-state (the leading candidate).

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of the
matched-vol placebo. Full walk-forward span, same engine, same panel
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
from market_data.store import read_snapshot
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from spine.vol import ewma_vol_series
from turnover_race.race import turnover_feature

__all__ = ["gap_vol_feature", "run_race"]

GATE = {
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
MARKET_SYMBOL = "SPY"
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-gap (+gapvol)",
    "spine 7f-gap (6f +gapvol)",
    "spine 8f-turn-gap 3-state",
)


def gap_vol_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """log EWMA vol of SPY overnight log-gaps, aligned to panel sessions."""
    doc = read_snapshot(store_dir, "charts", MARKET_SYMBOL)
    if doc is None:
        raise FileNotFoundError("SPY chart snapshot missing; run market_data fetch")
    opens = {r["date"]: float(r["open"]) for r in doc["data"]["rows"] if r.get("open")}
    closes = {
        r["date"]: float(r["close"]) for r in doc["data"]["rows"] if r.get("close")
    }
    o = np.array([opens.get(d, np.nan) for d in dates])
    c = np.array([closes.get(d, np.nan) for d in dates])
    gaps = np.full(len(dates), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps[1:] = np.log(o[1:] / c[:-1])
        return np.log(ewma_vol_series(gaps))


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    gap = gap_vol_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_gap = np.column_stack([features4, gap])
    f7_gap = np.column_stack([f6, gap])
    f7_turn = np.column_stack([f6, turn])
    f8 = np.column_stack([f6, turn, gap])
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
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_gap),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_gap),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f8, n_states=n3),
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
