"""Turnover-intensity race: trading-volume participation as a regime
feature, pre-registered before any result was computed.

Motivation, from the accumulated record: the only pattern that has ever
added OOS value is handing the regime model an orthogonal stress channel
(credit level, rate turbulence — and the two compound because their episodes
are disjoint). Every current feature is a *price or macro* measure; none
sees participation. Volume intensity is the classic microstructure stress
marker — capitulation days trade multiples of normal dollar volume at the
same realized vol, while low-participation grinds do not — and the raw data
(SPY OHLCV, 1993→present) has been in the store since day one, untouched by
any race. The event study's negative volume result concerned *news counts*;
traded dollar volume is a different quantity.

Feature, fixed a priori: log of SPY daily dollar volume (close x volume,
sessions with volume > 0), causal z (window 252, min obs 120 — the
event-study convention; the trailing z also absorbs the series' strong
secular growth), zero-filled where undefined so early walk-forward windows
stay fittable (the salience-race pattern). Causal by construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + turnover (5f-turn) — does the channel carry value alone?
2. spine 6f + turnover (7f-turn) — does it compound on the leader's
   features, or dilute (the quality-race failure mode)?
3. 7f-turn with the validated 3-state regime model — the compounding test
   on the leading candidate itself (n=3 inherited from the nstate race, not
   scanned).
References (non-gating): buy-hold, the untouched null, spine 6f 2-state,
spine 6f 3-state (the leading candidate).

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
from harness.signals import causal_zscores
from harness.strategies import BuyHold, VolTarget
from market_data.store import read_snapshot
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy

__all__ = ["run_race", "turnover_feature"]

GATE = {
    "z_window": 252,
    "z_min_obs": 120,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
MARKET_SYMBOL = "SPY"
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-turn (+turnover)",
    "spine 7f-turn (6f +turnover)",
    "spine 7f-turn 3-state",
)


def turnover_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Causal z of log SPY dollar volume; 0.0 where undefined (warm-up)."""
    doc = read_snapshot(store_dir, "charts", MARKET_SYMBOL)
    if doc is None:
        raise FileNotFoundError("SPY chart snapshot missing; run market_data fetch")
    dollar = {
        row["date"]: float(row["close"]) * float(row["volume"])
        for row in doc["data"]["rows"]
        if row.get("volume") and row.get("close") is not None
    }
    raw = np.array([dollar.get(d, np.nan) for d in dates])
    with np.errstate(invalid="ignore", divide="ignore"):
        log_dollar = np.log(raw)
    z = causal_zscores(log_dollar, int(GATE["z_window"]), int(GATE["z_min_obs"]))
    return np.where(np.isfinite(z), z, 0.0)


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_turn = np.column_stack([features4, turn])
    f7 = np.column_stack([f6, turn])
    first = panel.first_index_with_data(("SPY",))
    n3 = int(GATE["n_states_stack"])

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 2-state (ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 6f 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f6, n_states=n3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_turn),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f7, n_states=n3),
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
