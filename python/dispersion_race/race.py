"""Sector-dispersion race: cross-sectional stress localization as a regime
feature, pre-registered before any result was computed.

Motivation, from the accumulated record: the 3-state pass (nstate_race) was
derived from the finding that stress comes in *kinds* — the proven channels
fire on largely disjoint day sets — and its mechanism is a middle state that
keeps the book invested through mild/localized stress. Every current feature
is an index-level or macro *level* measure; none can distinguish a broad
selloff from a rotational/localized one (2000–02 tech, 2015–16 energy, 2023
regional banks) at the same index vol. Cross-sectional dispersion of sector
returns measures exactly that localization, from a data channel (the nine
1998-vintage SPDR sector ETFs, fetched 2026-08-01 with the standard adjClose
cross-check) no prior race has touched.

Feature, fixed a priori: daily cross-sectional std (population) of the nine
sector ETFs' adjClose returns on sessions where >= 5 sectors have a return;
smoothed by the incumbent EWMA vol estimator unchanged (lambda 0.94, min 60
obs — the rate-race pattern); log; then causal z (window 252, min obs 120,
the event-study convention) zero-filled where undefined so pre-1999
walk-forward windows stay fittable (the salience-race pattern). Causal by
construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + dispersion (5f-disp) — does the channel carry value alone?
2. spine 6f + dispersion (7f-disp) — does it compound on the leader's
   features, or dilute (the quality-race failure mode)?
3. 7f-disp with the validated 3-state regime model — the compounding test
   on the leading candidate itself (the champion-race axes-additivity
   pattern; n=3 is the validated granularity, not a scan).
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
from spine.vol import ewma_vol_series

__all__ = ["dispersion_feature", "run_race"]

SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
GATE = {
    "min_sectors": 5,
    "z_window": 252,
    "z_min_obs": 120,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-disp (+dispersion)",
    "spine 7f-disp (6f +dispersion)",
    "spine 7f-disp 3-state",
)


def _adj_close_by_date(store_dir: Path, symbol: str) -> dict[str, float]:
    doc = read_snapshot(store_dir, "charts", symbol)
    if doc is None:
        raise FileNotFoundError(f"chart snapshot {symbol} missing; run market_data fetch")
    return {
        row["date"]: float(row["adj_close"])
        for row in doc["data"]["rows"]
        if row.get("adj_close") is not None
    }


def dispersion_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Causal z of log EWMA-smoothed sector-return dispersion; 0.0 where
    undefined (pre-coverage and z warm-up)."""
    returns = np.full((len(dates), len(SECTORS)), np.nan)
    for j, symbol in enumerate(SECTORS):
        by_date = _adj_close_by_date(store_dir, symbol)
        prices = np.array([by_date.get(d, np.nan) for d in dates])
        with np.errstate(invalid="ignore"):
            returns[1:, j] = prices[1:] / prices[:-1] - 1.0

    spread = np.full(len(dates), np.nan)
    counts = np.isfinite(returns).sum(axis=1)
    enough = counts >= int(GATE["min_sectors"])
    for t in np.flatnonzero(enough):
        row = returns[t][np.isfinite(returns[t])]
        spread[t] = row.std()

    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.log(ewma_vol_series(spread))
    z = causal_zscores(smooth, int(GATE["z_window"]), int(GATE["z_min_obs"]))
    return np.where(np.isfinite(z), z, 0.0)


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    disp = dispersion_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_disp = np.column_stack([features4, disp])
    f7 = np.column_stack([f6, disp])
    first = panel.first_index_with_data(("SPY",))
    n3 = int(GATE["n_states_stack"])

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 2-state (ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 6f 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f6, n_states=n3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_disp),
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
