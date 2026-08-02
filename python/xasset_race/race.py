"""Cross-asset race: global (non-US) equity stress as a regime feature,
pre-registered before any result was computed.

Motivation, from the accumulated record: every proven channel is a *US*
measure — SPY price and volume, US credit, US rates — yet several of the
worst equity regimes were foreign-led or global before they were American
(1997 Asia, 1998 Russia/LTCM, 2010-12 eurozone, 2015 China devaluation).
The record's one working pattern is handing the regime model an orthogonal
stress channel with a named economic meaning; foreign-market turbulence is
the named quantity no current feature sees. Both learned-race postmortems
concluded the next axis is new *data*, not new transformations — this is
the first race on a data channel outside the US panel: five 1996-vintage
iShares country ETFs (EWJ, EWG, EWU, EWH, EWA — Japan, Germany, UK, Hong
Kong, Australia; fetched 2026-08-01 with the standard adjClose cross-check,
zero mismatches, all with full history to 1996-04). Distinct from closed
channels: the corr race closed the stock-bond co-movement *sign*; the
dispersion race closed *sector* localization within the US index. This is
cross-border stress *level* — a different quantity on new data. Registered
risk, honestly: the dispersion lesson (the 3rd state may already extract
whatever this channel carries from index-level features) applies here too;
that is exactly what the race adjudicates.

Feature, fixed a priori: per-ETF daily adjClose return; per-ETF log EWMA
vol (the incumbent estimator unchanged — lambda 0.94, min 60 obs,
annualized); cross-ETF mean on sessions where >= 3 of the 5 are defined;
then causal z (window 252, min obs 120 — the event-study convention),
zero-filled where undefined so pre-1996 walk-forward windows stay fittable
(the salience/dispersion pattern for late-inception channels). Causal by
construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + xasset (5f-xa) — does the channel carry value alone?
2. spine 6f + xasset (7f-xa) — does it compound on the macro channels, or
   dilute (the quality-race failure mode)?
3. spine 6f + turnover + xasset with the validated 3-state regime model
   (8f-turn-xa 3-state) — the compounding test on the leading candidate
   itself (n=3 inherited from the nstate race, not scanned).
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
from harness.signals import causal_zscores
from harness.strategies import BuyHold, VolTarget
from market_data.store import read_snapshot
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from spine.vol import ewma_vol_series
from turnover_race.race import turnover_feature

__all__ = ["run_race", "xasset_feature"]

COUNTRY_ETFS = ("EWJ", "EWG", "EWU", "EWH", "EWA")
GATE = {
    "min_etfs": 3,
    "z_window": 252,
    "z_min_obs": 120,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-xa (+xasset)",
    "spine 7f-xa (6f +xasset)",
    "spine 8f-turn-xa 3-state",
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


def xasset_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Causal z of mean country-ETF log EWMA vol; 0.0 where undefined
    (pre-coverage and z warm-up)."""
    n = len(dates)
    log_vols = np.full((n, len(COUNTRY_ETFS)), np.nan)
    for j, symbol in enumerate(COUNTRY_ETFS):
        by_date = _adj_close_by_date(store_dir, symbol)
        prices = np.array([by_date.get(d, np.nan) for d in dates])
        returns = np.full(n, np.nan)
        with np.errstate(invalid="ignore"):
            returns[1:] = prices[1:] / prices[:-1] - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            log_vols[:, j] = np.log(ewma_vol_series(returns))

    mean_vol = np.full(n, np.nan)
    enough = np.isfinite(log_vols).sum(axis=1) >= int(GATE["min_etfs"])
    for t in np.flatnonzero(enough):
        row = log_vols[t][np.isfinite(log_vols[t])]
        mean_vol[t] = row.mean()

    z = causal_zscores(mean_vol, int(GATE["z_window"]), int(GATE["z_min_obs"]))
    return np.where(np.isfinite(z), z, 0.0)


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    xa = xasset_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_xa = np.column_stack([features4, xa])
    f7_xa = np.column_stack([f6, xa])
    f7_turn = np.column_stack([f6, turn])
    f8 = np.column_stack([f6, turn, xa])
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
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_xa),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_xa),
        CHALLENGERS[2]: lambda: TunedSpineStrategy(panel, f8, n_states=n3),
    }
    runs = race_contenders(panel, contenders, first)
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "country_etfs": COUNTRY_ETFS,
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
