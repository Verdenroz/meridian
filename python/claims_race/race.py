"""Initial-claims race: real-economy deterioration as a regime feature,
pre-registered before any result was computed.

Motivation, from the accumulated record: the oil/FX verdicts (2026-08-02)
closed the market-price turbulence axis — five races agree the 3-state
leader's stress triangulation is saturated for channels derived from
market prices. Every feature the spine has ever read is a market price or
text about markets; none is a measured *quantity from the real economy*.
Weekly initial unemployment claims are the classic high-frequency labor
signal — recessions (the one stress kind every channel sees only through
its market echo) show up as a sustained rise in claims, sometimes before
credit or vol react (1990, 2001, 2007-08). FRED ICNSA, 1967+, weekly,
untouched by any race — a genuinely different data *kind*, per the
cross-race note, not another turbulence cousin.

Point-in-time honesty, frozen a priori: the NSA series (ICNSA) is used so
retrospectively re-estimated seasonal factors never leak into history;
seasonality is removed by construction via a 52-week year-over-year log
change. Each weekly observation (week ending Saturday) becomes available
5 calendar days later (the following Thursday's 8:30 ET release); the
feature at session t reads the latest observation available on or before
t. Registered limitation, honestly: upstream serves final vintages (no
ALFRED), so post-release revisions to claims counts are in the backtest
history; initial-claims revisions are small and the yoy log transform
damps them further — recorded as a known limitation, like survivorship.

Feature, fixed a priori: log(claims_w / claims_{w-52}) on the weekly NSA
series (52-week ratio kills seasonality; positive = labor-market
deterioration), lagged to availability as above, forward-filled across
sessions between releases. Causal by construction.

Challengers, all fixed a priori; references make the contribution
attributable:
1. spine 4f + claims (5f-clm) — does the channel carry value alone?
2. spine 6f + claims (7f-clm) — does it compound on the macro channels,
   or dilute (the quality-race failure mode)?
3. spine 6f + turnover + claims with the validated 3-state regime model
   (8f-turn-clm 3-state) — the compounding test on the leading candidate
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
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.features import _macro_by_date, build_features
from spine.strategy import TunedSpineStrategy
from turnover_race.race import turnover_feature

__all__ = ["claims_feature", "run_race"]

SERIES = "ICNSA"
GATE = {
    "yoy_weeks": 52,
    "publication_lag_days": 5,
    "n_states_stack": 3,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = (
    "spine 5f-clm (+claims)",
    "spine 7f-clm (6f +claims)",
    "spine 8f-turn-clm 3-state",
)


def claims_feature(store_dir: Path, dates: list[str]) -> np.ndarray:
    """52-week yoy log change of NSA initial claims, lagged to the Thursday
    release; session t reads the latest observation available on or before t."""
    weekly = sorted(_macro_by_date(store_dir, SERIES, "value").items())
    lag = timedelta(days=int(GATE["publication_lag_days"]))
    k = int(GATE["yoy_weeks"])
    available: list[tuple[str, float]] = []
    for i in range(k, len(weekly)):
        obs_date, value = weekly[i]
        prior = weekly[i - k][1]
        if value <= 0 or prior <= 0:
            continue
        avail = (date.fromisoformat(obs_date) + lag).isoformat()
        available.append((avail, float(np.log(value / prior))))

    out = np.full(len(dates), np.nan)
    last = np.nan
    j = 0
    for t, session in enumerate(dates):
        while j < len(available) and available[j][0] <= session:
            last = available[j][1]
            j += 1
        out[t] = last
    return out


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    credit = credit_spread_series(store_dir, panel.dates)
    ratevol = rate_vol_feature(store_dir, panel.dates)
    turn = turnover_feature(store_dir, panel.dates)
    clm = claims_feature(store_dir, panel.dates)
    f6 = np.column_stack([features4, credit, ratevol])
    f5_clm = np.column_stack([features4, clm])
    f7_clm = np.column_stack([f6, clm])
    f7_turn = np.column_stack([f6, turn])
    f8 = np.column_stack([f6, turn, clm])
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
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f5_clm),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(panel, f7_clm),
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
