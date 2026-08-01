"""Deep-history 8-K occurrence race, pre-registered.

Registered 2026-08-01, before any result was computed. The original
occurrence-flag race (2026-07-31) nominally passed but was held: its
9-name panel and 2022+ OOS were capped by the submissions endpoint's
~1,000-filing window, leaving n=344 active days and a bootstrap CI wide
enough to be luck. This race asks the same frozen question on the depth
that was always available: full per-CIK 8-K histories from SEC's
submissions archive (see fetch.py), the event study's 15-name panel
(fixed a priori there, reused unchanged), and OOS from 2004 — five
macro cycles including the GFC, roughly 5x the active days.

It is a NEW pre-registration, not an amendment: the original race keeps
its monthly re-race untouched, giving two independent looks at the same
hypothesis. Frozen mechanics inherited unchanged from that race (which
froze them from the event study): burst z >= 2 on a causal trailing
z-score (window 252, min 120), equity halved for 5 sessions, null
vol-target as the book. One deliberate difference, forced by the deep
panel and fixed a priori: per-name coverage starts at each name's first
observed 8-K, so raw counts would drift upward as names enter; the
intensity series is therefore the *rate* — daily count divided by names
covered that day — before z-scoring.

GATE is the amended §1.3 bar (outright Sharpe win, or non-inferiority
−0.05 + maxdd <= 2/3 null + matched-vol placebo); any (b)-route pass is
held, never shipped, pending re-races + behavior-match. Non-gating
diagnostics, also pre-registered: the event-study spearman signal check
and a seeded block-bootstrap 90% CI on the Sharpe difference. Changing
any constant after seeing results invalidates the race.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from eightk_race.race import BrakeVolTarget, brake_series, signal_check
from event_study.events import intensity_zscores
from event_study.study import PANEL
from harness.engine import Strategy
from harness.panel import build_panel
from harness.strategies import BuyHold, VolTarget
from quality_race.race import _amended_bar, _run

from deep_eightk.fetch import load_8k_dates
from deep_eightk.stats import sharpe_diff_bootstrap

GATE = {
    "burst_z": 2.0,
    "brake_scale": 0.5,
    "brake_days": 5,
    "z_window": 252,
    "z_min_obs": 120,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2.0 / 3.0,
}
OOS_START = "2004-01-01"
SIGNAL_HORIZON = 5


def intensity_rate(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Panel 8-K rate per trading day: count / names-covered. A filing maps
    to the first session at or after its date; a name is covered from its
    first observed 8-K onward."""
    counts: Counter[str] = Counter()
    first_covered: list[str] = []
    for symbol in PANEL:
        filed = load_8k_dates(store_dir, symbol)
        if not filed:
            raise FileNotFoundError(f"edgar_deep snapshot missing for {symbol}; run fetch")
        first_covered.append(filed[0])
        for date in filed:
            i = bisect_left(dates, date)
            if i < len(dates):
                counts[dates[i]] += 1

    coverage = np.array(
        [sum(1 for f in first_covered if f <= d) for d in dates], dtype=float
    )
    raw = np.array([float(counts.get(d, 0)) for d in dates])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(coverage > 0, raw / coverage, np.nan)
    return rate


def rate_zscores(dates: list[str], rate: np.ndarray) -> np.ndarray:
    """Causal z of the rate series, reusing the event study's convention
    (which expects a counts dict keyed by date)."""
    finite = {d: r for d, r in zip(dates, rate) if np.isfinite(r)}
    return intensity_zscores(
        dates, finite, window=int(GATE["z_window"]), min_obs=int(GATE["z_min_obs"])
    )


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, ("SPY",))
    rate = intensity_rate(store_dir, panel.dates)
    z = rate_zscores(panel.dates, rate)
    active = brake_series(z, GATE["burst_z"], int(GATE["brake_days"]))
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15% (null)": lambda: VolTarget("SPY"),
        "vol-target + deep 8-K brake": lambda: BrakeVolTarget(
            "SPY", active, GATE["brake_scale"]
        ),
    }
    runs = {
        label: _run(panel, factory, first, OOS_START)
        for label, factory in contenders.items()
    }

    null = runs["vol-target SPY 15% (null)"]
    chall = runs["vol-target + deep 8-K brake"]
    verdict = _amended_bar(
        "vol-target + deep 8-K brake", chall, null, panel, first, OOS_START
    )

    # Diagnostics (non-gating): signal spearman + bootstrap on the OOS span.
    diag_signal = signal_check(panel, z)
    date_set = {
        d for d in chall["result"]["dates"] if d >= OOS_START
    } & {d for d in null["result"]["dates"] if d >= OOS_START}
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))

    def _aligned(run: dict) -> np.ndarray:
        return np.array(
            [r for d, r in zip(run["result"]["dates"], run["result"]["returns"])
             if d in date_set]
        )

    cash = np.array([cash_by_date[d] for d in sorted(date_set)])
    diag_boot = sharpe_diff_bootstrap(_aligned(chall), _aligned(null), cash)

    oos_dates = [d for d in null["result"]["dates"] if d >= OOS_START]
    active_oos = int(
        sum(
            1
            for d, a in zip(panel.dates, active)
            if a and oos_dates and d >= OOS_START
        )
    )
    return {
        "gate": dict(GATE),
        "panel": list(PANEL),
        "oos_span": (oos_dates[0], oos_dates[-1]),
        "brake_active_days_oos": active_oos,
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": verdict,
        "diagnostics": {"signal": diag_signal, "bootstrap": diag_boot},
        "race_pass": bool(verdict["passes"]),
    }
