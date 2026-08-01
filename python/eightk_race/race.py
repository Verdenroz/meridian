"""Occurrence-flag race: does an 8-K filing-intensity brake beat the null?

The event study established that 8-K *occurrence* (not LLM severity — that
gate failed) precedes per-name forward-vol elevation. This race tests the
only deployment the ETF core allows: panel-aggregate 8-K intensity as a
market-level defense. Challenger = the null vol-target with a brake that
halves equity exposure for the 5 sessions after a filing-intensity burst
(causal trailing z >= 2). It ships only if it beats the untouched null.

GATE and all constants below were fixed before any result was computed;
changing them after seeing results invalidates the race. burst_z and
brake_days reuse the event study's pre-registered burst threshold and
primary horizon; brake_scale is a single a-priori choice, not tuned.

The intensity panel is availability-driven, not outcome-driven: the EDGAR
submissions window reaches back ~1000 filings per company, so per-name 8-K
history starts anywhere from 2015 to 2026. The panel is the 9 study names
whose coverage reaches back to 2021-01-06 or earlier; the OOS comparison
span starts 2022-01-01 so the trailing z-score is warm and every panel
name is covered throughout. Known caveats: ~4.6y OOS is one macro cycle,
and per-name coverage before each name's window start is invisible.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from event_study.events import intensity_zscores, load_filings
from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.signals import BrakeVolTarget, brake_series
from harness.stats import spearman
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from spine.vol import ewma_vol_series, forward_realized_vol

__all__ = ["BrakeVolTarget", "brake_series", "run_race"]

# Study names with 8-K coverage since <= 2021-01-06 (availability-driven).
INTENSITY_PANEL = (
    "AAPL", "MSFT", "AMZN", "NVDA", "JNJ", "KO", "HD", "CVX", "UNH",
)

GATE = {
    "burst_z": 2.0,
    "brake_scale": 0.5,
    "brake_days": 5,
    # Diagnostic only (explains the race outcome, does not gate):
    "signal_min_spearman": 0.10,
    "max_p_value": 0.01,
}
OOS_START = "2022-01-01"
SIGNAL_HORIZON = 5


def eightk_daily_counts(store_dir: Path, dates: list[str]) -> dict[str, int]:
    """Trading-day 8-K counts across the intensity panel; a filing maps to
    the session the market could first react (its date, else the next)."""
    counts: Counter[str] = Counter()
    for symbol in INTENSITY_PANEL:
        for filing in load_filings(store_dir, symbol) or []:
            if filing["form"] != "8-K":
                continue
            i = bisect_left(dates, filing["date"])
            if i < len(dates):
                counts[dates[i]] += 1
    return dict(counts)


def _oos_metrics(panel: Panel, result: dict, oos_start: str) -> dict:
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    kept = [
        (r, cash_by_date[d])
        for d, r in zip(result["dates"], result["returns"])
        if d >= oos_start
    ]
    returns, cash = ([x[i] for x in kept] for i in (0, 1))
    return compute_metrics(returns, cash)


def signal_check(panel: Panel, z: np.ndarray) -> dict:
    """Diagnostic: does panel 8-K intensity rank SPY's residual forward vol?"""
    returns = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
    with np.errstate(invalid="ignore"):
        ratio = forward_realized_vol(returns, SIGNAL_HORIZON) / ewma_vol_series(returns)
    mask = np.isfinite(z) & np.isfinite(ratio) & (ratio > 0)
    rho, p = spearman(z[mask], np.log(ratio[mask]))
    return {"n_days": int(mask.sum()), "spearman": rho, "p_value": p}


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, ("SPY",))
    z = intensity_zscores(panel.dates, eightk_daily_counts(store_dir, panel.dates))
    active = brake_series(z, GATE["burst_z"], int(GATE["brake_days"]))
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15% (null)": lambda: VolTarget("SPY"),
        "vol-target + 8-K brake": lambda: BrakeVolTarget(
            "SPY", active, GATE["brake_scale"]
        ),
    }
    metrics = {
        label: _oos_metrics(panel, run_walk_forward(panel, factory, first), OOS_START)
        for label, factory in contenders.items()
    }
    challenger = metrics["vol-target + 8-K brake"]["sharpe"]
    null = metrics["vol-target SPY 15% (null)"]["sharpe"]
    signal = signal_check(panel, z)
    return {
        "gate": dict(GATE),
        "oos_start": OOS_START,
        "brake_active_days": int(active.sum()),
        "n_days": panel.n_days,
        "metrics": metrics,
        "signal": signal,
        "verdict": {
            "challenger_sharpe": challenger,
            "null_sharpe": null,
            "race_pass": challenger > null,
            "signal_pass": (
                signal["spearman"] >= GATE["signal_min_spearman"]
                and signal["p_value"] <= GATE["max_p_value"]
            ),
        },
    }
