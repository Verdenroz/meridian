"""The event study proper: pre-registered gate, three legs, one verdict.

Test: on event days (filings; news bursts), is forward realized vol elevated
relative to the standing EWMA forecast — i.e. do events carry vol information
the incumbent forecaster does not already have? Ratios RV(t+1..t+h)/forecast_t
are unitless, so they pool across symbols.

GATE below was fixed before any result was computed. Changing it after seeing
results invalidates the study — rerun with a new pre-registration instead.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
from market_data.client import DEFAULT_BASE_URL
from market_data.pipeline import ensure_symbol
from market_data.store import read_snapshot
from spine.vol import ewma_vol_series, forward_realized_vol

from event_study.events import (
    STUDY_FORMS,
    fetch_filings,
    intensity_zscores,
    load_filings,
    market_daily_counts,
    news_daily_counts,
)
from event_study.stats import mann_whitney_u, spearman

# Primary horizon: one trading week of forward realized vol.
HORIZON_DAYS = 5

GATE = {
    "horizon_days": HORIZON_DAYS,
    # An event class passes if event-day ratios beat baseline by both:
    "min_median_uplift": 0.05,
    "max_p_value": 0.01,
    # Market-level news intensity passes on rank correlation with residual vol:
    "market_min_spearman": 0.10,
    # A per-name news "burst" day: count z-score at or above this.
    "news_burst_z": 2.0,
    "min_events": 100,
}

# Large caps with deep chart + filing + news coverage. Fixed a priori.
PANEL = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ",
    "XOM", "WMT", "PG", "KO", "UNH", "HD", "CVX",
)
SECONDARY_HORIZON = 21


def _symbol_series(store_dir: Path, symbol: str) -> tuple[list[str], np.ndarray]:
    doc = read_snapshot(store_dir, "charts", symbol)
    if doc is None:
        raise FileNotFoundError(f"no chart snapshot for {symbol}")
    rows = doc["data"]["rows"]
    dates = [r["date"] for r in rows]
    adj = np.array([np.nan if r["adj_close"] is None else r["adj_close"] for r in rows])
    returns = np.full(len(adj), np.nan)
    returns[1:] = adj[1:] / adj[:-1] - 1.0
    return dates, returns


def _vol_ratios(returns: np.ndarray, horizon: int) -> np.ndarray:
    forecast = ewma_vol_series(returns)
    forward = forward_realized_vol(returns, horizon)
    with np.errstate(invalid="ignore"):
        return forward / forecast


def _event_indices(dates: list[str], event_dates: list[str]) -> set[int]:
    """Map each event date to the trading day it becomes known (that date if
    it trades, else the next trading day)."""
    out: set[int] = set()
    for date in event_dates:
        i = bisect_left(dates, date)
        if i < len(dates):
            out.add(i)
    return out


def _uplift(event_ratios: np.ndarray, baseline_ratios: np.ndarray) -> dict:
    effect, p = mann_whitney_u(event_ratios, baseline_ratios)
    baseline_median = float(np.median(baseline_ratios))
    return {
        "n_events": len(event_ratios),
        "n_baseline": len(baseline_ratios),
        "median_uplift": float(np.median(event_ratios) / baseline_median - 1.0),
        "effect_size": effect,
        "p_value": p,
    }


def _pool_event_class(
    store_dir: Path,
    per_symbol_events: dict[str, set[int]],
    horizon: int,
) -> dict | None:
    events: list[float] = []
    baseline: list[float] = []
    for symbol, idx in per_symbol_events.items():
        dates, returns = _symbol_series(store_dir, symbol)
        ratios = _vol_ratios(returns, horizon)
        for t in range(len(dates)):
            if not np.isfinite(ratios[t]):
                continue
            (events if t in idx else baseline).append(float(ratios[t]))
    if len(events) < GATE["min_events"] or len(baseline) < GATE["min_events"]:
        return None
    return _uplift(np.array(events), np.array(baseline))


def leg_filings(store_dir: Path, horizon: int) -> dict[str, dict | None]:
    """Per form class: pooled event-day vol ratios vs non-filing days."""
    results: dict[str, dict | None] = {}
    for form in STUDY_FORMS:
        per_symbol: dict[str, set[int]] = {}
        for symbol in PANEL:
            filings = load_filings(store_dir, symbol)
            if not filings:
                continue
            dates, _ = _symbol_series(store_dir, symbol)
            form_dates = [f["date"] for f in filings if f["form"] == form]
            per_symbol[symbol] = _event_indices(dates, form_dates)
        results[form] = _pool_event_class(store_dir, per_symbol, horizon)
    return results


def leg_news_bursts(store_dir: Path, horizon: int) -> dict | None:
    """Per-name burst days (count z >= threshold) vs ordinary covered days."""
    events: list[float] = []
    baseline: list[float] = []
    for symbol in PANEL:
        counts = news_daily_counts(symbol)
        if not counts:
            continue
        dates, returns = _symbol_series(store_dir, symbol)
        z = intensity_zscores(dates, counts)
        ratios = _vol_ratios(returns, horizon)
        covered = np.isfinite(z) & np.isfinite(ratios)
        for t in np.flatnonzero(covered):
            target = events if z[t] >= GATE["news_burst_z"] else baseline
            target.append(float(ratios[t]))
    if len(events) < GATE["min_events"] or len(baseline) < GATE["min_events"]:
        return None
    return _uplift(np.array(events), np.array(baseline))


def leg_market_intensity(store_dir: Path, horizon: int) -> dict:
    """Does aggregate news intensity rank-predict SPY's residual forward vol?"""
    dates, returns = _symbol_series(store_dir, "SPY")
    z = intensity_zscores(dates, market_daily_counts())
    ratios = _vol_ratios(returns, horizon)
    mask = np.isfinite(z) & np.isfinite(ratios) & (ratios > 0)
    rho, p = spearman(z[mask], np.log(ratios[mask]))
    return {"n_days": int(mask.sum()), "spearman": rho, "p_value": p}


def _passes(result: dict | None) -> bool:
    return (
        result is not None
        and result["median_uplift"] >= GATE["min_median_uplift"]
        and result["p_value"] <= GATE["max_p_value"]
    )


def run_study(
    store_dir: Path, base_url: str = DEFAULT_BASE_URL, refresh: bool = False
) -> dict:
    for symbol in PANEL:
        ensure_symbol(base_url, store_dir, symbol)
        if refresh or load_filings(store_dir, symbol) is None:
            fetch_filings(base_url, store_dir, symbol)

    report: dict = {"gate": dict(GATE), "horizons": {}}
    for horizon in (HORIZON_DAYS, SECONDARY_HORIZON):
        report["horizons"][horizon] = {
            "filings": leg_filings(store_dir, horizon),
            "news_bursts": leg_news_bursts(store_dir, horizon),
            "market_intensity": leg_market_intensity(store_dir, horizon),
        }

    primary = report["horizons"][HORIZON_DAYS]
    filings_pass = any(_passes(r) for r in primary["filings"].values())
    bursts_pass = _passes(primary["news_bursts"])
    market = primary["market_intensity"]
    market_pass = (
        market["spearman"] >= GATE["market_min_spearman"]
        and market["p_value"] <= GATE["max_p_value"]
    )
    report["verdict"] = {
        "filings_pass": filings_pass,
        "news_bursts_pass": bursts_pass,
        "market_intensity_pass": market_pass,
        "overall_pass": filings_pass or bursts_pass or market_pass,
    }
    return report
