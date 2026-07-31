"""Score study: pre-registered gate for LLM severity as a vol-risk feature.

The event study established that 8-K filing days carry forward-vol
information the EWMA forecaster lacks. This gate asks the question the
scored panel must answer before any allocator work: does *scored severity*
rank that uplift — i.e. do LLM scores add information beyond the bare fact
that an 8-K was filed?

Test: on scored 8-K event days, the unitless ratio RV(t+1..t+h)/forecast_t
must be (A) rank-correlated with severity across all scored days, and
(B) materially higher for serious filings (severity >= 5) than routine
ones (<= 2). Both legs must pass. Direction (risk_up vs risk_down) is
reported as a diagnostic only — it does not gate.

GATE below was fixed before any result was computed. Changing it after
seeing results invalidates the study — rerun with a new pre-registration.
Join rules, also fixed a priori: a filing maps to the trading day the
market could first react (the filing date if it trades, else the next
session); when two filings land on one trading day the max-severity score
wins; only rows from the pinned scorer version are used.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
from event_pipeline.store import load_scores
from event_study.stats import mann_whitney_u, spearman
from event_study.study import PANEL, _symbol_series, _vol_ratios

HORIZON_DAYS = 5
SECONDARY_HORIZON = 21
SCORER_VERSION = "qwen3.6:35b-a3b@c021ead6a939"

GATE = {
    "horizon_days": HORIZON_DAYS,
    # Leg A: severity must rank event-day log vol ratios.
    "min_spearman": 0.10,
    "min_scored_days": 500,
    # Leg B: serious vs routine bands must separate in median ratio.
    "serious_min_severity": 5,
    "routine_max_severity": 2,
    "min_median_uplift": 0.10,
    "min_events_per_band": 50,
    "max_p_value": 0.01,
}


def scored_event_days(dates: list[str], scores: list[dict]) -> dict[int, dict]:
    """Map scored filings to trading-day indices; max severity wins ties."""
    out: dict[int, dict] = {}
    for row in scores:
        i = bisect_left(dates, row["filing_date"])
        if i >= len(dates):
            continue
        cur = out.get(i)
        if cur is None or row["severity"] > cur["severity"]:
            out[i] = {"severity": row["severity"], "direction": row["direction"]}
    return out


def panel_events(
    store_dir: Path, scores_path: Path, horizon: int
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Pooled (severity, direction, vol ratio) across the fixed panel."""
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in load_scores(scores_path, SCORER_VERSION):
        by_symbol[row["symbol"]].append(row)
    severities: list[int] = []
    directions: list[str] = []
    ratios: list[float] = []
    for symbol in PANEL:
        rows = by_symbol.get(symbol)
        if not rows:
            continue
        dates, returns = _symbol_series(store_dir, symbol)
        vol_ratio = _vol_ratios(returns, horizon)
        for i, info in scored_event_days(dates, rows).items():
            if np.isfinite(vol_ratio[i]):
                severities.append(int(info["severity"]))
                directions.append(str(info["direction"]))
                ratios.append(float(vol_ratio[i]))
    return np.array(severities), directions, np.array(ratios)


def leg_rank(severities: np.ndarray, ratios: np.ndarray) -> dict:
    rho, p = spearman(severities.astype(float), np.log(ratios))
    return {"n_days": len(severities), "spearman": rho, "p_value": p}


def leg_bands(severities: np.ndarray, ratios: np.ndarray) -> dict | None:
    serious = ratios[severities >= GATE["serious_min_severity"]]
    routine = ratios[severities <= GATE["routine_max_severity"]]
    if min(len(serious), len(routine)) < GATE["min_events_per_band"]:
        return None
    effect, p = mann_whitney_u(serious, routine)
    return {
        "n_serious": len(serious),
        "n_routine": len(routine),
        "median_uplift": float(np.median(serious) / np.median(routine) - 1.0),
        "effect_size": effect,
        "p_value": p,
    }


def diagnostic_direction(
    severities: np.ndarray, directions: list[str], ratios: np.ndarray
) -> dict | None:
    """Non-gating: among severity >= 3 days, does risk_up out-rank risk_down?"""
    mask = severities >= 3
    up = ratios[mask & (np.array(directions) == "risk_up")]
    down = ratios[mask & (np.array(directions) == "risk_down")]
    if min(len(up), len(down)) < 10:
        return None
    effect, p = mann_whitney_u(up, down)
    return {
        "n_risk_up": len(up),
        "n_risk_down": len(down),
        "median_up": float(np.median(up)),
        "median_down": float(np.median(down)),
        "effect_size": effect,
        "p_value": p,
    }


def run_study(store_dir: Path, scores_path: Path) -> dict:
    report: dict = {
        "gate": dict(GATE),
        "scorer_version": SCORER_VERSION,
        "horizons": {},
    }
    for horizon in (HORIZON_DAYS, SECONDARY_HORIZON):
        severities, directions, ratios = panel_events(store_dir, scores_path, horizon)
        report["horizons"][horizon] = {
            "rank": leg_rank(severities, ratios),
            "bands": leg_bands(severities, ratios),
            "direction": diagnostic_direction(severities, directions, ratios),
        }
    primary = report["horizons"][HORIZON_DAYS]
    rank = primary["rank"]
    rank_pass = (
        rank["n_days"] >= GATE["min_scored_days"]
        and rank["spearman"] >= GATE["min_spearman"]
        and rank["p_value"] <= GATE["max_p_value"]
    )
    bands = primary["bands"]
    bands_pass = (
        bands is not None
        and bands["median_uplift"] >= GATE["min_median_uplift"]
        and bands["p_value"] <= GATE["max_p_value"]
    )
    report["verdict"] = {
        "rank_pass": rank_pass,
        "bands_pass": bands_pass,
        "overall_pass": rank_pass and bands_pass,
    }
    return report
