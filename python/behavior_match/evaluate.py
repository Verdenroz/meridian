"""Behavior-match evaluation: apply the frozen criteria to the shadow log.

Refuses to run before the pre-registered opening date. Produces one
verdict per strategy; PASS requires C1 (coverage), C2 (fidelity), and
C3 (behavior in-distribution) together.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from shadow.log import load_decisions

from behavior_match import stats
from behavior_match.criteria import CRITERIA
from behavior_match.replay import (
    build_race_panel,
    reference_rows,
    retro_weights,
    trading_days_between,
)

STRATEGIES = ("null_voltarget", "credit_brake", "spine5f_credit")


class EvaluationNotOpen(RuntimeError):
    """Raised when evaluation is attempted before the pre-registered date."""


def check_evaluation_date(today: date) -> None:
    earliest = date.fromisoformat(str(CRITERIA["earliest_evaluation"]))
    if today < earliest:
        raise EvaluationNotOpen(
            f"behavior-match evaluation opens {earliest}; today is {today} — "
            "the pre-registration forbids peeking at partial windows"
        )


def window_stats(name: str, rows: list[dict]) -> dict[str, float]:
    out = {
        "mean_gross_exposure": stats.mean_gross_exposure(rows),
        "turnover_annualized": stats.turnover_annualized(rows),
    }
    if name == "credit_brake":
        out["brake_active_frac"] = stats.brake_active_frac(rows)
    return out


def _coverage(rows: list[dict], trading_days: list[str]) -> dict:
    logged = {r["decision_date"] for r in rows}
    _missing, max_gap = stats.missing_and_max_gap(logged, trading_days)
    bad_version = sorted(
        {
            r["job_version"]
            for r in rows
            if r["job_version"] != CRITERIA["log_job_version"]
        }
    )
    ok = (
        len(logged) >= int(CRITERIA["min_logged_days"])
        and max_gap <= int(CRITERIA["max_gap_sessions"])
        and not bad_version
    )
    return {
        "logged_days": len(logged),
        "trading_days": len(trading_days),
        "max_gap_sessions": max_gap,
        "unexpected_job_versions": bad_version,
        "pass": ok,
    }


def _fidelity(
    rows: list[dict], retro: dict[tuple[str, str], dict[str, float]]
) -> dict:
    diffs = [
        stats.max_weight_diff(
            r["weights"], retro[(r["decision_date"], r["strategy"])]
        )
        for r in rows
    ]
    matches = sum(1 for d in diffs if d <= float(CRITERIA["weight_tol"]))
    worst = max(diffs, default=0.0)
    rate = matches / len(diffs) if diffs else 0.0
    ok = rate >= float(CRITERIA["min_match_rate"]) and worst <= float(
        CRITERIA["weight_tol_hard"]
    )
    return {
        "rows": len(diffs),
        "match_rate": round(rate, 4),
        "worst_diff": round(worst, 6),
        "pass": ok,
    }


def _behavior(name: str, rows: list[dict], reference: list[dict]) -> dict:
    observed = window_stats(name, rows)
    windows = stats.rolling_windows(reference, len(rows))
    report: dict[str, dict] = {}
    ok = True
    for stat, value in observed.items():
        ref_values = [window_stats(name, w)[stat] for w in windows]
        lo, hi = stats.percentile_band(
            ref_values, float(CRITERIA["band_lo_pct"]), float(CRITERIA["band_hi_pct"])
        )
        inside = lo <= value <= hi
        ok = ok and inside
        report[stat] = {
            "value": round(value, 6),
            "band": [round(lo, 6), round(hi, 6)],
            "pass": inside,
        }
    return {"stats": report, "pass": ok}


def evaluate(
    log_path: Path, store_dir: Path, today: date | None = None
) -> dict:
    check_evaluation_date(today or datetime.now(UTC).date())
    lo, hi = str(CRITERIA["window_start"]), str(CRITERIA["window_end"])
    rows = [
        r
        for r in load_decisions(log_path)
        if lo <= r["decision_date"] <= hi
    ]
    panel = build_race_panel(store_dir)
    trading_days = trading_days_between(panel, lo, hi)
    decision_dates = sorted({r["decision_date"] for r in rows})
    retro = retro_weights(panel, store_dir, decision_dates)
    reference = reference_rows(panel, store_dir)

    strategies = []
    for name in STRATEGIES:
        mine = sorted(
            (r for r in rows if r["strategy"] == name),
            key=lambda r: r["decision_date"],
        )
        coverage = _coverage(mine, trading_days)
        fidelity = _fidelity(mine, retro)
        behavior = _behavior(name, mine, reference[name])
        verdict = (
            "PASS"
            if coverage["pass"] and fidelity["pass"] and behavior["pass"]
            else "FAIL"
        )
        strategies.append(
            {
                "strategy": name,
                "coverage": coverage,
                "fidelity": fidelity,
                "behavior": behavior,
                "verdict": verdict,
            }
        )
    return {"criteria": dict(CRITERIA), "strategies": strategies}
