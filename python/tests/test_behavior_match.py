"""Behavior-match invariants: frozen criteria sanity, the no-peeking guard,
and the pure window statistics the verdicts are computed from."""

from __future__ import annotations

from datetime import date

import pytest
from behavior_match import stats
from behavior_match.criteria import CRITERIA
from behavior_match.evaluate import (
    EvaluationNotOpen,
    _behavior,
    _coverage,
    _fidelity,
    check_evaluation_date,
)


def row(d: str, weights: dict[str, float], **meta: object) -> dict:
    return {
        "decision_date": d,
        "strategy": "null_voltarget",
        "job_version": CRITERIA["log_job_version"],
        "weights": weights,
        "cash_weight": round(1.0 - sum(weights.values()), 6),
        "meta": dict(meta),
    }


def test_criteria_are_internally_consistent() -> None:
    assert CRITERIA["window_start"] < CRITERIA["window_end"]
    assert CRITERIA["window_end"] < CRITERIA["earliest_evaluation"]
    assert CRITERIA["reference_end"] < CRITERIA["window_start"]
    assert 0 < CRITERIA["weight_tol"] < CRITERIA["weight_tol_hard"]
    assert 0 < CRITERIA["min_match_rate"] <= 1
    assert CRITERIA["band_lo_pct"] < CRITERIA["band_hi_pct"]


def test_guard_refuses_before_opening_date() -> None:
    with pytest.raises(EvaluationNotOpen):
        check_evaluation_date(date(2026, 10, 15))
    check_evaluation_date(date(2026, 11, 2))


def test_max_weight_diff_over_symbol_union() -> None:
    assert stats.max_weight_diff({"SPY": 0.8}, {"SPY": 0.7, "TLT": 0.05}) == pytest.approx(0.1)
    assert stats.max_weight_diff({}, {}) == 0.0
    assert stats.max_weight_diff({"SPY": 0.5}, {}) == pytest.approx(0.5)


def test_turnover_annualized() -> None:
    rows = [row("d1", {"SPY": 1.0}), row("d2", {"SPY": 0.5}), row("d3", {"SPY": 0.5})]
    assert stats.turnover_annualized(rows) == pytest.approx(0.5 * 252 / 2)
    assert stats.turnover_annualized(rows[:1]) == 0.0


def test_missing_and_max_gap() -> None:
    days = ["d1", "d2", "d3", "d4", "d5"]
    assert stats.missing_and_max_gap({"d1", "d2", "d3", "d4", "d5"}, days) == (0, 0)
    assert stats.missing_and_max_gap({"d1", "d4"}, days) == (3, 2)
    assert stats.missing_and_max_gap(set(), days) == (5, 5)


def test_percentile_band_and_rolling_windows() -> None:
    rows = [row(f"d{i}", {"SPY": i / 10}) for i in range(10)]
    windows = stats.rolling_windows(rows, 3)
    assert len(windows) == 8
    assert stats.rolling_windows(rows, 11) == []
    lo, hi = stats.percentile_band([1.0, 2.0, 3.0], 0.0, 100.0)
    assert (lo, hi) == (1.0, 3.0)


def test_coverage_flags_gaps_and_versions() -> None:
    days = [f"2026-08-{i:02d}" for i in range(1, 11)]
    rows = [row(d, {"SPY": 1.0}) for d in days]
    good = _coverage(rows, days)
    assert good["max_gap_sessions"] == 0 and not good["unexpected_job_versions"]
    rows[3]["job_version"] = "shadow-v2"
    assert _coverage(rows, days)["unexpected_job_versions"] == ["shadow-v2"]


def test_fidelity_tolerances() -> None:
    rows = [row("d1", {"SPY": 0.800000}), row("d2", {"SPY": 0.5})]
    retro = {("d1", "null_voltarget"): {"SPY": 0.805}, ("d2", "null_voltarget"): {"SPY": 0.5}}
    report = _fidelity(rows, retro)
    assert report["match_rate"] == 1.0 and report["pass"]
    retro[("d1", "null_voltarget")] = {"SPY": 0.86}  # beyond the hard tolerance
    assert not _fidelity(rows, retro)["pass"]


def test_behavior_band_membership() -> None:
    shadow = [row(f"s{i}", {"SPY": 0.5}) for i in range(3)]
    reference = [row(f"r{i}", {"SPY": 0.4 + 0.02 * (i % 10)}) for i in range(50)]
    report = _behavior("null_voltarget", shadow, reference)
    assert report["stats"]["mean_gross_exposure"]["pass"]
    outlier = [row(f"s{i}", {"SPY": 1.0}) for i in range(3)]
    assert not _behavior("null_voltarget", outlier, reference)["pass"]
