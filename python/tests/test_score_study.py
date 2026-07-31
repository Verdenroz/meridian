"""Invariant tests for the score study: point-in-time join and gate logic."""

import numpy as np
from score_study.study import GATE, leg_bands, scored_event_days

DATES = ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"]


def _row(date: str, severity: int, direction: str = "risk_up") -> dict:
    return {"filing_date": date, "severity": severity, "direction": direction}


class TestScoredEventDays:
    def test_trading_day_maps_to_itself(self):
        out = scored_event_days(DATES, [_row("2026-07-28", 3)])
        assert set(out) == {1}

    def test_non_trading_day_maps_to_next_session(self):
        # 07-26 is not in the trading calendar; first reaction is 07-27.
        out = scored_event_days(DATES, [_row("2026-07-26", 4)])
        assert set(out) == {0}
        assert out[0]["severity"] == 4

    def test_filing_after_last_session_is_dropped(self):
        assert scored_event_days(DATES, [_row("2026-08-04", 9)]) == {}

    def test_collision_keeps_max_severity(self):
        rows = [_row("2026-07-29", 2, "neutral"), _row("2026-07-29", 7, "risk_down")]
        out = scored_event_days(DATES, rows)
        assert out[2] == {"severity": 7, "direction": "risk_down"}

    def test_collision_order_independent(self):
        rows = [_row("2026-07-29", 7), _row("2026-07-29", 2)]
        assert scored_event_days(DATES, rows)[2]["severity"] == 7


class TestLegBands:
    def test_insufficient_events_returns_none(self):
        n = GATE["min_events_per_band"] - 1
        severities = np.array([7] * n + [1] * n)
        assert leg_bands(severities, np.ones(2 * n)) is None

    def test_separated_bands_pass_shape(self):
        rng = np.random.default_rng(5)
        n = GATE["min_events_per_band"]
        severities = np.array([7] * n + [1] * (4 * n))
        ratios = np.concatenate(
            [rng.lognormal(0.3, 0.2, n), rng.lognormal(0.0, 0.2, 4 * n)]
        )
        result = leg_bands(severities, ratios)
        assert result is not None
        assert result["median_uplift"] > GATE["min_median_uplift"]
        assert result["p_value"] < GATE["max_p_value"]

    def test_mid_band_severities_excluded(self):
        n = GATE["min_events_per_band"]
        severities = np.array([7] * n + [4] * n + [1] * n)
        result = leg_bands(severities, np.ones(3 * n))
        assert result is not None
        assert result["n_serious"] == n
        assert result["n_routine"] == n
