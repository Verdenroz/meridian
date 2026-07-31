"""Invariant tests for the event study: stats, alignment, causality, gate."""

import numpy as np
import pytest
from event_study.events import intensity_zscores
from event_study.stats import mann_whitney_u, spearman
from event_study.study import _event_indices, _passes, _uplift


class TestStats:
    def test_mwu_detects_shifted_distribution(self):
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 0.5, 200)
        effect, p = mann_whitney_u(a + 0.5, a)
        assert effect > 0.2
        assert p < 1e-6

    def test_mwu_null_is_uniform_ish(self):
        rng = np.random.default_rng(2)
        ps = []
        for _ in range(200):
            a = rng.normal(0, 1, 60)
            b = rng.normal(0, 1, 60)
            ps.append(mann_whitney_u(a, b)[1])
        # one-sided p under the null should be centered, rarely tiny
        assert 0.35 < float(np.mean(ps)) < 0.65
        assert float(np.mean(np.array(ps) < 0.01)) < 0.05

    def test_mwu_handles_ties(self):
        a = np.array([1.0, 2.0, 2.0, 3.0] * 10)
        b = np.array([1.0, 1.0, 2.0, 2.0] * 10)
        effect, p = mann_whitney_u(a, b)
        assert effect > 0
        assert 0 < p < 0.05

    def test_spearman_monotonic_relationship(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 500)
        rho, p = spearman(x, np.exp(x) + rng.normal(0, 0.1, 500))
        assert rho > 0.9
        assert p < 1e-10

    def test_spearman_null(self):
        rng = np.random.default_rng(4)
        rho, _p = spearman(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        assert abs(rho) < 0.15


class TestAlignment:
    def test_event_maps_to_same_trading_day(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert _event_indices(dates, ["2024-01-03"]) == {1}

    def test_weekend_event_rolls_to_next_trading_day(self):
        dates = ["2024-01-05", "2024-01-08", "2024-01-09"]  # Fri, Mon, Tue
        assert _event_indices(dates, ["2024-01-06"]) == {1}

    def test_event_after_history_end_is_dropped(self):
        dates = ["2024-01-02", "2024-01-03"]
        assert _event_indices(dates, ["2024-02-01"]) == set()


class TestIntensity:
    def test_zscore_is_causal(self):
        dates = [f"d{i:04d}" for i in range(400)]
        counts = {d: (8 if i % 2 else 12) for i, d in enumerate(dates)}
        counts["d0350"] = 100
        z = intensity_zscores(dates, counts, window=252, min_obs=120)
        # burst day itself spikes; the day before is unaffected by the future
        z_mut = intensity_zscores(
            dates, {**counts, "d0399": 500}, window=252, min_obs=120
        )
        np.testing.assert_allclose(
            np.nan_to_num(z[:399]), np.nan_to_num(z_mut[:399])
        )
        assert z[350] > 5

    def test_zscore_warmup_is_nan(self):
        dates = [f"d{i:04d}" for i in range(200)]
        z = intensity_zscores(dates, {d: 5 for d in dates}, window=252, min_obs=120)
        assert np.isnan(z[:120]).all()


class TestGate:
    def test_uplift_computation(self):
        events = np.array([1.5] * 150)
        baseline = np.array([1.0] * 500)
        result = _uplift(events, baseline)
        assert result["median_uplift"] == pytest.approx(0.5)
        assert result["p_value"] < 1e-10

    def test_gate_requires_both_uplift_and_significance(self):
        assert _passes({"median_uplift": 0.10, "p_value": 1e-5})
        assert not _passes({"median_uplift": 0.04, "p_value": 1e-5})
        assert not _passes({"median_uplift": 0.10, "p_value": 0.05})
        assert not _passes(None)
