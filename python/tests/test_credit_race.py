"""Invariant tests for the credit race: causal z-scores, widening brake."""

import numpy as np
from credit_race.race import causal_zscores, widening_brake


class TestCausalZscores:
    def test_no_future_dependence(self):
        rng = np.random.default_rng(1)
        values = rng.normal(0, 1, 400)
        base = causal_zscores(values)
        shocked = values.copy()
        shocked[300:] = 50.0
        assert np.allclose(
            causal_zscores(shocked)[:300], base[:300], equal_nan=True
        )

    def test_warmup_is_nan(self):
        rng = np.random.default_rng(2)
        out = causal_zscores(rng.normal(0, 1, 400), window=252, min_obs=120)
        assert np.isnan(out[:120]).all()
        assert np.isfinite(out[121:]).any()

    def test_current_value_excluded_from_baseline(self):
        # A lone spike must register as a spike, not shift its own baseline.
        values = np.zeros(200)
        values += np.linspace(0, 0.01, 200)  # tiny drift so std > 0
        values[150] = 10.0
        out = causal_zscores(values, window=100, min_obs=50)
        assert out[150] > 5.0


class TestWideningBrake:
    def test_spread_spike_engages_brake(self):
        rng = np.random.default_rng(3)
        spread = 2.0 + np.cumsum(rng.normal(0, 0.01, 500))
        spread[400:] += 1.5  # abrupt widening
        active = widening_brake(spread)
        assert active[400:406].any()
        assert not active[:400].any() or active[:400].mean() < 0.2

    def test_flat_spread_never_brakes(self):
        rng = np.random.default_rng(4)
        spread = 2.0 + rng.normal(0, 0.005, 500)
        assert widening_brake(spread).mean() < 0.2
