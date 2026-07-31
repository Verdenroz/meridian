"""Invariant tests for the vol race: causality, calibration honesty."""

from types import SimpleNamespace

import numpy as np
from vol_race.forecasters import HAR_MIN_OBS, har_lite_vol_series
from vol_race.race import CalibratedVolTarget


class TestHarLite:
    def test_warmup_is_nan(self):
        rng = np.random.default_rng(1)
        out = har_lite_vol_series(rng.normal(0, 0.01, 40))
        assert np.isnan(out[: HAR_MIN_OBS - 1]).all()
        assert np.isfinite(out[HAR_MIN_OBS - 1 :]).all()

    def test_causal_no_future_dependence(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, 100)
        base = har_lite_vol_series(returns)
        shocked = returns.copy()
        shocked[60:] = 0.25  # violent future change
        assert np.allclose(
            har_lite_vol_series(shocked)[:60], base[:60], equal_nan=True
        )

    def test_vol_spike_raises_forecast(self):
        returns = np.full(60, 0.001)
        returns[40:] = 0.03
        out = har_lite_vol_series(returns)
        assert out[45] > out[39] * 2


class TestCalibration:
    def _strategy(self, returns):
        forecast = np.full(len(returns), 0.10)
        return CalibratedVolTarget("SPY", forecast, returns, horizon=21)

    def test_scale_uses_only_closed_forward_windows(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0, 0.01, 300)
        t = 200
        strategy = self._strategy(returns)
        strategy.fit(SimpleNamespace(t=t))
        shocked = returns.copy()
        shocked[t + 1 :] = 0.30  # future data must not move the calibration
        other = self._strategy(shocked)
        other.fit(SimpleNamespace(t=t))
        assert strategy._scale == other._scale

    def test_closed_window_shock_moves_the_scale(self):
        # Guard against over-exclusion: data well inside the training span
        # (windows closed long before t) must influence the calibration.
        rng = np.random.default_rng(4)
        returns = rng.normal(0, 0.01, 300)
        t = 200
        strategy = self._strategy(returns)
        strategy.fit(SimpleNamespace(t=t))
        shocked = returns.copy()
        shocked[100] = 0.30
        other = self._strategy(shocked)
        other.fit(SimpleNamespace(t=t))
        assert other._scale > strategy._scale

    def test_biased_forecast_gets_rescaled(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0, 0.01, 400)  # ~16% annualized realized vol
        strategy = self._strategy(returns)  # constant 10% forecast, biased low
        strategy.fit(SimpleNamespace(t=399))
        assert strategy._scale > 1.2
