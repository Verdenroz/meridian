"""Invariant tests for the stock sleeve: causality, eligibility, sizing
guardrails, and the pooled ridge's training-window honesty."""

import numpy as np
from harness.engine import run_backtest
from harness.panel import Panel
from sleeve.forecasters import (
    PooledRidgeForecaster,
    StaticForecast,
    finite_run_lengths,
    forward_realized_vol_matrix,
    ridge_features,
    trailing_realized_vol,
)
from sleeve.panel import _fill_isolated_gaps
from sleeve.risk import basket_vol, capped_inverse_vol_weights
from sleeve.strategy import SleeveVolTarget, sleeve_data

RNG = np.random.default_rng(7)


def _synthetic_panel(n_days=900, n_names=12, inception=None):
    inception = inception or {}
    names = tuple(f"S{i}" for i in range(n_names))
    market = RNG.normal(0.0003, 0.01, n_days)
    prices = {"SPY": list(100 * np.cumprod(1 + market))}
    for j, name in enumerate(names):
        rets = 0.8 * market + RNG.normal(0, 0.008 * (1 + j / n_names), n_days)
        series = list(100 * np.cumprod(1 + rets))
        start = inception.get(name, 0)
        prices[name] = [None] * start + series[start:]
    dates = [f"d{i:05d}" for i in range(n_days)]
    return Panel(dates, prices, [2.0] * n_days), names


def _strategy(data, forecaster, sizing="inverse_vol", min_names=8):
    return SleeveVolTarget(
        data, forecaster, sizing=sizing, target_vol=0.15, band=0.02,
        max_name_weight=0.10, min_names=min_names, min_history=252,
        corr_window=252, corr_refit_days=21,
    )


class TestForecasterCausality:
    def test_trailing_realized_ignores_future(self):
        r = RNG.normal(0, 0.01, (300, 3))
        base = trailing_realized_vol(r, 22)
        shocked = r.copy()
        shocked[200:] = 0.3
        assert np.allclose(base[:200], trailing_realized_vol(shocked, 22)[:200], equal_nan=True)

    def test_forward_realized_is_future_only(self):
        r = RNG.normal(0, 0.01, (100, 2))
        fwd = forward_realized_vol_matrix(r, 21)
        assert np.isnan(fwd[-21:]).all()
        expected = np.sqrt(np.mean(r[1:22, 0] ** 2) * 252)
        assert np.isclose(fwd[0, 0], expected)

    def test_run_lengths_reset_on_gap(self):
        r = np.ones((6, 1))
        r[3, 0] = np.nan
        assert finite_run_lengths(r)[:, 0].tolist() == [1, 2, 3, 0, 1, 2]

    def test_ridge_features_are_causal(self):
        r = RNG.normal(0, 0.01, (400, 3))
        m = RNG.normal(0, 0.01, 400)
        base = ridge_features(r, m)
        shocked_r, shocked_m = r.copy(), m.copy()
        shocked_r[300:] = 0.2
        shocked_m[300:] = 0.2
        assert np.allclose(base[:300], ridge_features(shocked_r, shocked_m)[:300], equal_nan=True)


class TestPooledRidge:
    def _forecaster(self, r, m):
        return PooledRidgeForecaster(ridge_features(r, m), forward_realized_vol_matrix(r, 21), 21, 1.0)

    def test_fit_uses_only_closed_forward_windows(self):
        r = RNG.normal(0, 0.01, (800, 5))
        m = RNG.normal(0, 0.01, 800)
        t = 600
        base = self._forecaster(r, m)
        base.fit(t)
        shocked = r.copy()
        shocked[t + 1 :] = 0.3
        other = self._forecaster(shocked, m)
        other.fit(t)
        assert np.allclose(base.coef, other.coef)

    def test_closed_window_shock_moves_the_fit(self):
        r = RNG.normal(0, 0.01, (800, 5))
        m = RNG.normal(0, 0.01, 800)
        base = self._forecaster(r, m)
        base.fit(600)
        shocked = r.copy()
        shocked[300:330] = 0.1
        other = self._forecaster(shocked, m)
        other.fit(600)
        assert not np.allclose(base.coef, other.coef)

    def test_forecasts_positive_where_defined(self):
        r = RNG.normal(0, 0.01, (800, 5))
        m = RNG.normal(0, 0.01, 800)
        pred = self._forecaster(r, m).fit(600)
        finite = np.isfinite(pred)
        assert finite[100:].all()
        assert (pred[finite] > 0).all()


class TestSizing:
    def test_capped_weights_sum_to_one_and_respect_cap(self):
        vols = np.array([0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        w = capped_inverse_vol_weights(vols, 0.10)
        assert np.isclose(w.sum(), 1.0)
        assert w.max() <= 0.10 + 1e-9
        assert w.argmin() == vols.argmax()

    def test_infeasible_cap_degrades_to_equal(self):
        w = capped_inverse_vol_weights(np.array([0.1, 0.2, 0.3]), 0.10)
        assert np.allclose(w, 1 / 3)

    def test_basket_vol_below_average_when_diversified(self):
        vols = np.full(4, 0.2)
        corr = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
        assert basket_vol(np.full(4, 0.25), vols, corr) < 0.2


class TestSleeveStrategy:
    def test_gap_fill_is_isolated_only(self):
        assert _fill_isolated_gaps([1.0, None, 2.0]) == ([1.0, 1.0, 2.0], 1)
        assert _fill_isolated_gaps([None, None, 1.0, 2.0]) == ([None, None, 1.0, 2.0], 0)
        long_gap = [1.0] + [None] * 5 + [2.0]
        assert _fill_isolated_gaps(long_gap) == (long_gap, 0)

    def test_backtest_respects_guardrails_and_cap(self):
        panel, names = _synthetic_panel()
        data = sleeve_data(panel, "SPY", names)
        forecast = StaticForecast(np.full(data.returns.shape, 0.2))
        strategy = _strategy(data, forecast)
        result = run_backtest(panel, strategy, start=300)
        weights = result["final_weights"]
        assert "SPY" not in weights
        assert sum(weights.values()) <= 1.0 + 1e-9
        assert min(weights.values()) >= 0.0
        targets = strategy._last_target
        assert max(targets.values()) <= 0.10 + 1e-9

    def test_falls_back_to_index_when_thin(self):
        panel, names = _synthetic_panel(n_names=5)
        data = sleeve_data(panel, "SPY", names)
        strategy = _strategy(data, StaticForecast(np.full(data.returns.shape, 0.2)))
        result = run_backtest(panel, strategy, start=300)
        assert set(result["final_weights"]) == {"SPY"}

    def test_late_inception_joins_only_after_min_history(self):
        panel, names = _synthetic_panel(inception={"S0": 500})
        data = sleeve_data(panel, "SPY", names)
        strategy = _strategy(data, StaticForecast(np.full(data.returns.shape, 0.2)))
        strategy.fit(type("V", (), {"t": 300})())
        assert "S0" not in strategy.target_weights(type("V", (), {"t": 700})())
        strategy._last_target = None
        assert "S0" in strategy.target_weights(type("V", (), {"t": 760})())

    def test_band_suppresses_small_drift(self):
        panel, names = _synthetic_panel()
        data = sleeve_data(panel, "SPY", names)
        strategy = _strategy(data, StaticForecast(np.full(data.returns.shape, 0.2)))
        strategy.fit(type("V", (), {"t": 300})())
        assert strategy.target_weights(type("V", (), {"t": 300})()) is not None
        assert strategy.target_weights(type("V", (), {"t": 301})()) is None
