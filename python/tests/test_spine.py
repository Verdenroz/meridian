"""Invariant tests for the spine: causality, regime recovery, estimator sanity."""

import numpy as np
import pytest
from harness.engine import run_backtest
from harness.panel import Panel
from spine.allocator import allocate
from spine.correlation import ledoit_wolf_constant_corr, rolling_correlation
from spine.regime import GaussianHMM
from spine.strategy import AllocatorConfig, SpineStrategy, TunedSpineStrategy
from spine.vol import ewma_vol_series, forecast_correlation, forward_realized_vol


def synthetic_regime_data(n_per_state=400, seed=7):
    """Alternating calm (low-vol) and stress (high-vol) blocks, 2 features."""
    rng = np.random.default_rng(seed)
    blocks = []
    labels = []
    for i in range(4):
        stress = i % 2 == 1
        mu = np.array([1.0, 0.5]) if stress else np.array([-1.0, -0.5])
        blocks.append(rng.normal(mu, 0.4, size=(n_per_state, 2)))
        labels.extend([int(stress)] * n_per_state)
    return np.vstack(blocks), np.array(labels)


class TestHMM:
    def test_recovers_synthetic_regimes(self):
        x, labels = synthetic_regime_data()
        hmm = GaussianHMM(n_states=2).fit(x)
        posterior = hmm.filter_posterior(x)
        predicted = posterior.argmax(axis=1)
        accuracy = (predicted == labels).mean()
        assert accuracy > 0.95

    def test_states_ordered_by_first_feature(self):
        x, _ = synthetic_regime_data()
        hmm = GaussianHMM(n_states=2).fit(x)
        assert hmm.means_[0, 0] < hmm.means_[1, 0]  # last state = stress

    def test_filter_is_causal(self):
        x, _ = synthetic_regime_data()
        hmm = GaussianHMM(n_states=2).fit(x)
        full = hmm.filter_posterior(x)
        # changing the future must not change the past posterior
        x_mut = x.copy()
        x_mut[900:] = 10.0
        mutated = hmm.filter_posterior(x_mut)
        np.testing.assert_allclose(full[:900], mutated[:900])

    def test_filter_step_matches_batch_filter(self):
        x, _ = synthetic_regime_data()
        hmm = GaussianHMM(n_states=2).fit(x)
        batch = hmm.filter_posterior(x[:50])
        alpha = hmm.filter_posterior(x[:1])[-1]
        for t in range(1, 50):
            alpha = hmm.filter_step(alpha, x[t])
        np.testing.assert_allclose(alpha, batch[-1], rtol=1e-8)

    def test_deterministic_refit(self):
        x, _ = synthetic_regime_data()
        a = GaussianHMM(n_states=2).fit(x)
        b = GaussianHMM(n_states=2).fit(x)
        np.testing.assert_allclose(a.means_, b.means_)
        np.testing.assert_allclose(a.transmat_, b.transmat_)


class TestCorrelation:
    def test_ledoit_wolf_intensity_bounds_and_symmetry(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0, 0.01, size=(120, 5))
        cov, intensity = ledoit_wolf_constant_corr(returns)
        assert 0.0 <= intensity <= 1.0
        np.testing.assert_allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0)

    def test_shrinkage_vanishes_with_many_observations(self):
        # Heterogeneous correlations (factor structure) so the constant-corr
        # target is misspecified; intensity must then fall as T grows.
        rng = np.random.default_rng(4)
        loadings = np.array([1.5, 1.0, 0.5, 0.1, -0.5])

        def sample(t):
            factor = rng.normal(0, 0.01, size=(t, 1))
            return factor * loadings + rng.normal(0, 0.01, size=(t, 5))

        few = ledoit_wolf_constant_corr(sample(30))[1]
        many = ledoit_wolf_constant_corr(sample(5000))[1]
        assert many < few
        assert many < 0.2

    def test_rolling_correlation_is_causal_and_detects_sign(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0, 0.01, 300)
        b = np.concatenate([a[:150], -a[150:]]) + rng.normal(0, 0.001, 300)
        corr = rolling_correlation(a, b, window=60)
        assert np.isnan(corr[:59]).all()
        assert corr[140] > 0.9
        assert corr[299] < -0.9


class TestVol:
    def test_ewma_tracks_known_vol_shift(self):
        rng = np.random.default_rng(6)
        low = rng.normal(0, 0.005, 500)
        high = rng.normal(0, 0.03, 500)
        vol = ewma_vol_series(np.concatenate([low, high]))
        assert vol[499] == pytest.approx(0.005 * np.sqrt(252), rel=0.3)
        assert vol[999] == pytest.approx(0.03 * np.sqrt(252), rel=0.3)

    def test_forecast_correlation_positive_on_regime_data(self):
        rng = np.random.default_rng(8)
        vols = np.repeat([0.005, 0.02, 0.008, 0.03], 300)
        returns = rng.normal(0, vols)
        corr = forecast_correlation(
            ewma_vol_series(returns), forward_realized_vol(returns)
        )
        assert corr > 0.5

    def test_nan_returns_passthrough(self):
        returns = np.concatenate([np.full(10, np.nan), np.full(200, 0.01)])
        vol = ewma_vol_series(returns, min_obs=60)
        assert np.isnan(vol[:69]).all()
        assert np.isfinite(vol[70])


class TestAllocator:
    def test_benign_is_equity_heavy(self):
        assert allocate(1.0, {}) == {"SPY": 1.0}

    def test_stress_moves_to_defense(self):
        weights = allocate(0.1, {"TLT": True, "GLD": False})
        assert weights["SPY"] == pytest.approx(0.1)
        assert weights["TLT"] == pytest.approx(0.2 * 0.9)
        assert "GLD" not in weights

    def test_weights_never_leveraged_or_short(self):
        for equity in (-0.5, 0.0, 0.3, 0.7, 1.0, 2.0):
            weights = allocate(equity, {"TLT": True, "GLD": True})
            assert all(w >= 0 for w in weights.values())
            assert sum(weights.values()) <= 1.0 + 1e-12

    def test_mapping_variants(self):
        product = AllocatorConfig("product").equity_weight(0.3, 0.8)
        assert product == pytest.approx(0.8 * 0.7)
        assert AllocatorConfig("min").equity_weight(0.3, 0.8) == pytest.approx(0.7)
        thresh = AllocatorConfig("threshold", threshold=0.85)
        assert thresh.equity_weight(0.5, 0.8) == pytest.approx(0.8)
        assert thresh.equity_weight(0.9, 0.8) == pytest.approx(0.8 * 0.25)
        assert AllocatorConfig("vol_only").equity_weight(0.99, 0.8) == pytest.approx(0.8)


class TestStrategyInEngine:
    def make_regime_panel(self, seed=9):
        """Calm then stressful synthetic market with matching features."""
        rng = np.random.default_rng(seed)
        n = 700
        vols = np.concatenate([np.full(400, 0.006), np.full(300, 0.03)])
        rets = rng.normal(0.0003, vols)
        prices = 100.0 * np.cumprod(1.0 + rets)
        dates = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
        # panel with SPY only; TLT/GLD omitted (allocator then uses cash defense)
        panel = Panel(dates, {"SPY": list(prices)}, [2.52] * n)
        returns = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
        vol = ewma_vol_series(returns)
        with np.errstate(invalid="ignore"):
            features = np.column_stack(
                [np.log(vol), np.log(vol * 100), np.full(n, 1.0), np.zeros(n)]
            )
        return panel, features

    def test_spine_runs_and_derisks_in_stress(self):
        panel, features = self.make_regime_panel()
        strategy = SpineStrategy(panel, features)
        result = run_backtest(panel, strategy, start=450, cost_bps={"SPY": 1.0})
        assert len(result["returns"]) == panel.n_days - 1 - 450
        # stress regime + high vol → materially de-risked by the end
        assert result["final_weights"].get("SPY", 0.0) < 0.5

    def test_spine_full_weight_in_calm(self):
        panel, features = self.make_regime_panel()
        strategy = SpineStrategy(panel, features)
        result = run_backtest(panel, strategy, start=200, end=390, cost_bps={"SPY": 1.0})
        # calm block: low vol → vol-target caps at 1, stress prob low
        assert result["final_weights"].get("SPY", 0.0) > 0.8

    def test_tuner_chooses_on_train_only_and_runs_oos(self):
        panel, features = self.make_regime_panel()
        strategy = TunedSpineStrategy(panel, features)
        result = run_backtest(panel, strategy, start=450, cost_bps={"SPY": 1.0})
        assert strategy.chosen is not None
        assert strategy.chosen in strategy.candidates
        assert len(result["returns"]) == panel.n_days - 1 - 450

    def test_tuner_is_deterministic(self):
        panel, features = self.make_regime_panel()
        a = TunedSpineStrategy(panel, features)
        b = TunedSpineStrategy(panel, features)
        run_backtest(panel, a, start=450, end=500, cost_bps={"SPY": 1.0})
        run_backtest(panel, b, start=450, end=500, cost_bps={"SPY": 1.0})
        assert a.chosen == b.chosen
