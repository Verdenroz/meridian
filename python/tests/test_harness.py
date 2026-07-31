"""Invariant tests for the harness: engine math, guardrails, no-lookahead."""

import itertools
import math

import pytest
from harness.engine import GuardrailViolation, run_backtest
from harness.metrics import compute_metrics, max_drawdown
from harness.panel import Panel
from harness.strategies import BuyHold, VolTarget
from harness.view import HistoryView
from harness.walkforward import run_walk_forward, walk_forward_splits


def make_panel(prices, cash_pct=0.0, symbol="AAA"):
    dates = [f"2024-01-{i + 1:02d}" for i in range(len(prices))]
    return Panel(dates, {symbol: prices}, [cash_pct] * len(prices))


class FixedWeight:
    """Sets a fixed weight once at the start, then drifts."""

    def __init__(self, symbol, weight):
        self.symbol, self.weight = symbol, weight
        self._done = False

    def fit(self, view):
        self._done = False

    def target_weights(self, view):
        if self._done:
            return None
        self._done = True
        return {self.symbol: self.weight}


class TestEngine:
    def test_buy_hold_tracks_asset_after_costs(self):
        panel = make_panel([100.0, 110.0, 99.0])
        result = run_backtest(panel, BuyHold("AAA"), start=0, cost_bps={"AAA": 0.0})
        assert result["returns"] == pytest.approx([0.10, -0.10])
        assert result["final_value"] == pytest.approx(0.99)

    def test_entry_cost_nets_out_of_first_return(self):
        panel = make_panel([100.0, 110.0])
        result = run_backtest(panel, BuyHold("AAA"), start=0, cost_bps={"AAA": 100.0})
        # 1% one-way cost on full entry: (1.10)*(0.99) - 1
        assert result["returns"][0] == pytest.approx(1.10 * 0.99 - 1.0)
        assert result["final_value"] == pytest.approx(1.10 * 0.99)

    def test_partial_weight_earns_cash_on_remainder(self):
        panel = make_panel([100.0, 110.0], cash_pct=25.2)  # 0.1%/day cash
        result = run_backtest(
            panel, FixedWeight("AAA", 0.5), start=0, cost_bps={"AAA": 0.0}
        )
        assert result["returns"][0] == pytest.approx(0.5 * 0.10 + 0.5 * 0.001)

    def test_weights_drift_with_returns(self):
        panel = make_panel([100.0, 200.0])
        result = run_backtest(
            panel, FixedWeight("AAA", 0.5), start=0, cost_bps={"AAA": 0.0}
        )
        # 0.5 → asset doubles → 1.0/1.5 of portfolio
        assert result["final_weights"]["AAA"] == pytest.approx(2.0 / 3.0)

    def test_guardrail_rejects_leverage_and_shorts(self):
        panel = make_panel([100.0, 101.0, 102.0])
        with pytest.raises(GuardrailViolation):
            run_backtest(panel, FixedWeight("AAA", 1.5), start=0)
        with pytest.raises(GuardrailViolation):
            run_backtest(panel, FixedWeight("AAA", -0.5), start=0)

    def test_turnover_accounts_both_legs(self):
        panel = make_panel([100.0, 100.0, 100.0])

        class Flip:
            def fit(self, view):
                pass

            def target_weights(self, view):
                return {"AAA": 1.0} if view.t % 2 == 0 else {"AAA": 0.0}

        result = run_backtest(panel, Flip(), start=0, cost_bps={"AAA": 0.0})
        # enter 1.0, exit 1.0, enter 1.0
        assert result["turnover_oneway"] == pytest.approx(3.0)


class TestNoLookahead:
    def test_view_cannot_see_future_returns(self):
        panel = make_panel([100.0, 101.0, 102.0, 103.0])
        view = HistoryView(panel, 1)
        window = view.returns_window("AAA", 10)
        assert window == pytest.approx([0.01])  # only the day-1 return exists yet

    def test_view_today_matches_index(self):
        panel = make_panel([100.0, 101.0])
        assert HistoryView(panel, 0).today == "2024-01-01"


class TestVolTarget:
    def test_scales_to_target_over_realized_vol(self):
        # Alternating ±1% daily → ann vol ≈ 16%; expect weight ≈ 0.15/0.16
        prices = [100.0]
        for i in range(400):
            prices.append(prices[-1] * (1.01 if i % 2 == 0 else 0.99))
        panel = make_panel(prices)
        strat = VolTarget("AAA", target_vol=0.15)
        targets = strat.target_weights(HistoryView(panel, len(prices) - 1))
        assert targets is not None
        vol = 0.01 * math.sqrt(252)
        assert targets["AAA"] == pytest.approx(0.15 / vol, rel=0.05)

    def test_caps_at_full_investment(self):
        prices = [100.0 * (1.0001**i) for i in range(300)]  # near-zero vol
        panel = make_panel(prices)
        targets = VolTarget("AAA").target_weights(HistoryView(panel, 299))
        assert targets == {"AAA": 1.0}

    def test_goes_to_cash_without_enough_history(self):
        panel = make_panel([100.0, 101.0, 102.0])
        targets = VolTarget("AAA").target_weights(HistoryView(panel, 2))
        assert targets == {"AAA": 0.0}

    def test_band_suppresses_micro_rebalances(self):
        prices = [100.0]
        for i in range(400):
            prices.append(prices[-1] * (1.01 if i % 2 == 0 else 0.99))
        panel = make_panel(prices)
        strat = VolTarget("AAA")
        first = strat.target_weights(HistoryView(panel, 398))
        second = strat.target_weights(HistoryView(panel, 399))
        assert first is not None and second is None


class TestMetrics:
    def test_max_drawdown_known_series(self):
        # +10%, -50%, +20%: trough at 0.55 from peak 1.10 → 50%
        assert max_drawdown([0.10, -0.50, 0.20]) == pytest.approx(0.50)

    def test_cagr_round_trip(self):
        returns = [0.001] * 504
        m = compute_metrics(returns)
        assert m["cagr"] == pytest.approx(1.001**252 - 1, rel=1e-9)
        assert m["days"] == 504


class TestWalkForward:
    def test_splits_are_disjoint_and_cover_tail(self):
        splits = walk_forward_splits(0, 1000, min_train_days=400, test_days=252)
        assert splits[0][1] == 400
        assert splits[-1][2] == 999
        for (_, _, prev_end), (_, next_start, _) in itertools.pairwise(splits):
            assert next_start == prev_end + 1

    def test_fit_end_precedes_test_start(self):
        for fit_end, test_start, _ in walk_forward_splits(5, 800, 300, 100):
            assert fit_end == test_start - 1

    def test_stitched_buy_hold_matches_full_run(self):
        prices = [100.0 * (1.0 + 0.0003 * ((i * 7) % 5 - 2)) ** i for i in range(700)]
        panel = make_panel(prices)
        wf = run_walk_forward(
            panel,
            lambda: BuyHold("AAA"),
            first_index=0,
            min_train_days=100,
            test_days=200,
            cost_bps={"AAA": 0.0},
        )
        full = run_backtest(panel, BuyHold("AAA"), start=99, cost_bps={"AAA": 0.0})
        assert wf["dates"] == full["dates"]
        # zero costs + refit-free strategy → identical stitched daily returns
        assert wf["returns"] == pytest.approx(full["returns"])
