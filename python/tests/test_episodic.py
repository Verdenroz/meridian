"""Invariant tests for episodic memory: causal state, k-NN hygiene, outcomes."""

from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pytest
from episodic.briefing import build_briefing, format_briefing
from episodic.knn import retrieve_analogs
from episodic.outcomes import forward_outcomes
from episodic.state import STATE_FEATURES, STATE_SYMBOLS, build_state_matrix
from harness.panel import Panel
from market_data.store import write_snapshot

N_DAYS = 500


def make_dates(n=N_DAYS):
    start = date(2015, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def make_panel(dates, seed=3, mutate_from=None):
    rng = np.random.default_rng(seed)
    adj_close = {}
    for j, symbol in enumerate(STATE_SYMBOLS):
        rets = rng.normal(0.0003, 0.01 + 0.002 * j, size=len(dates))
        prices = 100.0 * np.cumprod(1.0 + rets)
        if mutate_from is not None:
            prices = prices.copy()
            prices[mutate_from:] *= 1.5
        adj_close[symbol] = [float(p) for p in prices]
    return Panel(dates, adj_close, [3.0] * len(dates))


def make_store(tmp_path, dates):
    for name, value in (("VIXCLS", 20.0), ("T10Y3M", 1.5), ("BAA10Y", 2.0)):
        rows = [{"date": d, "value": value} for d in dates]
        write_snapshot(tmp_path, "macro", name, {"rows": rows})
    write_snapshot(
        tmp_path, "macro", "treasury_yields",
        {"rows": [{"date": d, "y3M": 3.0} for d in dates]},
    )
    return tmp_path


class TestStateMatrix:
    def test_shape_and_warmup(self, tmp_path):
        dates = make_dates()
        store = make_store(tmp_path, dates)
        states = build_state_matrix(make_panel(dates), store)
        assert states.shape == (N_DAYS, len(STATE_FEATURES))
        assert np.isnan(states[0]).any()  # warm-up rows are NaN
        assert np.all(np.isfinite(states[-1]))  # settled rows are complete

    def test_causal(self, tmp_path):
        dates = make_dates()
        store = make_store(tmp_path, dates)
        cut = 300
        full = build_state_matrix(make_panel(dates), store)
        mutated = build_state_matrix(make_panel(dates, mutate_from=cut), store)
        # changing future prices must not change past state rows
        np.testing.assert_allclose(full[:cut], mutated[:cut])


class TestRetrieval:
    def make_states(self, n=400, seed=11):
        rng = np.random.default_rng(seed)
        states = rng.normal(size=(n, 3))
        return states, [f"d{i:04d}" for i in range(n)]

    def test_exclusion_window(self):
        states, dates = self.make_states()
        query = len(dates) - 1
        analogs = retrieve_analogs(states, dates, query, k=5, exclude_recent=63)
        assert analogs
        assert all(a.index <= query - 63 for a in analogs)

    def test_min_separation(self):
        states, dates = self.make_states()
        analogs = retrieve_analogs(states, dates, len(dates) - 1, k=5, min_separation=21)
        indices = sorted(a.index for a in analogs)
        assert all(b - a >= 21 for a, b in pairwise(indices))

    def test_planted_duplicate_is_nearest(self):
        states, dates = self.make_states()
        query = len(dates) - 1
        states[100] = states[query]
        analogs = retrieve_analogs(states, dates, query)
        assert analogs[0].index == 100
        assert analogs[0].distance == pytest.approx(0.0)

    def test_nan_rows_never_returned(self):
        states, dates = self.make_states()
        states[:150] = np.nan
        analogs = retrieve_analogs(states, dates, len(dates) - 1, k=10)
        assert analogs
        assert all(a.index >= 150 for a in analogs)

    def test_missing_query_raises(self):
        states, dates = self.make_states()
        states[-1, 0] = np.nan
        with pytest.raises(ValueError, match="missing features"):
            retrieve_analogs(states, dates, len(dates) - 1)

    def test_short_history_returns_empty(self):
        states, dates = self.make_states(n=50)
        assert retrieve_analogs(states, dates, 49, exclude_recent=63) == []


class TestOutcomes:
    def test_constant_return_window(self):
        dates = make_dates(60)
        prices = [100.0 * 1.01**i for i in range(60)]
        panel = Panel(dates, {"SPY": prices}, [0.0] * 60)
        out = forward_outcomes(panel, 10, horizon=21)
        assert out is not None
        assert out["fwd_return"] == pytest.approx(1.01**21 - 1.0)
        assert out["fwd_vol"] == pytest.approx(np.sqrt(0.01**2 * 252))
        assert out["fwd_max_drawdown"] == pytest.approx(0.0)

    def test_incomplete_window_is_none(self):
        dates = make_dates(30)
        panel = Panel(dates, {"SPY": [100.0 + i for i in range(30)]}, [0.0] * 30)
        assert forward_outcomes(panel, 20, horizon=21) is None


class TestBriefing:
    def test_end_to_end(self, tmp_path):
        dates = make_dates()
        store = make_store(tmp_path, dates)
        panel = make_panel(dates)
        states = build_state_matrix(panel, store)
        query = panel.n_days - 1
        analogs = retrieve_analogs(states, panel.dates, query)
        assert analogs
        briefing = build_briefing(panel, states, query, analogs)
        assert briefing["query_date"] == dates[-1]
        assert briefing["prior"] is not None
        text = format_briefing(briefing)
        assert dates[-1] in text
        assert "analog prior" in text
        assert "explanation only" in text
