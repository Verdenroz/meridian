"""Tests for the shared race framework: gate routing, serialization, stats."""

from typing import cast

import numpy as np
from deep_eightk.stats import BLOCK_LEN, N_RESAMPLES, SEED
from deep_eightk.stats import sharpe_diff_bootstrap as frozen_bootstrap
from harness.gates import amended_bar
from harness.panel import Panel
from harness.racing import metrics_table, plain
from harness.stats import sharpe_diff_bootstrap

GATE = {"noninferiority_margin": 0.05, "maxdd_ratio": 2.0 / 3.0}
PANEL = cast(Panel, None)  # amended_bar only forwards it to the placebo run


def _entry(sharpe: float, maxdd: float, ann_vol: float = 0.10) -> dict:
    return {"metrics": {"sharpe": sharpe, "max_drawdown": maxdd, "ann_vol": ann_vol}}


class TestAmendedBar:
    def test_outright_sharpe_win_passes_route_a(self):
        v = amended_bar(_entry(0.60, 0.30), _entry(0.50, 0.20), PANEL, 0, GATE)
        assert v["passes"] and v["route"] == "a"
        assert v["co_criterion"] is None, "route (a) must not run the placebo"

    def test_inferior_sharpe_fails_without_placebo(self):
        v = amended_bar(_entry(0.40, 0.10), _entry(0.50, 0.30), PANEL, 0, GATE)
        assert not v["passes"]
        assert v["co_criterion"]["noninferior_sharpe"] is False
        assert v["co_criterion"]["placebo"] is None

    def test_weak_tail_fails_without_placebo(self):
        v = amended_bar(_entry(0.49, 0.25), _entry(0.50, 0.30), PANEL, 0, GATE)
        assert not v["passes"]
        assert v["co_criterion"]["tail_two_thirds"] is False
        assert v["co_criterion"]["placebo"] is None

    def test_b_route_passes_only_past_the_placebo(self):
        def losing_placebo(panel, factory, first, oos_start=None):
            return _entry(0.45, 0.15)

        v = amended_bar(
            _entry(0.48, 0.10, ann_vol=0.08),
            _entry(0.50, 0.30, ann_vol=0.12),
            PANEL, 0, GATE, run=losing_placebo,
        )
        assert v["passes"] and str(v["route"]).startswith("b")
        assert v["co_criterion"]["beats_placebo"] is True
        assert v["co_criterion"]["placebo"]["scale"] == 0.08 / 0.12

    def test_b_route_blocked_by_winning_placebo(self):
        def winning_placebo(panel, factory, first, oos_start=None):
            return _entry(0.55, 0.15)

        v = amended_bar(
            _entry(0.48, 0.10, ann_vol=0.08),
            _entry(0.50, 0.30, ann_vol=0.12),
            PANEL, 0, GATE, run=winning_placebo,
        )
        assert not v["passes"]
        assert v["co_criterion"]["beats_placebo"] is False


class TestSerialization:
    def test_plain_coerces_nested_numpy_scalars(self):
        raw = {"a": np.float64(1.5), "b": [np.bool_(True), (np.int64(2),)]}
        out = plain(raw)
        assert out == {"a": 1.5, "b": [True, [2]]}
        assert type(out["a"]) is float and type(out["b"][0]) is bool

    def test_metrics_table_renders_all_rows(self):
        table = metrics_table(
            {"null": {"sharpe": 0.5}, "challenger": {"sharpe": 0.6}},
            columns=("sharpe",),
        )
        lines = table.splitlines()
        assert len(lines) == 3 and lines[0].endswith("sharpe")
        assert "0.6000" in lines[2]


class TestSharpeDiffBootstrap:
    def test_seeded_and_frozen_wrapper_matches_shared_mechanics(self):
        rng = np.random.default_rng(7)
        chall = rng.normal(0.0005, 0.01, 300)
        null = rng.normal(0.0004, 0.01, 300)
        cash = np.full(300, 0.0001)
        a = frozen_bootstrap(chall, null, cash)
        b = sharpe_diff_bootstrap(
            chall, null, cash,
            block_len=BLOCK_LEN, n_resamples=N_RESAMPLES, seed=SEED,
        )
        assert a == b, "frozen wrapper must be byte-identical to the mechanics"
        assert 0.0 <= a["p_leq_0"] <= 1.0
