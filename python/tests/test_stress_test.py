from __future__ import annotations

from pathlib import Path

import pytest
from harness.panel import Panel
from stress_test import GATE
from stress_test.__main__ import record_once
from stress_test.evaluate import book_returns, find_episode, path_stats, verdict

SMALL = {**GATE, "start": "d03", "lookback": 5, "max_sessions": 3, "trigger_drawdown": 0.10}


def _dates(n: int) -> list[str]:
    return [f"d{i:02d}" for i in range(n)]


class TestEpisode:
    def test_the_first_trigger_after_start_opens_the_window_at_the_high(self) -> None:
        px = [100.0, 105.0, 104.0, 99.0, 94.0, 96.0, 106.0, 107.0]
        ep = find_episode(_dates(8), px, SMALL)
        assert ep is not None
        assert (ep.peak, ep.trigger, ep.end, ep.closed) == (1, 4, 6, True)

    def test_a_trigger_before_start_is_ignored(self) -> None:
        px = [100.0, 89.0, 90.0, 91.0, 92.0]
        assert find_episode(_dates(5), px, SMALL) is None

    def test_an_unrecovered_episode_closes_at_the_session_cap(self) -> None:
        px = [100.0, 100.0, 100.0, 100.0, 89.0, 88.0, 87.0, 86.0, 85.0]
        ep = find_episode(_dates(9), px, SMALL)
        assert ep is not None and (ep.end, ep.closed) == (7, True)

    def test_an_episode_still_inside_its_cap_is_open(self) -> None:
        px = [100.0, 100.0, 100.0, 100.0, 89.0, 88.0]
        ep = find_episode(_dates(6), px, SMALL)
        assert ep is not None and not ep.closed


class TestBooks:
    def _panel(self) -> Panel:
        return Panel(_dates(4), {"SPY": [100.0, 110.0, 99.0, 99.0]}, [0.0] * 4)

    def test_missing_rows_carry_the_prior_weights(self) -> None:
        rows = [{"decision_date": "d00", "strategy": "s", "weights": {"SPY": 1.0}}]
        rets = book_returns(rows, "s", self._panel(), 1, 3)
        assert rets == pytest.approx([-0.10, 0.0])

    def test_a_weight_change_pays_the_one_way_cost(self) -> None:
        rows = [
            {"decision_date": "d00", "strategy": "s", "weights": {"SPY": 1.0}},
            {"decision_date": "d01", "strategy": "s", "weights": {}},
        ]
        (ret,) = book_returns(rows, "s", self._panel(), 1, 2)
        assert ret == pytest.approx(-1.0e-4)

    def test_path_stats(self) -> None:
        stats = path_stats([0.10, -0.10])
        assert stats["total_return"] == pytest.approx(-0.01)
        assert stats["max_drawdown"] == pytest.approx(0.10)


class TestVerdict:
    def test_cutting_the_fall_but_missing_the_recovery_fails(self) -> None:
        null = {"max_drawdown": 0.15, "total_return": 0.0}
        timid = {"max_drawdown": 0.05, "total_return": -0.03}
        assert verdict(timid, null) == {"tail": True, "timing": False, "passes": False}

    def test_a_better_tail_within_tolerance_passes(self) -> None:
        null = {"max_drawdown": 0.15, "total_return": 0.0}
        leader = {"max_drawdown": 0.09, "total_return": -0.02}
        assert verdict(leader, null)["passes"]

    def test_the_verdict_is_recorded_once(self, tmp_path: Path) -> None:
        path = tmp_path / "stress.jsonl"
        assert not record_once(path, {"status": "open"})
        assert record_once(path, {"status": "judged"})
        assert not record_once(path, {"status": "judged"})
        assert len(path.read_text().splitlines()) == 1
