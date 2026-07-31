"""Invariant tests for the shadow job: log idempotency, staleness, rows."""

from datetime import date
from pathlib import Path

import pytest
from harness.panel import Panel
from harness.view import HistoryView
from shadow.job import check_staleness, decision_row
from shadow.log import append_decision, load_decisions, logged_keys


def _panel() -> Panel:
    dates = [f"2026-07-{d:02d}" for d in (27, 28, 29, 30, 31)]
    prices: list[float | None] = [100.0, 101.0, 100.5, 102.0, 101.0]
    return Panel(dates, {"SPY": prices}, [4.0] * 5)


class _StubStrategy:
    def fit(self, view: HistoryView) -> None:
        self.fitted_at = view.t

    def target_weights(self, view: HistoryView) -> dict[str, float]:
        return {"SPY": 0.6, "TLT": 0.0}


class TestLog:
    def test_roundtrip_and_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        append_decision(path, {"decision_date": "2026-07-31", "strategy": "a"})
        append_decision(path, {"decision_date": "2026-07-31", "strategy": "b"})
        assert logged_keys(path) == {("2026-07-31", "a"), ("2026-07-31", "b")}
        assert len(load_decisions(path)) == 2

    def test_corrupt_lines_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        append_decision(path, {"decision_date": "2026-07-31", "strategy": "a"})
        path.write_text(path.read_text() + "not json\n")
        assert logged_keys(path) == {("2026-07-31", "a")}
        assert len(load_decisions(path)) == 1


class TestStaleness:
    def test_fresh_passes(self) -> None:
        check_staleness("2026-07-31", date(2026, 8, 3))

    def test_stale_refuses(self) -> None:
        with pytest.raises(RuntimeError, match="stale"):
            check_staleness("2026-07-20", date(2026, 8, 3))


class TestDecisionRow:
    def test_row_shape_and_cash_complement(self) -> None:
        panel = _panel()
        view = HistoryView(panel, panel.n_days - 1)
        row = decision_row("stub", _StubStrategy(), view, "2026-07-31T21:30:00+00:00")
        assert row["decision_date"] == "2026-07-31"
        assert row["strategy"] == "stub"
        assert row["weights"] == {"SPY": 0.6}  # zero weights dropped
        assert row["cash_weight"] == pytest.approx(0.4)

    def test_strategy_sees_latest_close_only(self) -> None:
        panel = _panel()
        view = HistoryView(panel, panel.n_days - 1)
        strategy = _StubStrategy()
        decision_row("stub", strategy, view, "2026-07-31T21:30:00+00:00")
        assert strategy.fitted_at == panel.n_days - 1
