"""Invariant tests for the shadow job: log idempotency, staleness, rows."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from harness.panel import Panel
from harness.view import HistoryView
from shadow.job import check_staleness, dates_to_log, decision_row
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


DATES = ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29"]
TODAY = date(2026, 9, 29)


class TestCatchUp:

    def test_a_withheld_session_is_owed_to_contenders_on_record(self) -> None:
        logged = {("2026-09-24", "a"), ("2026-09-25", "a")}
        assert dates_to_log(DATES, logged, ["a"], TODAY) == [
            ("2026-09-28", ["a"]),
            ("2026-09-29", ["a"]),
        ]

    def test_a_new_contender_starts_at_the_latest_date_only(self) -> None:
        logged = {("2026-09-25", "a")}
        owed = dates_to_log(DATES, logged, ["a", "new"], TODAY)
        assert owed == [("2026-09-28", ["a"]), ("2026-09-29", ["a", "new"])]

    def test_sessions_beyond_the_staleness_window_stay_missing(self) -> None:
        dates = ["2026-09-18", "2026-09-21", "2026-09-29"]
        logged = {("2026-09-18", "a")}
        assert dates_to_log(dates, logged, ["a"], TODAY) == [("2026-09-29", ["a"])]

    def test_nothing_is_owed_once_logged(self) -> None:
        logged = {(d, "a") for d in DATES}
        assert dates_to_log(DATES, logged, ["a"], TODAY) == []


class TestRefresh:
    def _stub(self, monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
        def yields(_url: str, year: int) -> list[dict]:
            calls.append(year)
            return [{"date": f"06/30/{year}", "y3M": 1.0}]

        monkeypatch.setattr("shadow.job.fetch_symbol", lambda *_: None)
        monkeypatch.setattr("shadow.job.fetch_fred_series", lambda *_: [])
        monkeypatch.setattr("shadow.job.fetch_treasury_yields", yields)

    def test_a_fresh_store_gets_every_treasury_year(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from market_data.macro import TREASURY_FIRST_YEAR
        from market_data.store import read_snapshot
        from shadow.job import refresh

        calls: list[int] = []
        self._stub(monkeypatch, calls)
        refresh("http://x", tmp_path)
        doc = read_snapshot(tmp_path, "macro", "treasury_yields")
        assert doc is not None
        rows = doc["data"]["rows"]
        this_year = datetime.now(UTC).year
        assert {int(r["date"][:4]) for r in rows} == set(range(TREASURY_FIRST_YEAR, this_year + 1))

        calls.clear()
        refresh("http://x", tmp_path)
        assert calls == [datetime.now(UTC).year]
