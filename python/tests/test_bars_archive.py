from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from bars_archive.__main__ import archive_symbol
from bars_archive.fetch import (
    ET,
    FetchError,
    by_session,
    completed,
    parse_chart,
    windows,
)
from bars_archive.store import read_session, session_path, write_session


def _payload(interval: str, stamps: list[int]) -> dict:
    nodes = [
        {"timestamp": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10}
        for t in stamps
    ]
    return {"data": {"ticker": {"chart": {"interval": interval, "candles": {"nodes": nodes}}}}}


def _et(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(UTC)


class TestStore:
    def test_an_archived_session_is_never_rewritten(self, tmp_path: Path) -> None:
        path = session_path(tmp_path, "1m", date(2026, 9, 28), "SPY")
        assert write_session(path, [{"ts": 1, "c": 1.0}])
        assert not write_session(path, [{"ts": 1, "c": 2.0}])
        assert read_session(path) == [{"ts": 1, "c": 1.0}]

    def test_identical_rows_give_identical_bytes(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
        write_session(a, [{"ts": 1}])
        write_session(b, [{"ts": 1}])
        assert a.read_bytes() == b.read_bytes()


class TestFetch:
    def test_a_daily_payload_for_a_minute_request_is_rejected(self) -> None:
        with pytest.raises(FetchError):
            parse_chart(_payload("1d", [0]), "1m")

    def test_graphql_errors_raise(self) -> None:
        with pytest.raises(FetchError, match="HTTP 422"):
            parse_chart({"errors": [{"message": "HTTP 422"}]}, "1m")

    def test_windows_start_at_et_midnight_and_respect_the_chunk(self) -> None:
        now = _et(2026, 9, 29, 17, 40)
        spans = windows("1m", now)
        first = datetime.fromtimestamp(spans[0][0], UTC).astimezone(ET)
        assert (first.date(), first.hour, first.minute) == (date(2026, 9, 1), 0, 0)
        assert all(e - s <= timedelta(days=7).total_seconds() for s, e in spans)
        assert spans[-1][1] == int(now.timestamp())

    def test_sessions_group_by_exchange_date(self) -> None:
        late = int(_et(2026, 9, 28, 15, 59).timestamp())
        early = int(_et(2026, 9, 29, 9, 30).timestamp())
        sessions = by_session([{"ts": late}, {"ts": early}])
        assert list(sessions) == [date(2026, 9, 28), date(2026, 9, 29)]

    def test_the_session_in_progress_is_not_completed(self) -> None:
        today = date(2026, 9, 29)
        assert not completed(today, _et(2026, 9, 29, 15, 0))
        assert not completed(today, _et(2026, 9, 29, 16, 10))
        assert completed(today, _et(2026, 9, 29, 16, 30))
        assert completed(date(2026, 9, 28), _et(2026, 9, 29, 9, 0))


class TestArchive:
    def test_only_completed_sessions_are_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamps = [int(_et(2026, 9, 28, 10, 0).timestamp()), int(_et(2026, 9, 29, 10, 0).timestamp())]
        monkeypatch.setattr(
            "bars_archive.fetch._post_graphql", lambda *_: _payload("1m", stamps)
        )
        now = _et(2026, 9, 29, 12, 0)
        assert archive_symbol(tmp_path, "http://x", "SPY", "1m", now) == 1
        assert session_path(tmp_path, "1m", date(2026, 9, 28), "SPY").exists()
        assert not session_path(tmp_path, "1m", date(2026, 9, 29), "SPY").exists()
        assert archive_symbol(tmp_path, "http://x", "SPY", "1m", now) == 0
