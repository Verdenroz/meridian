"""Storage invariants for the live tone store: append-only, id-deduped,
point-in-time dated — bugs here silently poison the OOS record."""

from __future__ import annotations

import json
from pathlib import Path

from tone_live import live


def _archive(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "feeds"
    d.mkdir()
    (d / "2026-08-01.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return d


def _fake_scoring(monkeypatch, tone: float = -0.5) -> None:
    monkeypatch.setattr(live, "score_missing", lambda texts: len(texts))
    monkeypatch.setattr(live, "read_cached_tone", lambda text: tone)


ROWS = [
    {
        "id": "aaa",
        "fetched_at": "2026-08-01T14:00:00+00:00",
        "source": "Bloomberg",
        "title": "Stocks slide",
    },
    {
        "id": "bbb",
        "fetched_at": "2026-08-01T02:00:00+00:00",
        "source": "CNBC",
        "title": "Rally continues",
    },
    # Non-whitelisted and untitled rows must never enter the store.
    {
        "id": "ccc",
        "fetched_at": "2026-08-01T03:00:00+00:00",
        "source": "South China Morning Post",
        "title": "General news",
    },
    {"id": "ddd", "fetched_at": "2026-08-01T04:00:00+00:00", "source": "WSJ"},
]


def test_score_new_filters_orders_and_dedupes(tmp_path, monkeypatch):
    _fake_scoring(monkeypatch)
    archive = _archive(tmp_path, ROWS)
    store = tmp_path / "scores.jsonl"

    assert live.score_new(archive, store) == 2
    lines = [json.loads(x) for x in store.read_text().splitlines()]
    assert [r["id"] for r in lines] == ["bbb", "aaa"]  # fetched_at order
    assert all(r["date"] == "2026-08-01" for r in lines)
    assert all(r["model"] == live.SENTIMENT_MODEL for r in lines)

    # Idempotent: a second run appends nothing and edits nothing.
    before = store.read_text()
    assert live.score_new(archive, store) == 0
    assert store.read_text() == before


def test_daily_tone_aggregates_by_fetched_date(tmp_path, monkeypatch):
    _fake_scoring(monkeypatch, tone=-0.5)
    archive = _archive(tmp_path, ROWS[:2])
    store = tmp_path / "scores.jsonl"
    live.score_new(archive, store)
    days = live.daily_tone(store)
    assert days == {"2026-08-01": (2, -0.5)}
