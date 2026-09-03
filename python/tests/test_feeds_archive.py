"""Invariant tests for the feeds archive: dedupe, append-only storage,
and index idempotency — the places where bugs silently poison data."""

import hashlib
import json
from pathlib import Path

from feeds_archive.fetch import _item_row, canonical_url
from feeds_archive.store import append_rows, build_db, load_seen_ids


def test_canonical_url_strips_tracking_and_normalizes() -> None:
    url = "HTTPS://Example.com/Article?utm_source=x&id=7&fbclid=abc#frag"
    assert canonical_url(url) == "https://example.com/Article?id=7"


def test_row_id_ignores_sentiment_churn() -> None:
    base = {"title": "Fed holds rates", "summary": "FOMC statement", "url": "https://a.b/c"}
    rescored = {**base, "sentiment": "negative", "score": -0.8}
    assert _item_row("news", "cnbc", None, base)["id"] == (
        _item_row("news", "cnbc", None, rescored)["id"]
    )


def test_row_id_changes_on_headline_revision() -> None:
    base = {"title": "Fed holds rates", "summary": "s", "url": "https://a.b/c"}
    revised = {**base, "title": "Fed holds rates, signals cut"}
    assert _item_row("news", "cnbc", None, base)["id"] != (
        _item_row("news", "cnbc", None, revised)["id"]
    )


def test_append_then_load_seen_ids_dedupes(tmp_path: Path) -> None:
    rows = [_item_row("news", "s", None, {"title": f"t{i}", "url": f"https://x/{i}"}) for i in range(3)]
    append_rows(tmp_path, rows)
    seen = load_seen_ids(tmp_path)
    assert seen == {r["id"] for r in rows}


def test_append_is_append_only(tmp_path: Path) -> None:
    first = [_item_row("news", "s", None, {"title": "one", "url": "https://x/1"})]
    second = [_item_row("news", "s", None, {"title": "two", "url": "https://x/2"})]
    day_file = append_rows(tmp_path, first)
    before = day_file.read_text()
    append_rows(tmp_path, second)
    assert day_file.read_text().startswith(before)


def test_symbol_folds_into_row_id() -> None:
    payload = {"title": "Apple event", "url": "https://a.b/apple"}
    ids = {
        _item_row("news_symbol", "cnbc", sym, payload)["id"]
        for sym in ("AAPL", "MSFT", None)
    }
    assert len(ids) == 3


def test_symbolless_row_id_matches_pre_pername_scheme() -> None:
    # The live archive's ids must not churn: symbol-less rows keep the
    # original source|url|content formula exactly.
    payload = {"title": "Fed holds rates", "summary": "s", "url": "https://a.b/c"}
    content_hash = hashlib.sha256(b"Fed holds rates\x1fs").hexdigest()[:16]
    legacy = hashlib.sha256(f"cnbc|https://a.b/c|{content_hash}".encode()).hexdigest()[:32]
    assert _item_row("news", "cnbc", None, payload)["id"] == legacy


def test_load_seen_ids_lookback_window(tmp_path: Path) -> None:
    old = tmp_path / "2020-01-01.jsonl"
    old.write_text(json.dumps({"id": "old-id"}) + "\n")
    new = tmp_path / "2026-01-01.jsonl"
    new.write_text(json.dumps({"id": "new-id"}) + "\n")
    assert load_seen_ids(tmp_path, 1) == {"new-id"}
    assert load_seen_ids(tmp_path, 60) == {"old-id", "new-id"}


def test_pername_universe_file_is_clean() -> None:
    from feeds_archive.__main__ import DEFAULT_PERNAME_UNIVERSE, _read_symbols

    symbols = _read_symbols(str(DEFAULT_PERNAME_UNIVERSE))
    assert len(symbols) == len(set(symbols)) > 400
    assert all(s == s.strip().upper() for s in symbols)


def test_build_db_is_idempotent(tmp_path: Path) -> None:
    rows = [_item_row("news", "s", None, {"title": f"t{i}", "url": f"https://x/{i}"}) for i in range(5)]
    append_rows(tmp_path, rows)
    db = tmp_path / "idx.db"
    first = build_db(tmp_path, db)
    second = build_db(tmp_path, db)
    assert first == {"inserted": 5, "total_items": 5, "heartbeats": 0}
    assert second == {"inserted": 0, "total_items": 5, "heartbeats": 0}


def test_pername_sweep_stops_at_time_budget(monkeypatch) -> None:
    import time

    import feeds_archive.__main__ as m

    calls: list[tuple[str, ...]] = []

    def fake_fetch(base_url: str, symbols: tuple[str, ...], pause: float) -> tuple[list, dict]:
        calls.append(symbols)
        return [], {"fetched": {f"news:{s}": 1 for s in symbols}, "errors": {}}

    monkeypatch.setattr(m, "fetch_symbols", fake_fetch)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    symbols = tuple(f"S{i}" for i in range(m.PERNAME_CHUNK_SIZE + 3))
    _, stats = m._chunked_sweep("http://x", symbols, time.monotonic() + 1)
    assert calls == [symbols[: m.PERNAME_CHUNK_SIZE]]
    assert set(stats["errors"]) == {f"news:{s}" for s in symbols[m.PERNAME_CHUNK_SIZE :]}
    assert all(v == "skipped: time budget" for v in stats["errors"].values())


def test_pername_rotation_shifts_without_loss() -> None:
    from feeds_archive.__main__ import _rotated

    symbols = ("A", "B", "C", "D")
    assert _rotated(symbols, 1) == ("B", "C", "D", "A")
    assert _rotated(symbols, 6) == ("C", "D", "A", "B")
    assert set(_rotated(symbols, 99)) == set(symbols)
    assert _rotated((), 3) == ()
