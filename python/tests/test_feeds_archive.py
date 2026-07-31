"""Invariant tests for the feeds archive: dedupe, append-only storage,
and index idempotency — the places where bugs silently poison data."""

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


def test_build_db_is_idempotent(tmp_path: Path) -> None:
    rows = [_item_row("news", "s", None, {"title": f"t{i}", "url": f"https://x/{i}"}) for i in range(5)]
    append_rows(tmp_path, rows)
    db = tmp_path / "idx.db"
    first = build_db(tmp_path, db)
    second = build_db(tmp_path, db)
    assert first == {"inserted": 5, "total_items": 5, "heartbeats": 0}
    assert second == {"inserted": 0, "total_items": 5, "heartbeats": 0}
