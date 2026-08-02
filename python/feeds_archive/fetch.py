"""Fetch news/feed items from the finance-query API and normalize into rows."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

USER_AGENT = "meridian-feeds-archive/1.0"
HTTP_TIMEOUT = 30

# Tracking params stripped before hashing so republished links dedupe cleanly.
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "mod", "cmpid"}


def canonical_url(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return url.strip()
    query = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    return urllib.parse.urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            urllib.parse.urlencode(query),
            "",
        )
    )


def _get_json(base_url: str, path: str):
    url = f"{base_url}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp)


def _item_row(kind: str, source: str, symbol: str | None, payload: dict) -> dict:
    url = canonical_url(payload.get("url") or payload.get("link") or "")
    # Content hash covers only editorial fields, so recomputed sentiment or
    # image churn upstream does not create phantom "revisions".
    content = f"{payload.get('title', '')}\x1f{payload.get('summary', '')}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    # Symbol folds into the id so one article attributed to two names stays
    # two rows; symbol-less kinds keep their pre-2026-08-02 ids unchanged.
    key = f"{source}|{url}|{content_hash}" + (f"|{symbol}" if symbol else "")
    row_id = hashlib.sha256(key.encode()).hexdigest()[:32]
    return {
        "id": row_id,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "source": source,
        "symbol": symbol,
        "title": payload.get("title"),
        "url": url,
        "published": payload.get("published") or payload.get("time"),
        "payload": payload,
    }


def fetch_all(
    base_url: str, sources: tuple[str, ...], symbols: tuple[str, ...] = ()
) -> tuple[list[dict], dict]:
    """Fetch every configured endpoint. Per-endpoint failures are recorded in
    the returned stats dict, never raised — a partial sweep still archives."""
    rows: list[dict] = []
    stats: dict = {"fetched": {}, "errors": {}}

    try:
        entries = _get_json(base_url, f"/v2/feeds?sources={','.join(sources)}")
        stats["fetched"]["feeds"] = len(entries)
        rows.extend(_item_row("feed", e.get("source", "?"), None, e) for e in entries)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        stats["errors"]["feeds"] = str(exc)[:200]

    try:
        articles = _get_json(base_url, "/v2/news?count=50")
        stats["fetched"]["news"] = len(articles)
        rows.extend(_item_row("news", a.get("source", "?"), None, a) for a in articles)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        stats["errors"]["news"] = str(exc)[:200]

    sym_rows, sym_stats = fetch_symbols(base_url, symbols)
    rows.extend(sym_rows)
    stats["fetched"].update(sym_stats["fetched"])
    stats["errors"].update(sym_stats["errors"])
    return rows, stats


def fetch_symbols(
    base_url: str, symbols: tuple[str, ...], pause: float = 0.0
) -> tuple[list[dict], dict]:
    """Per-symbol news sweep only. Per-symbol failures are recorded in the
    stats dict, never raised — a partial sweep still archives. `pause`
    throttles between requests: bursts trip upstream's rate limit (~140
    unthrottled requests before every response 404s for ~90s)."""
    rows: list[dict] = []
    stats: dict = {"fetched": {}, "errors": {}}
    for i, sym in enumerate(symbols):
        if pause and i:
            time.sleep(pause)
        try:
            articles = _get_json(base_url, f"/v2/news/{urllib.parse.quote(sym)}?count=25")
            stats["fetched"][f"news:{sym}"] = len(articles)
            rows.extend(
                _item_row("news_symbol", a.get("source", "?"), sym, a) for a in articles
            )
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            stats["errors"][f"news:{sym}"] = str(exc)[:200]
    return rows, stats
