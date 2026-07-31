"""Normalize upstream chart candles into clean daily rows keyed by UTC date."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_date(ts: int) -> str:
    # US sessions open 13:30/14:30 UTC, so a daily bar's UTC calendar date
    # always equals its exchange-local trading date.
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def normalize_candles(candles: list[dict]) -> tuple[list[dict], dict]:
    """Return (rows sorted by date, issues). Rows without a usable close are
    dropped and counted; a later date wins a duplicate-date collision."""
    by_date: dict[str, dict] = {}
    issues = {"dropped_no_close": 0, "duplicate_dates": 0, "missing_adj_close": 0}
    for c in candles:
        close = c.get("close")
        ts = c.get("timestamp")
        if ts is None or close is None or close <= 0:
            issues["dropped_no_close"] += 1
            continue
        adj = c.get("adjClose")
        if adj is None or adj <= 0:
            issues["missing_adj_close"] += 1
            adj = None
        date = utc_date(ts)
        if date in by_date:
            issues["duplicate_dates"] += 1
        by_date[date] = {
            "date": date,
            "open": c.get("open"),
            "high": c.get("high"),
            "low": c.get("low"),
            "close": float(close),
            "adj_close": float(adj) if adj is not None else None,
            "volume": c.get("volume"),
        }
    return [by_date[d] for d in sorted(by_date)], issues
