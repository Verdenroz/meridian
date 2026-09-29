"""Intraday bars from finance-query's GraphQL chart field.

REST ``/v2/chart`` drops ``interval`` when given ``start``/``end`` and
returns daily bars, so windows go through ``POST /graphql``. Yahoo refuses
1m windows longer than about a week, so 1m sweeps are chunked.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

USER_AGENT = "meridian-bars-archive/1.0"
HTTP_TIMEOUT = 60
ET = ZoneInfo("America/New_York")

GQL_INTERVAL = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTES"}
# Upstream lookback is 29 days (1m) and 58 (5m); starting a day inside it at
# ET midnight keeps every fetched session whole.
LOOKBACK_DAYS = {"1m": 28, "5m": 57}
CHUNK_DAYS = {"1m": 7, "5m": 57}

_QUERY = (
    '{ ticker(symbol: "%s") { chart(interval: %s, start: %d, end: %d) '
    "{ interval candles { nodes { timestamp open high low close volume } } } } }"
)


class FetchError(RuntimeError):
    pass


def _post_graphql(base_url: str, query: str) -> dict:
    req = urllib.request.Request(
        f"{base_url}/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp)


def parse_chart(payload: dict, interval: str) -> list[dict]:
    if payload.get("errors"):
        raise FetchError(payload["errors"][0].get("message", "graphql error"))
    chart = payload["data"]["ticker"]["chart"]
    if chart["interval"] != interval:
        raise FetchError(f"asked for {interval} bars, got {chart['interval']}")
    return [
        {
            "ts": n["timestamp"],
            "o": n["open"],
            "h": n["high"],
            "l": n["low"],
            "c": n["close"],
            "v": n["volume"],
        }
        for n in chart["candles"]["nodes"]
    ]


def windows(interval: str, now: datetime) -> list[tuple[int, int]]:
    today = now.astimezone(ET).date()
    start = datetime.combine(today - timedelta(days=LOOKBACK_DAYS[interval]), time(), ET)
    end = int(now.timestamp())
    step = int(timedelta(days=CHUNK_DAYS[interval]).total_seconds())
    edges = range(int(start.timestamp()), end, step)
    return [(s, min(s + step, end)) for s in edges]


def fetch_bars(base_url: str, symbol: str, interval: str, now: datetime) -> list[dict]:
    bars: dict[int, dict] = {}
    for start, end in windows(interval, now):
        query = _QUERY % (symbol, GQL_INTERVAL[interval], start, end)
        for bar in parse_chart(_post_graphql(base_url, query), interval):
            bars[bar["ts"]] = bar
    return [bars[ts] for ts in sorted(bars)]


def by_session(bars: list[dict]) -> dict[date, list[dict]]:
    sessions: dict[date, list[dict]] = {}
    for bar in bars:
        day = datetime.fromtimestamp(bar["ts"], UTC).astimezone(ET).date()
        sessions.setdefault(day, []).append(bar)
    return sessions


def completed(session: date, now: datetime) -> bool:
    """Upstream revises the last bars shortly after the bell; a session counts
    as closed only half an hour after it."""
    local = now.astimezone(ET)
    return session < local.date() or (session == local.date() and local.time() >= time(16, 30))
