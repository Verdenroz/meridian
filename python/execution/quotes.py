"""Quotes from finance-query /v2/quotes.

Outside regular hours the feed carries no bid/ask; the touch is then
synthesized from the last price and the harness cost table so paper fills
pay the same spread the backtest charged.
"""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime

from harness.costs import one_way_cost
from market_data.client import get_json

from execution.orders import Quote


def fetch_quotes(base_url: str, symbols: list[str]) -> dict[str, Quote]:
    joined = urllib.parse.quote(",".join(symbols))
    data = get_json(base_url, f"/v2/quotes?symbols={joined}")
    if not isinstance(data, dict) or "quotes" not in data:
        raise TypeError("unexpected quotes payload")
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    quotes = {q["symbol"]: quote_from_payload(q, fetched_at) for q in data["quotes"]}
    missing = [s for s in symbols if s not in quotes]
    if missing:
        raise RuntimeError(f"no quote for {', '.join(missing)}")
    return quotes


def quote_from_payload(payload: dict, fetched_at: str) -> Quote:
    symbol = payload["symbol"]
    last = float(payload["regularMarketPrice"])
    spread = one_way_cost(symbol)
    bid = payload.get("bid") or last * (1 - spread)
    ask = payload.get("ask") or last * (1 + spread)
    return Quote(
        symbol=symbol,
        bid=float(bid),
        ask=float(ask),
        last=last,
        market_state=str(payload.get("marketState") or "UNKNOWN"),
        fetched_at=fetched_at,
        prev_close=payload.get("regularMarketPreviousClose"),
    )
