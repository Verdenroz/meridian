"""Thin HTTP client for the finance-query v2 API (consumed read-only)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://finance-query.com"
USER_AGENT = "meridian-market-data/1.0"
HTTP_TIMEOUT = 60


def get_json(base_url: str, path: str) -> Any:
    req = urllib.request.Request(
        f"{base_url}{path}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp)


def fetch_chart(
    base_url: str, symbol: str, range_: str = "max", interval: str = "1d"
) -> dict:
    sym = urllib.parse.quote(symbol)
    data = get_json(base_url, f"/v2/chart/{sym}?range={range_}&interval={interval}")
    if not isinstance(data, dict) or "candles" not in data:
        raise TypeError(f"unexpected chart payload for {symbol}")
    return data


def fetch_dividends(base_url: str, symbol: str) -> list[dict]:
    sym = urllib.parse.quote(symbol)
    data = get_json(base_url, f"/v2/dividends/{sym}?range=max")
    if not isinstance(data, dict):
        raise TypeError(f"unexpected dividends payload for {symbol}")
    return list(data.get("dividends") or [])


def fetch_splits(base_url: str, symbol: str) -> list[dict]:
    data = get_json(base_url, f"/v2/splits/{urllib.parse.quote(symbol)}")
    if not isinstance(data, list):
        raise TypeError(f"unexpected splits payload for {symbol}")
    return data


def fetch_fred_series(base_url: str, series_id: str) -> list[dict]:
    data = get_json(base_url, f"/v2/fred/series/{urllib.parse.quote(series_id)}")
    if not isinstance(data, dict):
        raise TypeError(f"unexpected FRED payload for {series_id}")
    return list(data.get("observations") or [])


def fetch_treasury_yields(base_url: str, year: int) -> list[dict]:
    data = get_json(base_url, f"/v2/fred/treasury-yields?year={year}")
    if not isinstance(data, list):
        raise TypeError(f"unexpected treasury payload for {year}")
    return data
