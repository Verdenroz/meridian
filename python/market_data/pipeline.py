"""Fetch-verify-store pipeline for one symbol (shared by CLI and consumers)."""

from __future__ import annotations

from pathlib import Path

from market_data.client import fetch_chart, fetch_dividends, fetch_splits
from market_data.crosscheck import verify_adj_close
from market_data.series import normalize_candles
from market_data.store import read_snapshot, write_snapshot


def fetch_symbol(base_url: str, store_dir: Path, symbol: str) -> dict:
    """Fetch chart+dividends+splits, cross-check adjClose (raises
    CrossCheckError on divergence), snapshot everything. Returns the report."""
    chart = fetch_chart(base_url, symbol)
    rows, issues = normalize_candles(chart["candles"])
    dividends = fetch_dividends(base_url, symbol)
    splits = fetch_splits(base_url, symbol)
    report = verify_adj_close(symbol, rows, dividends)
    write_snapshot(
        store_dir,
        "charts",
        symbol,
        {"rows": rows, "issues": issues, "crosscheck": report, "meta": chart.get("meta")},
    )
    write_snapshot(store_dir, "dividends", symbol, {"dividends": dividends})
    write_snapshot(store_dir, "splits", symbol, {"splits": splits})
    return report


def ensure_symbol(base_url: str, store_dir: Path, symbol: str) -> None:
    """Fetch a symbol's chart only if it is not already snapshotted."""
    if read_snapshot(store_dir, "charts", symbol) is None:
        fetch_symbol(base_url, store_dir, symbol)
