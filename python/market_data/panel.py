"""Assemble date-aligned total-return panels from stored chart snapshots."""

from __future__ import annotations

from pathlib import Path

from market_data.store import read_snapshot


def load_total_return_panel(store_dir: Path, symbols: tuple[str, ...]) -> dict:
    """Union-of-dates panel of adjusted closes; None before a symbol's
    inception or on its non-trading days. Consumers decide their own alignment."""
    per_symbol: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        doc = read_snapshot(store_dir, "charts", symbol)
        if doc is None:
            raise FileNotFoundError(
                f"no stored chart for {symbol}; run `python -m market_data fetch`"
            )
        per_symbol[symbol] = {
            r["date"]: r["adj_close"] for r in doc["data"]["rows"]
        }
    dates = sorted(set().union(*per_symbol.values())) if per_symbol else []
    return {
        "dates": dates,
        "adj_close": {
            symbol: [series.get(d) for d in dates]
            for symbol, series in per_symbol.items()
        },
    }
