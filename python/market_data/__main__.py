"""CLI: fetch historical market/macro data, verify adjClose, report status.

Usage (from repo root):
    PYTHONPATH=python python -m market_data fetch
    PYTHONPATH=python python -m market_data verify
    PYTHONPATH=python python -m market_data status
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from market_data import DEFAULT_FRED_SERIES, DEFAULT_SYMBOLS
from market_data.client import (
    DEFAULT_BASE_URL,
    fetch_fred_series,
    fetch_treasury_yields,
)
from market_data.crosscheck import CrossCheckError, verify_adj_close
from market_data.macro import TREASURY_FIRST_YEAR, normalize_fred, normalize_treasury
from market_data.pipeline import fetch_symbol as _fetch_symbol
from market_data.store import (
    DEFAULT_STORE_DIR,
    list_snapshots,
    read_snapshot,
    write_snapshot,
)


def _fetch_macro(base_url: str, store_dir: Path, fred_ids: tuple[str, ...]) -> None:
    for series_id in fred_ids:
        rows = normalize_fred(fetch_fred_series(base_url, series_id))
        write_snapshot(store_dir, "macro", series_id, {"rows": rows})
        print(f"  macro {series_id}: {len(rows)} rows ({rows[0]['date']}..{rows[-1]['date']})")
    this_year = datetime.now(UTC).year
    yields: list[dict] = []
    for year in range(TREASURY_FIRST_YEAR, this_year + 1):
        yields.extend(fetch_treasury_yields(base_url, year))
    rows = normalize_treasury(yields)
    write_snapshot(store_dir, "macro", "treasury_yields", {"rows": rows})
    print(f"  macro treasury_yields: {len(rows)} rows ({rows[0]['date']}..{rows[-1]['date']})")


def cmd_fetch(args: argparse.Namespace) -> int:
    failures = 0
    for symbol in args.symbols:
        try:
            report = _fetch_symbol(args.base_url, args.store_dir, symbol)
            print(
                f"  {symbol}: {report['rows']} rows, "
                f"{report['adjustment_events']} adjustment events, "
                f"{len(report['mismatches'])} tolerated mismatches"
            )
        except CrossCheckError as exc:
            failures += 1
            print(f"  {symbol}: CROSS-CHECK FAILED — {exc}", file=sys.stderr)
    if not args.no_macro:
        _fetch_macro(args.base_url, args.store_dir, args.fred)
    return 1 if failures else 0


def cmd_verify(args: argparse.Namespace) -> int:
    failures = 0
    for symbol in args.symbols:
        chart = read_snapshot(args.store_dir, "charts", symbol)
        divs = read_snapshot(args.store_dir, "dividends", symbol)
        if chart is None or divs is None:
            print(f"  {symbol}: not in store", file=sys.stderr)
            failures += 1
            continue
        try:
            report = verify_adj_close(
                symbol, chart["data"]["rows"], divs["data"]["dividends"]
            )
            print(
                f"  {symbol}: ok ({report['adjustment_events']} events, "
                f"{len(report['mismatches'])} tolerated mismatches)"
            )
        except CrossCheckError as exc:
            failures += 1
            print(f"  {symbol}: CROSS-CHECK FAILED — {exc}", file=sys.stderr)
    return 1 if failures else 0


def cmd_status(args: argparse.Namespace) -> int:
    snapshots = list_snapshots(args.store_dir)
    if not snapshots:
        print(f"store empty: {args.store_dir}")
        return 0
    for snap in snapshots:
        if "error" in snap:
            print(f"  {snap['kind']}/{snap['name']}: UNREADABLE")
            continue
        doc = read_snapshot(args.store_dir, snap["kind"], snap["name"])
        data = doc["data"] if doc else {}
        rows = data.get("rows") or data.get("dividends") or data.get("splits") or []
        span = f" ({rows[0]['date']}..{rows[-1]['date']})" if rows and "date" in rows[0] else ""
        print(f"  {snap['kind']}/{snap['name']}: {len(rows)} rows{span}, fetched {snap['fetched_at']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="market_data")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument(
        "--symbols",
        type=lambda s: tuple(x.strip().upper() for x in s.split(",") if x.strip()),
        default=DEFAULT_SYMBOLS,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="fetch charts+dividends+splits+macro, verify, store")
    fetch.add_argument("--no-macro", action="store_true")
    fetch.add_argument(
        "--fred",
        type=lambda s: tuple(x.strip().upper() for x in s.split(",") if x.strip()),
        default=DEFAULT_FRED_SERIES,
    )
    fetch.set_defaults(func=cmd_fetch)
    verify = sub.add_parser("verify", help="re-run adjClose cross-check on stored data")
    verify.set_defaults(func=cmd_verify)
    status = sub.add_parser("status", help="list stored snapshots")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
