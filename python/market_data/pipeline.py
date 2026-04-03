"""Training corpus pipeline — orchestrates data collection across the symbol universe.

Usage:

  # Collect daily history for all 100 symbols (parallel, 8 workers)
  python -m market_data.pipeline collect \\
      --symbols-file symbols.txt \\
      --interval 1d \\
      --start 2010-01-01 \\
      --data-dir data/mmaps \\
      --workers 8

  # Print a plan manifest (dims, sources, row estimates) without collecting
  python -m market_data.pipeline plan --symbols-file symbols.txt

  # Print shard counts per split
  python -m market_data.pipeline stats --data-dir data/mmaps
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env so FRED_API_KEY and other secrets are available to subprocesses
load_dotenv()

from train.feature_schema import (
    CALENDAR_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    PER_ASSET_FEATURE_DIM,
    PORTFOLIO_STATE_DIM,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    symbols_file: Path = Path("symbols.txt")
    data_dir: Path = Path("data/mmaps")
    reports_dir: Path = Path("reports/ingestion")
    interval: str = "1d"
    start: str | None = "2010-01-01"
    end: str | None = None
    workers: int = 4
    binary: Path | None = None


def load_symbols(symbols_file: Path) -> list[str]:
    """Load newline-separated symbols from file, stripping comments and blanks."""
    symbols = []
    for line in symbols_file.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            symbols.append(line)
    return symbols


# ── Plan ──────────────────────────────────────────────────────────────────────

# Minutes per trading year (250 days × 390 min/day)
_MINUTES_PER_YEAR = 250 * 390
_HISTORY_YEARS = 5


def build_plan(symbols: list[str]) -> dict:
    """Return a manifest describing the data pipeline for the given symbols."""
    block_dims = {
        "price_volume":        10,
        "technical":           42,
        "options":             12,
        "fundamental":         30,
        "risk_analytics":       8,
        "event_state":         10,
        "cross_asset_context":  8,
        "text":                64,
    }
    per_asset_rows = _MINUTES_PER_YEAR * _HISTORY_YEARS
    return {
        "n_symbols":             len(symbols),
        "per_asset_feature_dim": PER_ASSET_FEATURE_DIM,
        "market_feature_dim":    MARKET_FEATURE_DIM,
        "portfolio_state_dim":   PORTFOLIO_STATE_DIM,
        "calendar_feature_dim":  CALENDAR_FEATURE_DIM,
        "shard_row_dim":         PER_ASSET_FEATURE_DIM,
        "blocks":                block_dims,
        "data_sources": [
            "finance-query: Ticker.chart() 1-minute OHLCV",
            "finance-query: Ticker.indicators() — 42 technical indicators",
            "finance-query: Ticker.options() — daily chain snapshots",
            "finance-query: Ticker.financials() + key_stats() — XBRL fundamentals",
            "finance-query: Ticker.risk() — daily risk analytics",
            "finance-query: Ticker.insider_transactions() — event reconstruction",
            "finance-query: Ticker.grading_history() — analyst events",
            "finance-query: edgar::submissions() — SEC filing events",
            "finance-query: fred::treasury_yields() — yield curve (daily)",
            "finance-query: fred::series() × 12 — macro series (various)",
            "finance-query: finance::fear_and_greed() — fear/greed index",
            "finance-query: finance::sector() × 11 — GICS sector returns",
            "finance-query: crypto::coin() — BTC/ETH market data",
            "finance-query: feeds::fetch_all() — 31 RSS sources",
        ],
        "estimated_rows_per_symbol": per_asset_rows,
        "estimated_total_rows":      per_asset_rows * len(symbols),
        "estimated_shard_bytes_per_symbol": per_asset_rows * PER_ASSET_FEATURE_DIM * 4,
        "estimated_total_gb": (
            per_asset_rows * PER_ASSET_FEATURE_DIM * 4 * len(symbols) / 1e9
        ),
        "symbols": symbols,
    }


# ── Collect ───────────────────────────────────────────────────────────────────

def _collect_one(symbol: str, cfg: PipelineConfig) -> tuple[str, bool, str]:
    """Collect one symbol. Returns (symbol, success, message)."""
    from market_data.finance_query_collector import collect_symbol_to_dir

    try:
        collect_symbol_to_dir(
            symbol=symbol,
            data_dir=cfg.data_dir,
            interval=cfg.interval,
            start=cfg.start,
            end=cfg.end,
            binary=cfg.binary,
            reports_dir=cfg.reports_dir,
        )
        return symbol, True, "ok"
    except Exception as exc:
        return symbol, False, str(exc)


def collect_universe(cfg: PipelineConfig) -> None:
    """Collect historical data for all symbols in cfg.symbols_file.

    Runs up to cfg.workers symbols in parallel via a ThreadPoolExecutor.
    Collection is I/O bound (network), so threading is appropriate here.
    """
    symbols = load_symbols(cfg.symbols_file)
    if not symbols:
        print("[pipeline] no symbols found in", cfg.symbols_file, file=sys.stderr)
        return

    print(
        f"[pipeline] collecting {len(symbols)} symbols  "
        f"interval={cfg.interval}  workers={cfg.workers}",
        file=sys.stderr,
    )

    failed: list[str] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(_collect_one, sym, cfg): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, ok, msg = fut.result()
            completed += 1
            status = "ok" if ok else f"FAILED: {msg}"
            print(
                f"[pipeline] {completed}/{len(symbols)}  {sym}  {status}",
                file=sys.stderr,
            )
            if not ok:
                failed.append(sym)

    if failed:
        print(
            f"[pipeline] {len(failed)} symbol(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("[pipeline] all symbols collected successfully", file=sys.stderr)


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(data_dir: Path) -> None:
    """Print per-split shard counts and total row estimates."""
    from market_data.dataset_ingestion import enumerate_shards, shard_row_count

    for split in ("train", "validation", "test"):
        shards = enumerate_shards(data_dir, split)
        if not shards:
            print(f"  {split:12s}  0 shards")
            continue
        total_rows = sum(shard_row_count(p) for p in shards)
        print(f"  {split:12s}  {len(shards):4d} shards  {total_rows:>12,} rows")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Meridian training corpus pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── plan ──
    p_plan = sub.add_parser("plan", help="Print pipeline plan manifest (no collection)")
    p_plan.add_argument("--symbols-file", type=Path, default=Path("symbols.txt"))

    # ── collect ──
    p_collect = sub.add_parser("collect", help="Collect historical data for all symbols")
    p_collect.add_argument("--symbols-file", type=Path, default=Path("symbols.txt"))
    p_collect.add_argument("--data-dir",     type=Path, default=Path("data/mmaps"))
    p_collect.add_argument("--reports-dir",  type=Path, default=Path("reports/ingestion"))
    p_collect.add_argument("--interval",     default="1d", choices=["1m", "5m", "1h", "1d"])
    p_collect.add_argument("--start",        default="2010-01-01")
    p_collect.add_argument("--end",          default=None)
    p_collect.add_argument("--workers",      type=int, default=4)
    p_collect.add_argument("--binary",       type=Path, default=None)

    # ── stats ──
    p_stats = sub.add_parser("stats", help="Print shard counts per split")
    p_stats.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))

    args = parser.parse_args()

    if args.command == "plan":
        symbols = load_symbols(args.symbols_file)
        print(_json.dumps(build_plan(symbols), indent=2))

    elif args.command == "collect":
        cfg = PipelineConfig(
            symbols_file=args.symbols_file,
            data_dir=args.data_dir,
            reports_dir=args.reports_dir,
            interval=args.interval,
            start=args.start,
            end=args.end,
            workers=args.workers,
            binary=args.binary,
        )
        collect_universe(cfg)

    elif args.command == "stats":
        print_stats(args.data_dir)


if __name__ == "__main__":
    _cli()
