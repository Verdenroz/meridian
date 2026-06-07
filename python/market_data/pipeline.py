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
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass, replace
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
    start: str | None = None
    end: str | None = None
    workers: int = 4
    binary: Path | None = None
    max_retries: int = 5
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 60.0
    retry_jitter_seconds: float = 1.0
    request_permits: int | None = None
    skip_existing: bool = False  # skip symbols whose shards already exist


def load_symbols(symbols_file: Path) -> list[str]:
    """Load newline-separated symbols from file, stripping comments and blanks."""
    symbols = []
    for line in symbols_file.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            symbols.append(line)
    return symbols


# ── Plan ──────────────────────────────────────────────────────────────────────

# v4.0 uses daily bars (finance-query --interval 1d). Spec §11.1 documents
# the realised corpus as ~1,695 rows per asset; this advisory estimate
# mirrors that order of magnitude without pretending to be exact.
_TRADING_DAYS_PER_YEAR = 252
_HISTORY_YEARS = 7


def build_plan(symbols: list[str]) -> dict:
    """Return a manifest describing the data pipeline for the given symbols."""
    from train.feature_schema import PER_ASSET_BLOCKS
    block_dims = {b.name: b.dim for b in PER_ASSET_BLOCKS} | {"text": 64}
    per_asset_rows_daily_estimate = _TRADING_DAYS_PER_YEAR * _HISTORY_YEARS
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
        "estimated_rows_per_symbol": per_asset_rows_daily_estimate,
        "estimated_total_rows":      per_asset_rows_daily_estimate * len(symbols),
        "estimated_shard_bytes_per_symbol": per_asset_rows_daily_estimate * PER_ASSET_FEATURE_DIM * 4,
        "estimated_total_gb": (
            per_asset_rows_daily_estimate * PER_ASSET_FEATURE_DIM * 4 * len(symbols) / 1e9
        ),
        "symbols": symbols,
    }


# ── Collect ───────────────────────────────────────────────────────────────────

_RATE_LIMIT_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "rate-limit",
    "rate limited",
    "throttl",
)


def _effective_request_permits(cfg: PipelineConfig) -> int:
    if cfg.request_permits is None:
        return max(1, cfg.workers)
    return max(1, min(cfg.request_permits, cfg.workers))


def _is_rate_limit_error(exc: Exception) -> bool:
    stderr = getattr(exc, "stderr", "") or ""
    text = f"{type(exc).__name__}: {exc}\n{stderr}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _retry_delay_seconds(
    retry_index: int,
    cfg: PipelineConfig,
    rng: Callable[[], float] = random.random,
) -> float:
    exp = cfg.retry_base_seconds * (2 ** max(0, retry_index - 1))
    capped = min(exp, cfg.retry_max_seconds)
    # Full-jitter around the capped exponential delay to avoid herd retries.
    jitter_span = min(capped, max(0.0, cfg.retry_jitter_seconds))
    jitter = jitter_span * max(0.0, rng())
    return max(0.0, capped - jitter)


def _collect_one(
    symbol: str,
    cfg: PipelineConfig,
    semaphore: threading.Semaphore | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> tuple[str, bool, str]:
    """Collect one symbol. Returns (symbol, success, message)."""
    from market_data.finance_query_collector import collect_symbol_to_dir

    max_attempts = max(1, cfg.max_retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            if semaphore is None:
                collect_symbol_to_dir(
                    symbol=symbol,
                    data_dir=cfg.data_dir,
                    interval=cfg.interval,
                    start=cfg.start,
                    end=cfg.end,
                    binary=cfg.binary,
                    reports_dir=cfg.reports_dir,
                )
            else:
                with semaphore:
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
            if attempt >= max_attempts or not _is_rate_limit_error(exc):
                return symbol, False, str(exc)
            delay = _retry_delay_seconds(attempt, cfg, rng=rng)
            print(
                f"[pipeline] {symbol} rate-limited (attempt {attempt}/{max_attempts}); "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            sleeper(delay)


def collect_universe(cfg: PipelineConfig) -> None:
    """Collect historical data for all symbols in cfg.symbols_file.

    Runs up to cfg.workers symbols in parallel via a ThreadPoolExecutor.
    Collection is I/O bound (network), so threading is appropriate here.
    """
    symbols = load_symbols(cfg.symbols_file)
    if not symbols:
        print("[pipeline] no symbols found in", cfg.symbols_file, file=sys.stderr)
        return

    # Skip symbols whose shards already exist in all three splits.
    if cfg.skip_existing:
        from market_data.dataset_ingestion import enumerate_shards

        # Clean up stale .tmp files from previous interrupted runs.
        for split in ("train", "validation", "test"):
            split_dir = cfg.data_dir / split
            if split_dir.exists():
                for tmp in split_dir.glob("*.f32bin.tmp"):
                    tmp.unlink(missing_ok=True)

        splits = ("train", "validation", "test")
        existing = {
            s for s in symbols
            if all(
                cfg.data_dir / split / f"{s}.f32bin" in
                set(enumerate_shards(cfg.data_dir, split))
                for split in splits
            )
        }
        if existing:
            print(
                f"[pipeline] skipping {len(existing)} symbols with existing shards: "
                f"{', '.join(sorted(existing))}",
                file=sys.stderr,
            )
        symbols = [s for s in symbols if s not in existing]
        if not symbols:
            print("[pipeline] all symbols already collected", file=sys.stderr)
            return

    print(
        f"[pipeline] collecting {len(symbols)} symbols  "
        f"interval={cfg.interval}  workers={cfg.workers}  "
        f"permits={_effective_request_permits(cfg)}  "
        f"max_retries={cfg.max_retries}",
        file=sys.stderr,
    )

    failed: list[str] = []
    completed = 0
    semaphore = threading.Semaphore(_effective_request_permits(cfg))

    # Prewarm the macro lookup cache by running the first symbol
    # synchronously. The Rust binary writes data/.macro_cache.json on its
    # first invocation; subsequent parallel processes read it instead of
    # hammering FRED with 8× concurrent fetches and getting rate-limited.
    # Spec §6.3 requires yield curve + 12 FRED series; without prewarming,
    # parallel processes race to refetch the same data 8× and ~75% fail.
    if symbols:
        first = symbols[0]
        print(
            f"[pipeline] prewarming macro cache via {first} (synchronous)…",
            file=sys.stderr,
        )
        sym, ok, msg = _collect_one(first, cfg, semaphore=semaphore)
        completed += 1
        status = "ok" if ok else f"FAILED: {msg}"
        print(f"[pipeline] {completed}/{len(symbols)}  {sym}  {status}", file=sys.stderr)
        if not ok:
            failed.append(sym)
        rest = symbols[1:]
    else:
        rest = []

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(_collect_one, sym, cfg, semaphore): sym for sym in rest
        }
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
            f"[pipeline] entering sequential recovery pass for {len(failed)} symbol(s)",
            file=sys.stderr,
        )
        cooldown = min(max(5.0, cfg.retry_base_seconds), 60.0)
        time.sleep(cooldown)
        recovery_cfg = replace(
            cfg,
            workers=1,
            request_permits=1,
            max_retries=max(cfg.max_retries, 10),
            retry_base_seconds=max(cfg.retry_base_seconds, 5.0),
            retry_max_seconds=max(cfg.retry_max_seconds, 180.0),
            retry_jitter_seconds=max(cfg.retry_jitter_seconds, 30.0),
        )
        recovery_sem = threading.Semaphore(1)
        still_failed: list[str] = []
        for sym in failed:
            rsym, ok, msg = _collect_one(sym, recovery_cfg, semaphore=recovery_sem)
            status = "ok (recovery)" if ok else f"FAILED (recovery): {msg}"
            print(
                f"[pipeline] recovery {rsym}  {status}",
                file=sys.stderr,
            )
            if not ok:
                still_failed.append(rsym)
        failed = still_failed

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
    p_collect.add_argument("--start",        default=None)
    p_collect.add_argument("--end",          default=None)
    p_collect.add_argument("--workers",      type=int, default=4)
    p_collect.add_argument("--binary",       type=Path, default=None)
    p_collect.add_argument("--max-retries", type=int, default=5)
    p_collect.add_argument("--retry-base-seconds", type=float, default=2.0)
    p_collect.add_argument("--retry-max-seconds", type=float, default=60.0)
    p_collect.add_argument("--retry-jitter-seconds", type=float, default=1.0)
    p_collect.add_argument("--request-permits", type=int, default=None)
    p_collect.add_argument("--skip-existing", action="store_true",
                           help="Skip symbols whose shards already exist in all splits")

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
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
            retry_jitter_seconds=args.retry_jitter_seconds,
            request_permits=args.request_permits,
            skip_existing=args.skip_existing,
        )
        collect_universe(cfg)

    elif args.command == "stats":
        print_stats(args.data_dir)


if __name__ == "__main__":
    _cli()
