"""FNSPID news ingestion → per-symbol JSONL cache for the Rust Block-8 collector.

Downloads the FNSPID corpus (`Zihan1004/FNSPID`), filters it to the Meridian
100-asset universe and the train/val date window, and writes one cache file per
symbol at `{data_dir}/.fnspid_cache/{SYMBOL}.jsonl` — a stream of
`{"ts": <unix_seconds>, "text": <headline>}` objects.

The Rust historical collector (`src/historical/fnspid.rs`) reads these and merges
them onto the EDGAR filing-text floor. Both sources then flow through the SAME
FinBERT scorer + nomic embedder, so only the text *source* differs between the
historical (FNSPID) and live (`feeds` RSS) regimes — the representation pipeline
is identical, which is the train/inference-alignment requirement in
DATA_PIPELINE_ISSUES.md (C1/H1).

We deliberately use the article **headline** (`Article_title`) as the text,
because that is exactly what the live `news_sentiment` poller scores over the
realtime RSS stream. Keeping both regimes at headline granularity makes the
Block-6 sentiment slot source-invariant.

Date window: only articles dated on/before `--end` (default 2023-12-31) are
written, since the train/val splits are ≤2023 and live `feeds` carries 2024+
forward. This keeps the historical cache strictly within the no-leakage window.

Live inference NEVER reads this cache; it uses the realtime `feeds` stream.

Usage:
    python -m market_data.fnspid_ingest \
        --symbols-file portfolio_symbols.txt --data-dir data --end 2023-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

# FNSPID ships as raw CSVs (the HF dataset-viewer/parquet export is broken), so
# we download the file directly and stream it. `nasdaq_exteral_data.csv` is the
# canonical, widest-coverage news file; `All_external.csv` is a lighter
# headline-only alternative.
DEFAULT_REPO = "Zihan1004/FNSPID"
DEFAULT_NEWS_FILE = "Stock_news/nasdaq_exteral_data.csv"
LIGHT_NEWS_FILE = "Stock_news/All_external.csv"

# Columns we actually need. Restricting parse columns keeps the 23 GB file's
# memory footprint flat even though we stream it in chunks.
_USECOLS = ["Date", "Article_title", "Stock_symbol"]
_CHUNK_ROWS = 250_000


@dataclass(frozen=True)
class IngestConfig:
    symbols_file: Path
    data_dir: Path
    repo: str = DEFAULT_REPO
    news_file: str = DEFAULT_NEWS_FILE
    end: date = date(2023, 12, 31)
    start: date | None = None
    max_headline_chars: int = 512


def _load_universe(symbols_file: Path) -> list[str]:
    from market_data.pipeline import load_symbols

    return load_symbols(symbols_file)


def _fnspid_aliases(symbol: str) -> set[str]:
    """Map a canonical universe symbol to the ticker spellings FNSPID may use.

    FNSPID tags rows with a plain `Stock_symbol`. Class-share tickers differ in
    punctuation across providers (`BRK-B` vs `BRK.B` vs `BRKB`), so we match any
    of those spellings back to the canonical symbol.
    """
    s = symbol.upper()
    variants = {s, s.replace("-", "."), s.replace("-", ""), s.replace(".", "-")}
    return {v for v in variants if v}


def _build_alias_index(symbols: list[str]) -> dict[str, str]:
    """Reverse map: every FNSPID spelling → canonical universe symbol."""
    index: dict[str, str] = {}
    for sym in symbols:
        for alias in _fnspid_aliases(sym):
            index[alias] = sym
    return index


def _parse_fnspid_ts(raw: str) -> int | None:
    """Parse a FNSPID `Date` cell (e.g. `2023-12-16 23:00:00 UTC`) → unix secs.

    Returns None for unparseable/empty cells (skipped, never fabricated).
    """
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace(" UTC", "").replace("Z", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    # Fall back to pandas' flexible parser for odd rows.
    try:
        import pandas as pd

        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return int(dt.timestamp())
    except Exception:
        return None


def ingest(cfg: IngestConfig) -> dict[str, int]:
    """Run the ingestion. Returns a per-symbol article-count summary."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    symbols = _load_universe(cfg.symbols_file)
    if not symbols:
        raise RuntimeError(f"no symbols found in {cfg.symbols_file}")
    alias_index = _build_alias_index(symbols)

    print(
        f"[fnspid] downloading {cfg.repo}:{cfg.news_file} "
        f"(cached after first run)…",
        file=sys.stderr,
    )
    csv_path = hf_hub_download(
        repo_id=cfg.repo,
        filename=cfg.news_file,
        repo_type="dataset",
    )

    end_ts = int(
        datetime.combine(cfg.end, datetime.max.time())
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    start_ts = (
        int(
            datetime.combine(cfg.start, datetime.min.time())
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        if cfg.start
        else None
    )

    # Accumulate per canonical symbol. Filtered to ~100 names, so even the full
    # corpus collapses to a few hundred-K rows total — comfortably in memory.
    per_symbol: dict[str, set[tuple[int, str]]] = {s: set() for s in symbols}
    scanned = 0
    kept = 0

    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in _USECOLS,
        chunksize=_CHUNK_ROWS,
        dtype=str,
        on_bad_lines="skip",
    )
    for chunk in reader:
        scanned += len(chunk)
        # Restrict to universe tickers before any expensive per-row work.
        chunk = chunk[chunk["Stock_symbol"].isin(alias_index.keys())]
        for sym_raw, date_raw, title in zip(
            chunk["Stock_symbol"], chunk["Date"], chunk["Article_title"]
        ):
            canonical = alias_index.get(str(sym_raw).upper())
            if canonical is None:
                continue
            if title is None or (isinstance(title, float)):
                continue
            text = str(title).strip()
            if not text:
                continue
            ts = _parse_fnspid_ts(date_raw)
            if ts is None or ts > end_ts:
                continue
            if start_ts is not None and ts < start_ts:
                continue
            if len(text) > cfg.max_headline_chars:
                text = text[: cfg.max_headline_chars]
            per_symbol[canonical].add((ts, text))
            kept += 1
        print(
            f"[fnspid] scanned {scanned:,} rows  kept {kept:,}…",
            file=sys.stderr,
            end="\r",
        )

    print(file=sys.stderr)
    cache_dir = cfg.data_dir / ".fnspid_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    for sym, rows in per_symbol.items():
        out_path = cache_dir / f"{sym}.jsonl"
        if not rows:
            # Honest absence: no news ⇒ no file (Rust treats missing as empty).
            if out_path.exists():
                out_path.unlink()
            summary[sym] = 0
            continue
        ordered = sorted(rows, key=lambda r: r[0])
        with out_path.open("w", encoding="utf-8") as fh:
            for ts, text in ordered:
                fh.write(json.dumps({"ts": ts, "text": text}, ensure_ascii=False))
                fh.write("\n")
        summary[sym] = len(ordered)

    covered = sum(1 for v in summary.values() if v > 0)
    total = sum(summary.values())
    print(
        f"[fnspid] wrote {total:,} headlines across {covered}/{len(symbols)} "
        f"symbols → {cache_dir}",
        file=sys.stderr,
    )

    _write_market_daily_aggregate(per_symbol, cfg.data_dir)
    return summary


def _write_market_daily_aggregate(
    per_symbol: dict[str, set[tuple[int, str]]], data_dir: Path
) -> None:
    """Emit the universe-wide daily news aggregate for the market news block (M1).

    For each UTC day, records the total article count and the breadth (number of
    distinct symbols with news) across the whole universe. The Rust collector
    (`src/historical/market_news_flow.rs`) loads this, EMA-detrends + z-scores it,
    and overlays two of the 12 market-news dims so the block carries real DAILY
    variation instead of the near-constant ~monthly FRED-PCA proxy.
    """
    day_secs = 86_400
    counts: dict[int, int] = {}
    breadth: dict[int, set[str]] = {}
    for sym, rows in per_symbol.items():
        for ts, _text in rows:
            day_ts = (ts // day_secs) * day_secs
            counts[day_ts] = counts.get(day_ts, 0) + 1
            breadth.setdefault(day_ts, set()).add(sym)

    days = {
        str(day_ts): [counts[day_ts], len(breadth[day_ts])]
        for day_ts in sorted(counts)
    }
    out_path = data_dir / ".market_news_daily.json"
    out_path.write_text(
        json.dumps({"built_from": "fnspid", "days": days}, ensure_ascii=False)
    )
    print(
        f"[fnspid] wrote market daily news aggregate ({len(days):,} days) → {out_path}",
        file=sys.stderr,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest FNSPID news into the per-symbol Block-8 cache."
    )
    parser.add_argument("--symbols-file", type=Path, default=Path("portfolio_symbols.txt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--news-file",
        default=DEFAULT_NEWS_FILE,
        help=f"FNSPID CSV path in the repo (lighter alt: {LIGHT_NEWS_FILE})",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2023, 12, 31),
        help="Keep articles dated on/before this date (train/val ≤2023).",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Optional lower date bound.",
    )
    args = parser.parse_args()
    cfg = IngestConfig(
        symbols_file=args.symbols_file,
        data_dir=args.data_dir,
        repo=args.repo,
        news_file=args.news_file,
        end=args.end,
        start=args.start,
    )
    ingest(cfg)


if __name__ == "__main__":
    _cli()
