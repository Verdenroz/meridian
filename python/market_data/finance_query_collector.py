"""Historical data collector — Python bridge to the Rust DataBus binary.

The Rust DataBus (`meridian historical --symbol SYM`) emits newline-delimited
JSON rows to stdout.  This module invokes that binary and streams its output
line by line so that the Python ingestion pipeline can consume it without
buffering the entire dataset in memory.

Typical usage:

    for row in iter_rows("AAPL", interval="1d", start="2010-01-01"):
        process(row)

Or pipe directly into dataset_ingestion.ingest_jsonl:

    from market_data.dataset_ingestion import ingest_jsonl
    from market_data.finance_query_collector import stream_lines

    with stream_lines("AAPL", interval="1d", start="2010-01-01") as lines:
        ingest_jsonl(lines, data_dir=Path("data/mmaps"))
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# ── Binary location ───────────────────────────────────────────────────────────

def _find_binary() -> Path:
    """Locate the compiled meridian binary.

    Checks (in order):
      1. MERIDIAN_BIN environment variable
      2. ./target/release/meridian  (cargo build --release)
      3. ./target/debug/meridian    (cargo build)
    """
    import os
    env_path = os.environ.get("MERIDIAN_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"MERIDIAN_BIN={env_path} does not exist")

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "target" / "release" / "meridian",
        repo_root / "target" / "debug" / "meridian",
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "meridian binary not found. Build it with `cargo build --release` "
        "or set the MERIDIAN_BIN environment variable."
    )


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _build_cmd(
    symbol: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    binary: Path | None = None,
) -> list[str]:
    binary = binary or _find_binary()
    cmd = [str(binary), "historical", "--symbol", symbol, "--interval", interval]
    if start:
        cmd += ["--start", start]
    if end:
        cmd += ["--end", end]
    return cmd


@contextmanager
def stream_lines(
    symbol: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    binary: Path | None = None,
    env: dict[str, str] | None = None,
) -> Generator[Iterator[str], None, None]:
    """Context manager that yields a line iterator over the Rust binary's stdout.

    The subprocess is terminated and cleaned up on exit regardless of whether
    the caller consumed all lines.

    Args:
        symbol:   Ticker symbol (e.g. "AAPL").
        interval: Data resolution: "1m", "5m", "1h", or "1d".
        start:    Start date as "YYYY-MM-DD". Omit for maximum available history.
        end:      End date as "YYYY-MM-DD". Omit for today.
        binary:   Path to the meridian binary. Auto-detected if None.
        env:      Extra environment variables to pass (e.g. FRED_API_KEY).

    Yields:
        An iterator of raw UTF-8 lines (including trailing newline stripped).

    Raises:
        FileNotFoundError: If the binary cannot be found.
        subprocess.CalledProcessError: If the binary exits with a non-zero status.
    """
    import os
    proc_env = {**os.environ, **(env or {})}
    cmd = _build_cmd(symbol, interval, start, end, binary)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,  # pass Rust tracing logs through to caller's stderr
        text=True,
        encoding="utf-8",
        env=proc_env,
    )
    try:
        assert proc.stdout is not None
        yield (line.rstrip("\n") for line in proc.stdout)
    finally:
        proc.stdout and proc.stdout.close()
        proc.wait()
        if proc.returncode not in (0, -15):  # -15 = SIGTERM (normal cancel)
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def iter_rows(
    symbol: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    binary: Path | None = None,
    env: dict[str, str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield parsed JSONL row dicts from the Rust DataBus for one symbol.

    Memory-efficient: parses one line at a time without buffering.

    Example::

        for row in iter_rows("AAPL", interval="1d", start="2010-01-01"):
            print(row["timestamp"], row["price_volume"][3])  # close log-return
    """
    import json

    with stream_lines(symbol, interval, start, end, binary, env) as lines:
        for line in lines:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Batch collection ──────────────────────────────────────────────────────────

def collect_symbol_to_dir(
    symbol: str,
    data_dir: Path,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    binary: Path | None = None,
    env: dict[str, str] | None = None,
    reports_dir: Path | None = None,
) -> None:
    """Collect historical data for one symbol and write split-aware shards.

    Convenience wrapper that chains stream_lines → ingest_jsonl.

    Args:
        symbol:      Ticker symbol.
        data_dir:    Root shard directory (written to {data_dir}/{split}/{symbol}.f32bin).
        interval:    Data resolution ("1m", "5m", "1h", "1d").
        start:       Start date "YYYY-MM-DD".
        end:         End date "YYYY-MM-DD".
        binary:      Path to meridian binary (auto-detected if None).
        env:         Extra environment variables.
        reports_dir: Manifest output directory.
    """
    from market_data.dataset_ingestion import ingest_jsonl

    print(f"[collect] {symbol} interval={interval} start={start} end={end}", file=sys.stderr)

    with stream_lines(symbol, interval, start, end, binary, env) as lines:
        ingest_jsonl(lines, data_dir, reports_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect historical data for one symbol via the Rust DataBus"
    )
    parser.add_argument("--symbol",   required=True,                 help="Ticker symbol")
    parser.add_argument("--interval", default="1d",                  help="1m|5m|1h|1d")
    parser.add_argument("--start",    default=None,                  help="YYYY-MM-DD")
    parser.add_argument("--end",      default=None,                  help="YYYY-MM-DD")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument("--binary",   type=Path, default=None)
    args = parser.parse_args()

    collect_symbol_to_dir(
        symbol=args.symbol,
        data_dir=args.data_dir,
        interval=args.interval,
        start=args.start,
        end=args.end,
        binary=args.binary,
    )


if __name__ == "__main__":
    _cli()
