"""SQLite logging schema for Meridian paper trading — Section 13.2.

Three tables:
  minute_log    — one row per inference step (390/day during market hours)
  event_log     — one row per discrete financial event detected
  daily_summary — one row per trading day (computed at close)

All writes are append-only. Reads are for analysis and dashboard only.
"""

from __future__ import annotations

import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


# ── DDL ──────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS minute_log (
    id                   INTEGER PRIMARY KEY,
    -- Unix epoch in whole minutes (timestamp // 60)
    epoch_minute         INTEGER NOT NULL,
    -- Per-asset arrays stored as little-endian float32 blobs (100 × 4 bytes each)
    target_weights       BLOB    NOT NULL,   -- f32[100]
    executed_weights     BLOB    NOT NULL,   -- f32[100]
    prices               BLOB    NOT NULL,   -- f32[100]
    -- Portfolio-level scalars
    portfolio_value      REAL    NOT NULL,
    daily_return         REAL    NOT NULL DEFAULT 0.0,
    -- Model internals
    regime_posteriors    BLOB    NOT NULL,   -- f32[4]
    aggregate_surprise   REAL    NOT NULL DEFAULT 0.0,
    active_head          INTEGER NOT NULL DEFAULT 0,  -- 0=Return, 1=Sharpe
    event_count          INTEGER NOT NULL DEFAULT 0,
    inference_latency_us INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_minute_log_epoch ON minute_log (epoch_minute);

CREATE TABLE IF NOT EXISTS event_log (
    id           INTEGER PRIMARY KEY,
    epoch_minute INTEGER NOT NULL,
    symbol       TEXT,                 -- NULL for market-wide events
    event_kind   TEXT    NOT NULL,     -- EventKind enum name
    magnitude    REAL    NOT NULL DEFAULT 1.0,
    description  TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_event_log_epoch  ON event_log (epoch_minute);
CREATE INDEX IF NOT EXISTS idx_event_log_symbol ON event_log (symbol);

CREATE TABLE IF NOT EXISTS daily_summary (
    id                     INTEGER PRIMARY KEY,
    -- Calendar date as ISO string (YYYY-MM-DD)
    date                   TEXT    NOT NULL UNIQUE,
    daily_return           REAL    NOT NULL DEFAULT 0.0,
    daily_sharpe           REAL,
    max_intraday_drawdown  REAL    NOT NULL DEFAULT 0.0,
    turnover               REAL    NOT NULL DEFAULT 0.0,
    transaction_costs      REAL    NOT NULL DEFAULT 0.0,
    trade_count            INTEGER NOT NULL DEFAULT 0,
    -- Dominant regime: 0=Bull, 1=Bear, 2=HighVol, 3=LowVol
    dominant_regime        INTEGER NOT NULL DEFAULT 0,
    surprise_frequency     REAL    NOT NULL DEFAULT 0.0,
    -- JSON blob for top-5 VSN gate feature names
    top5_features          TEXT    NOT NULL DEFAULT '[]',
    -- JSON blob for MoE expert utilization (4 floats)
    expert_utilization     TEXT    NOT NULL DEFAULT '[]',
    portfolio_value_close  REAL    NOT NULL DEFAULT 0.0
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.executescript(_DDL)
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that commits on success and rolls back on exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Blob helpers ──────────────────────────────────────────────────────────────

def pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_f32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ── Write helpers ─────────────────────────────────────────────────────────────

def insert_minute(
    conn: sqlite3.Connection,
    epoch_minute: int,
    target_weights: list[float],      # length 100
    executed_weights: list[float],    # length 100
    prices: list[float],              # length 100
    portfolio_value: float,
    daily_return: float,
    regime_posteriors: list[float],   # length 4
    aggregate_surprise: float,
    active_head: int,
    event_count: int,
    inference_latency_us: int,
) -> None:
    conn.execute(
        """
        INSERT INTO minute_log (
            epoch_minute, target_weights, executed_weights, prices,
            portfolio_value, daily_return, regime_posteriors,
            aggregate_surprise, active_head, event_count, inference_latency_us
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            epoch_minute,
            pack_f32(target_weights),
            pack_f32(executed_weights),
            pack_f32(prices),
            portfolio_value,
            daily_return,
            pack_f32(regime_posteriors),
            aggregate_surprise,
            active_head,
            event_count,
            inference_latency_us,
        ),
    )


def insert_event(
    conn: sqlite3.Connection,
    epoch_minute: int,
    event_kind: str,
    magnitude: float,
    description: str,
    symbol: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO event_log (epoch_minute, symbol, event_kind, magnitude, description)
        VALUES (?, ?, ?, ?, ?)
        """,
        (epoch_minute, symbol, event_kind, magnitude, description),
    )


def upsert_daily_summary(
    conn: sqlite3.Connection,
    date: str,                         # YYYY-MM-DD
    daily_return: float,
    daily_sharpe: float | None,
    max_intraday_drawdown: float,
    turnover: float,
    transaction_costs: float,
    trade_count: int,
    dominant_regime: int,
    surprise_frequency: float,
    top5_features: str,                # JSON string
    expert_utilization: str,           # JSON string
    portfolio_value_close: float,
) -> None:
    conn.execute(
        """
        INSERT INTO daily_summary (
            date, daily_return, daily_sharpe, max_intraday_drawdown,
            turnover, transaction_costs, trade_count, dominant_regime,
            surprise_frequency, top5_features, expert_utilization,
            portfolio_value_close
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            daily_return          = excluded.daily_return,
            daily_sharpe          = excluded.daily_sharpe,
            max_intraday_drawdown = excluded.max_intraday_drawdown,
            turnover              = excluded.turnover,
            transaction_costs     = excluded.transaction_costs,
            trade_count           = excluded.trade_count,
            dominant_regime       = excluded.dominant_regime,
            surprise_frequency    = excluded.surprise_frequency,
            top5_features         = excluded.top5_features,
            expert_utilization    = excluded.expert_utilization,
            portfolio_value_close = excluded.portfolio_value_close
        """,
        (
            date,
            daily_return,
            daily_sharpe,
            max_intraday_drawdown,
            turnover,
            transaction_costs,
            trade_count,
            dominant_regime,
            surprise_frequency,
            top5_features,
            expert_utilization,
            portfolio_value_close,
        ),
    )
