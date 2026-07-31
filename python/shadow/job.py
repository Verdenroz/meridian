"""Daily shadow decision job: fetch, decide, append. The paper-trading clock.

Every run refreshes the store, points each contender at the latest close
through the same lookahead-proof HistoryView the backtests use, and appends
the target weights it would hold tomorrow. Behavior-match evaluation
criteria must be pre-registered separately BEFORE evaluation begins —
this module only accumulates the evidence.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from credit_race.race import RACE_SYMBOLS
from harness.panel import Panel, build_panel
from harness.view import HistoryView
from market_data import DEFAULT_FRED_SERIES
from market_data.client import fetch_fred_series, fetch_treasury_yields
from market_data.macro import normalize_fred, normalize_treasury
from market_data.pipeline import fetch_symbol
from market_data.store import read_snapshot, write_snapshot

from shadow.contenders import build_contenders, extract_meta, round_weights
from shadow.log import JOB_VERSION, append_decision, logged_keys

MAX_STALENESS_DAYS = 7


def refresh(base_url: str, store_dir: Path) -> None:
    """Refresh charts, FRED series, and the current treasury year (past
    years are immutable — only the open year is refetched and merged)."""
    for symbol in RACE_SYMBOLS:
        fetch_symbol(base_url, store_dir, symbol)
    for series_id in DEFAULT_FRED_SERIES:
        rows = normalize_fred(fetch_fred_series(base_url, series_id))
        write_snapshot(store_dir, "macro", series_id, {"rows": rows})
    year = datetime.now(UTC).year
    fresh = normalize_treasury(fetch_treasury_yields(base_url, year))
    doc = read_snapshot(store_dir, "macro", "treasury_yields")
    kept = [
        r for r in (doc["data"]["rows"] if doc else [])
        if not r["date"].startswith(str(year))
    ]
    merged = sorted(kept + fresh, key=lambda r: r["date"])
    write_snapshot(store_dir, "macro", "treasury_yields", {"rows": merged})


def check_staleness(decision_date: str, today: date) -> None:
    age = (today - date.fromisoformat(decision_date)).days
    if age > MAX_STALENESS_DAYS:
        raise RuntimeError(
            f"latest panel date {decision_date} is {age} days old — refusing "
            "to log decisions on stale data; check the fetch"
        )


def decision_row(
    name: str, strategy: object, view: HistoryView, fetched_at: str
) -> dict:
    strategy.fit(view)  # type: ignore[attr-defined]
    weights = strategy.target_weights(view) or {}  # type: ignore[attr-defined]
    weights = round_weights(weights)
    return {
        "decision_date": view.today,
        "fetched_at": fetched_at,
        "job_version": JOB_VERSION,
        "strategy": name,
        "weights": weights,
        "cash_weight": round(1.0 - sum(weights.values()), 6),
        "meta": extract_meta(name, strategy, view),
    }


def run_decision(
    store_dir: Path,
    log_path: Path,
    base_url: str,
    fetch: bool = True,
) -> dict:
    if fetch:
        refresh(base_url, store_dir)
    panel: Panel = build_panel(store_dir, RACE_SYMBOLS)
    t = panel.n_days - 1
    decision_date = panel.dates[t]
    check_staleness(decision_date, datetime.now(UTC).date())
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    already = logged_keys(log_path)
    written, skipped = [], []
    for name, strategy in build_contenders(panel, store_dir).items():
        if (decision_date, name) in already:
            skipped.append(name)
            continue
        row = decision_row(name, strategy, HistoryView(panel, t), fetched_at)
        append_decision(log_path, row)
        written.append(row)
    return {
        "decision_date": decision_date,
        "written": written,
        "skipped": skipped,
    }
