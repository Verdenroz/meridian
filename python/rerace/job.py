"""Monthly re-race of the held gates on their frozen pre-registrations.

The held races (eightk_race, credit_race) re-run as out-of-sample data
accrues: a real edge should strengthen, a fluke should fade. Nothing here
retunes anything — the race modules own their frozen parameters; this job
only refreshes data, runs them, and appends each verdict to an
append-only, git-tracked log so the evidence trail lives in the repo
rather than in CI logs. One row per (month, race); re-runs are no-ops.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from credit_race.race import run_race as run_credit_race
from eightk_race.race import INTENSITY_PANEL
from eightk_race.race import run_race as run_eightk_race
from event_study.events import fetch_filings
from shadow.job import refresh
from shadow.log import append_decision, load_decisions

DEFAULT_LOG_PATH = Path("data/races/reraces.jsonl")

RACES = {
    "eightk_race": run_eightk_race,
    "credit_race": run_credit_race,
}


def _plain(value: object) -> object:
    """Recursively coerce numpy scalars so verdicts serialize as JSON."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()  # type: ignore[attr-defined]
    return value


def logged_months(path: Path) -> set[tuple[str, str]]:
    """(YYYY-MM, race) pairs already on record."""
    return {
        (row["fetched_at"][:7], row["race"])
        for row in load_decisions(path)
        if "race" in row and "fetched_at" in row
    }


def compact_row(name: str, report: dict, fetched_at: str) -> dict:
    return {
        "race": name,
        "fetched_at": fetched_at,
        "race_pass": bool(report["verdict"]["race_pass"]),
        "verdict": _plain(report["verdict"]),
        "metrics": {
            label: {
                "sharpe": _plain(m["sharpe"]),
                "max_drawdown": _plain(m["max_drawdown"]),
            }
            for label, m in report["metrics"].items()
        },
    }


def run_reraces(
    store_dir: Path,
    log_path: Path,
    base_url: str,
    fetch: bool = True,
) -> dict:
    if fetch:
        refresh(base_url, store_dir)
        for symbol in INTENSITY_PANEL:
            fetch_filings(base_url, store_dir, symbol)
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    month = fetched_at[:7]
    already = logged_months(log_path)
    written, skipped = [], []
    for name, runner in RACES.items():
        if (month, name) in already:
            skipped.append(name)
            continue
        row = compact_row(name, runner(store_dir), fetched_at)
        append_decision(log_path, row)
        written.append(row)
    return {"month": month, "written": written, "skipped": skipped}
