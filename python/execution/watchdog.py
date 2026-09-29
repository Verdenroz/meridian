"""Freshness checks for the evidence clocks: did last session get logged?

The last session is the newest completed daily SPY candle upstream, so
holidays never raise a false alarm; the current day's bar is in progress
until the UTC date rolls, so it is excluded. Each finding alerts and the command exits nonzero, so
a CI run of it is itself the alarm.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from market_data.client import fetch_chart
from market_data.series import utc_date
from shadow.log import load_decisions

from execution.broker.paper_state import PaperStore


def last_session(base_url: str, today: str | None = None) -> str:
    today = today or datetime.now(UTC).strftime("%Y-%m-%d")
    candles = fetch_chart(base_url, "SPY", range_="5d")["candles"]
    days = (utc_date(c["timestamp"]) for c in candles if c.get("close"))
    return max(d for d in days if d < today)


def shadow_findings(log_path: Path, session: str, strategies: tuple[str, ...]) -> list[str]:
    latest = {r["strategy"]: r["decision_date"] for r in load_decisions(log_path)}
    return [
        f"shadow {s}: latest decision {latest.get(s, 'none')} but last session is {session}"
        for s in strategies
        if latest.get(s) != session
    ]


def paper_findings(store: PaperStore, decision_date: str) -> list[str]:
    """One journal per broker; a journal that was never started is silent."""
    label = store.directory.name
    rows = [r for r in store.journal_rows() if r["kind"] in ("cycle", "halt", "rejection")]
    if not rows:
        return [f"{label}: account open but no cycle ever logged"] if store.exists() else []
    latest = rows[-1]
    if latest["decision_date"] != decision_date:
        return [
            (
                f"{label}: last {latest['kind']} was for {latest['decision_date']}, "
                f"shadow decision {decision_date} not acted on"
            )
        ]
    if latest["kind"] != "cycle":
        return [f"{label}: {latest['kind']} on {decision_date}: {latest.get('reason', 'held')}"]
    return []


# Newest heartbeat each append-only archive may lag by, in days. Feeds poll
# every 10 minutes; the others are daily or weekday crons.
ARCHIVE_MAX_AGE_DAYS = {
    "archive/feeds/heartbeats": 1,
    "archive/pername/heartbeats": 2,
    "archive/bars/heartbeats": 4,
}
TONE_SCORES = "data/tone_live/scores.jsonl"
TONE_MAX_AGE_DAYS = 3


def archive_findings(root: Path, today: date) -> list[str]:
    findings = []
    for rel, max_age in ARCHIVE_MAX_AGE_DAYS.items():
        beats = sorted((root / rel).glob("????-??-??.jsonl"))
        newest = date.fromisoformat(beats[-1].stem) if beats else None
        if newest is None or (today - newest).days > max_age:
            findings.append(f"{rel}: newest heartbeat {newest or 'none'} is over {max_age}d old")
    scores = root / TONE_SCORES
    lines = scores.read_text().splitlines() if scores.exists() else []
    newest = date.fromisoformat(json.loads(lines[-1])["fetched_at"][:10]) if lines else None
    if newest is None or (today - newest).days > TONE_MAX_AGE_DAYS:
        findings.append(f"{TONE_SCORES}: newest score {newest or 'none'} is over {TONE_MAX_AGE_DAYS}d old")
    return findings

