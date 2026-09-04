"""Freshness checks for the evidence clocks: did last session get logged?

The last session is the newest daily SPY candle upstream, so holidays never
raise a false alarm. Each finding alerts and the command exits nonzero, so
a CI run of it is itself the alarm.
"""

from __future__ import annotations

from pathlib import Path

from market_data.client import fetch_chart
from market_data.series import utc_date
from shadow.log import load_decisions

from execution.broker.paper_state import PaperStore


def last_session(base_url: str) -> str:
    candles = fetch_chart(base_url, "SPY", range_="5d")["candles"]
    return max(utc_date(c["timestamp"]) for c in candles if c.get("close"))


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
