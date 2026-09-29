"""The registered VWAP-stretch rule on one session of SPY 1m bars. Pure.

A bar's decision happens at its close (start + 1 minute) and fills at the
next bar's open, so no decision reads a price it could not have seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, time
from itertools import pairwise

from bars_archive.fetch import ET

SESSION_OPEN = time(9, 30)


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_ts: int
    exit_ts: int


def bar_time(ts: int) -> time:
    return datetime.fromtimestamp(ts, UTC).astimezone(ET).time()


def _minutes_since_open(ts: int) -> float:
    close = datetime.fromtimestamp(ts + 60, UTC).astimezone(ET)
    opened = close.replace(hour=SESSION_OPEN.hour, minute=SESSION_OPEN.minute, second=0)
    return (close - opened).total_seconds() / 60


def vwap_path(bars: list[dict]) -> list[float]:
    """Upstream reports zero volume on some opening bars; until volume
    accrues the path is the close."""
    path, pv, vol = [], 0.0, 0.0
    for b in bars:
        pv += (b["h"] + b["l"] + b["c"]) / 3 * b["v"]
        vol += b["v"]
        path.append(pv / vol if vol > 0 else b["c"])
    return path


def session_sigma(prior_sessions: list[list[dict]]) -> float:
    rets = [
        math.log(b["c"] / a["c"])
        for bars in prior_sessions
        for a, b in pairwise(bars)
    ]
    mean = sum(rets) / len(rets)
    return math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))


def find_trade(spy: list[dict], sigma: float, gate: dict) -> Trade | None:
    entry_start = time.fromisoformat(gate["entry_start"])
    entry_end = time.fromisoformat(gate["entry_end"])
    flat_by = time.fromisoformat(gate["flat_by"])
    vwap = vwap_path(spy)
    for i, bar in enumerate(spy[:-1]):
        decided = bar_time(bar["ts"] + 60)
        if not entry_start <= decided <= entry_end:
            continue
        band = gate["k"] * sigma * math.sqrt(_minutes_since_open(bar["ts"]))
        stretch = math.log(bar["c"] / vwap[i])
        if abs(stretch) < band:
            continue
        symbol = "SPY" if stretch < 0 else "SH"
        return Trade(symbol, spy[i + 1]["ts"], _exit_ts(spy, vwap, i + 1, stretch < 0, flat_by))
    return None


def _exit_ts(spy: list[dict], vwap: list[float], entry: int, long_spy: bool, flat_by: time) -> int:
    for j in range(entry, len(spy)):
        if bar_time(spy[j]["ts"]) >= flat_by:
            return spy[j]["ts"]
        reverted = spy[j]["c"] >= vwap[j] if long_spy else spy[j]["c"] <= vwap[j]
        if reverted and j + 1 < len(spy):
            return spy[j + 1]["ts"]
    return spy[-1]["ts"]
