"""Book fills and the registered controls for one session.

Each trade deploys the whole fixed book in whole shares, pays the harness
one-way cost on both legs, and pays commission per order (IBKR tiered, and
a flat stress rate). Random-control draws are seeded per session, so a
session's control never depends on which other sessions were replayed.
"""

from __future__ import annotations

import bisect
import math
from datetime import date, time

import numpy as np
from execution.commissions import tiered_commission
from harness.costs import one_way_cost

from daytrade.signal import Trade, bar_time


def price_at(bars: list[dict], ts: int) -> float:
    """Open of the first bar at or after ts; thin symbols skip no-trade
    minutes. Past the last bar, that bar's close."""
    i = bisect.bisect_left([b["ts"] for b in bars], ts)
    return bars[i]["o"] if i < len(bars) else bars[-1]["c"]


def fill(bars: list[dict], symbol: str, entry_ts: int, exit_ts: int, gate: dict) -> dict:
    cost = one_way_cost(symbol)
    entry = price_at(bars, entry_ts) * (1 + cost)
    exit_ = price_at(bars, exit_ts) * (1 - cost)
    qty = math.floor(gate["book"] / entry)
    gross = qty * (exit_ - entry)
    tiered = tiered_commission(qty, entry) + tiered_commission(qty, exit_)
    stress = 2 * gate["stress_commission"]
    return {
        "symbol": symbol,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_px": round(entry, 4),
        "exit_px": round(exit_, 4),
        "qty": qty,
        "pnl": round(gross - tiered, 4),
        "pnl_stress": round(gross - stress, 4),
        "net_bps": round((gross - tiered) / (qty * entry) * 1e4, 4),
    }


def trade_fill(trade: Trade, bars: dict[str, list[dict]], gate: dict) -> dict:
    return fill(bars[trade.symbol], trade.symbol, trade.entry_ts, trade.exit_ts, gate)


def random_control(trade: Trade, spy: list[dict], session: date, gate: dict) -> Trade:
    stamps = [b["ts"] for b in spy]
    entry_start = time.fromisoformat(gate["entry_start"])
    entry_end = time.fromisoformat(gate["entry_end"])
    flat_by = time.fromisoformat(gate["flat_by"])
    window = [
        i for i, b in enumerate(spy[:-1]) if entry_start <= bar_time(b["ts"] + 60) <= entry_end
    ]
    hold = stamps.index(trade.exit_ts) - stamps.index(trade.entry_ts)
    rng = np.random.default_rng([gate["seed"], session.toordinal()])
    entry = int(rng.choice(window)) + 1
    flat = next((j for j, b in enumerate(spy) if bar_time(b["ts"]) >= flat_by), len(spy) - 1)
    return Trade(trade.symbol, stamps[entry], stamps[min(entry + hold, flat)])


def open_to_close(spy: list[dict], gate: dict) -> dict:
    return fill(spy, "SPY", spy[0]["ts"], spy[-1]["ts"] + 60, gate)
