"""Episode detection, shadow-book reconstruction, and the frozen verdict."""

from __future__ import annotations

from dataclasses import dataclass

from harness.costs import one_way_cost
from harness.panel import Panel

from stress_test import GATE


@dataclass(frozen=True)
class Episode:
    peak: int
    trigger: int
    end: int
    closed: bool


def find_episode(dates: list[str], px: list[float], gate: dict = GATE) -> Episode | None:
    last = len(px) - 1
    for i in range(len(px)):
        if dates[i] < gate["start"]:
            continue
        lo = max(0, i - gate["lookback"] + 1)
        high, peak = max((px[j], j) for j in range(lo, i + 1))
        if px[i] > high * (1 - gate["trigger_drawdown"]):
            continue
        cap = i + gate["max_sessions"]
        recovered = next((j for j in range(i + 1, min(cap, last) + 1) if px[j] >= high), None)
        end = recovered if recovered is not None else cap
        return Episode(peak, i, end, end <= last)
    return None


def _weights_by_date(rows: list[dict], strategy: str) -> dict[str, dict[str, float]]:
    return {r["decision_date"]: r["weights"] for r in rows if r["strategy"] == strategy}


def book_returns(rows: list[dict], strategy: str, panel: Panel, start: int, end: int) -> list[float]:
    """Weights logged at session k are held over k -> k+1; a missing row
    carries the prior weights."""
    logged = _weights_by_date(rows, strategy)
    held: dict[str, float] = {}
    for d in sorted(logged):
        if d <= panel.dates[start - 1]:
            held = logged[d]
    out = []
    for k in range(start, end):
        target = logged.get(panel.dates[k], held)
        turnover_cost = sum(
            abs(target.get(s, 0.0) - held.get(s, 0.0)) * one_way_cost(s)
            for s in set(target) | set(held)
        )
        risky = sum(w * (panel.returns[s][k + 1] or 0.0) for s, w in target.items())
        cash = (1 - sum(target.values())) * panel.cash_returns[k + 1]
        out.append(risky + cash - turnover_cost)
        held = target
    return out


def path_stats(returns: list[float]) -> dict[str, float]:
    level, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        level *= 1 + r
        peak = max(peak, level)
        worst = max(worst, 1 - level / peak)
    return {"total_return": level - 1, "max_drawdown": worst}


def verdict(leader: dict[str, float], null: dict[str, float], gate: dict = GATE) -> dict:
    tail = leader["max_drawdown"] <= gate["tail_ratio"] * null["max_drawdown"]
    timing = leader["total_return"] >= null["total_return"] - gate["return_tolerance"]
    return {"tail": tail, "timing": timing, "passes": tail and timing}
