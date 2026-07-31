"""Daily briefing: analogs with their aftermaths, plus the aggregate prior.

The prior is descriptive output for the human reading the briefing; nothing
downstream consumes it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from harness.panel import Panel

from episodic.knn import Analog
from episodic.outcomes import HORIZON, forward_outcomes


def build_briefing(
    panel: Panel,
    states: np.ndarray,
    query_index: int,
    analogs: list[Analog],
    horizon: int = HORIZON,
) -> dict:
    rows: list[dict] = []
    known: list[dict[str, float]] = []
    for a in analogs:
        outcomes = forward_outcomes(panel, a.index, horizon)
        rows.append({"date": a.date, "distance": a.distance, "outcomes": outcomes})
        if outcomes is not None:
            known.append(outcomes)
    prior = None
    if known:
        rets = [o["fwd_return"] for o in known]
        prior = {
            "median_fwd_vol": float(np.median([o["fwd_vol"] for o in known])),
            "median_fwd_return": float(np.median(rets)),
            "fwd_return_q25": float(np.percentile(rets, 25)),
            "fwd_return_q75": float(np.percentile(rets, 75)),
            "worst_fwd_drawdown": float(min(o["fwd_max_drawdown"] for o in known)),
        }
    return {
        "query_date": panel.dates[query_index],
        "horizon": horizon,
        # state feature 0 is log EWMA vol — the standing forecast for tomorrow
        "ewma_vol_forecast": float(np.exp(states[query_index, 0])),
        "analogs": rows,
        "prior": prior,
    }


def _month(day: str) -> str:
    return date.fromisoformat(day).strftime("%b %Y")


def format_briefing(briefing: dict) -> str:
    lines: list[str] = []
    months = ", ".join(_month(r["date"]) for r in briefing["analogs"])
    lines.append(f"nearest analogs for {briefing['query_date']}: {months or 'none'}")
    if briefing["analogs"]:
        h = briefing["horizon"]
        lines.append(
            f"  {'date':<12}{'dist':>7}{f'fwd{h}d ret':>12}"
            f"{f'fwd{h}d vol':>12}{f'fwd{h}d maxdd':>14}"
        )
        for row in briefing["analogs"]:
            o = row["outcomes"]
            cells = (
                f"{o['fwd_return'] * 100:>11.1f}%{o['fwd_vol'] * 100:>11.1f}%"
                f"{o['fwd_max_drawdown'] * 100:>13.1f}%"
                if o
                else f"{'—':>12}{'—':>12}{'—':>14}"
            )
            lines.append(f"  {row['date']:<12}{row['distance']:>7.2f}{cells}")
    prior = briefing["prior"]
    if prior is not None:
        lines.append(
            f"analog prior (median fwd {briefing['horizon']}d): "
            f"vol {prior['median_fwd_vol'] * 100:.1f}% "
            f"vs EWMA forecast {briefing['ewma_vol_forecast'] * 100:.1f}%; "
            f"return {prior['median_fwd_return'] * 100:+.1f}% "
            f"[IQR {prior['fwd_return_q25'] * 100:+.1f}%..{prior['fwd_return_q75'] * 100:+.1f}%]; "
            f"worst maxdd {prior['worst_fwd_drawdown'] * 100:.1f}%"
        )
    lines.append(
        "explanation only — analogs do not touch allocation; allocator use "
        "requires its own pre-registered race"
    )
    return "\n".join(lines)
