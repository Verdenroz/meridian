"""Tone-brake race: the deployment test the 2026-08-01 tone study unlocked,
pre-registered before any result was computed.

The tone study passed both legs: negative-tone days beat a matched
news-active control on 5-day forward vol, and corpus-wide negative tone
rank-predicts SPY residual vol (spearman 0.128). Its verdict unlocked tone
features "via their own future races" — this is that race. The signal is
short-horizon (gone by 21 days), so the deployment shape is the frozen
brake family, not a slow regime feature: the null vol-target with equity
halved for the 5 sessions after an unusually-negative-tone burst.

Signal construction, fixed a priori:
- Corpus-wide headline-weighted mean FinBERT tone per UTC date — exactly
  the passed study's market-leg series (per-symbol daily means recombined
  weighted by headline count; duplicates across symbols count per
  appearance, as in the study). Deterministic: 100% sentiment-cache hits,
  no new inference.
- A UTC date's headlines belong to the first trading session on or after
  it (weekend news lands on Monday) — the 8-K race's availability
  convention. Sessions with < 30 scored headlines are uncovered (NaN),
  inheriting the study's market-leg floor.
- Burst = causal trailing z of *negative* tone >= 2; brake mechanics
  (z window 252/min 120, scale 0.5, 5 sessions) inherited frozen from the
  occurrence-flag race, as the credit race inherited them.

Contenders: the untouched null, the tone brake (primary challenger), the
tone∪credit compound brake (challenger — two independent stress channels,
credit mechanics imported frozen from the credit race), and the credit
brake alone as a non-gating reference so any compound uplift is
attributable to its tone leg.

OOS span is availability-driven, stated procedurally before results: from
the first session with a finite tone z (warm-up complete) through the last
covered session (FNSPID coverage ends 2023) — after that the brake cannot
fire and scoring uncovered years would only dilute both arms equally.

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched null on the clipped span — outright after-cost Sharpe win, or
held route (b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's
+ beat of the matched-vol placebo. A (b)-route pass is held, never shipped
on this backtest alone. The signal spearman is diagnostic, non-gating.
Known caveats, pre-registered: FNSPID carries publisher timestamps (the
availability clock above is the standard next-session convention);
FinBERT scores older text (content-only, backtest-legal); one macro cycle
of genuinely negative-tone bursts. Changing any constant after seeing
results invalidates the race.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series, widening_brake
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.signals import BrakeVolTarget, brake_series, causal_zscores
from harness.stats import spearman
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from spine.vol import ewma_vol_series, forward_realized_vol
from tone_study.study import FNSPID_CACHE, SENTIMENT_MODEL, daily_tone

__all__ = ["market_daily_tone", "run_race", "tone_by_session"]

GATE = {
    "min_headlines": 30,
    "z_window": 252,
    "z_min_obs": 120,
    "burst_z": 2.0,
    "brake_scale": 0.5,
    "brake_days": 5,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
    "signal_min_spearman": 0.10,
    "max_p_value": 0.01,
}
SIGNAL_HORIZON = 5
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = ("vol-target + tone brake", "vol-target + tone∪credit brake")

# Derived, rebuildable aggregate of the deterministic sentiment cache.
TONE_CACHE = Path("data/tone_race/market_tone.json")


def market_daily_tone() -> dict[str, tuple[int, float]]:
    """UTC date -> (headline count, headline-weighted mean tone) over the
    whole FNSPID corpus; disk-cached because the underlying scan touches
    1.4M cache entries."""
    if TONE_CACHE.is_file():
        doc = json.loads(TONE_CACHE.read_text())
        if doc.get("model") == SENTIMENT_MODEL:
            return {d: (int(n), float(m)) for d, (n, m) in doc["days"].items()}
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for path in sorted(FNSPID_CACHE.glob("*.jsonl")):
        for date, (count, mean) in daily_tone(path.stem).items():
            box = totals[date]
            box[0] += count
            box[1] += count * mean
    days = {d: (int(c), s / c) for d, (c, s) in totals.items() if c > 0}
    TONE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TONE_CACHE.write_text(json.dumps({"model": SENTIMENT_MODEL, "days": days}))
    return days


def tone_by_session(dates: list[str], min_headlines: int) -> np.ndarray:
    """Mean tone aligned to sessions: a UTC date's headlines belong to the
    first session on or after it; under-covered sessions stay NaN."""
    count = np.zeros(len(dates))
    total = np.zeros(len(dates))
    for date, (n, mean) in market_daily_tone().items():
        i = bisect_left(dates, date)
        if i < len(dates):
            count[i] += n
            total[i] += n * mean
    out = np.full(len(dates), np.nan)
    mask = count >= min_headlines
    out[mask] = total[mask] / count[mask]
    return out


def negative_tone_z(tone: np.ndarray) -> np.ndarray:
    return causal_zscores(-tone, int(GATE["z_window"]), int(GATE["z_min_obs"]))


def _clipped_runner(lo: str, hi: str) -> Callable[..., dict]:
    """run_contender-shaped runner scoring only the registered OOS span —
    both bounds, unlike the shared runner's start-only clip."""

    def run(
        panel: Panel,
        factory: Callable[[], Strategy],
        first: int,
        oos_start: str | None = None,
    ) -> dict:
        result = run_walk_forward(panel, factory, first)
        cash_by_date = dict(zip(panel.dates, panel.cash_returns))
        kept = [
            (r, cash_by_date[d])
            for d, r in zip(result["dates"], result["returns"])
            if lo <= d <= hi
        ]
        metrics = compute_metrics([r for r, _ in kept], [c for _, c in kept])
        return {"result": result, "metrics": metrics, "oos_start": lo}

    return run


def signal_check(panel: Panel, z: np.ndarray) -> dict:
    """Diagnostic: does the negative-tone z rank SPY residual forward vol?"""
    returns = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
    with np.errstate(invalid="ignore"):
        ratio = forward_realized_vol(returns, SIGNAL_HORIZON) / ewma_vol_series(returns)
    mask = np.isfinite(z) & np.isfinite(ratio) & (ratio > 0)
    rho, p = spearman(z[mask], np.log(ratio[mask]))
    return {"n_days": int(mask.sum()), "spearman": rho, "p_value": p}


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, ("SPY",))
    tone = tone_by_session(panel.dates, int(GATE["min_headlines"]))
    z = negative_tone_z(tone)
    tone_active = brake_series(z, GATE["burst_z"], int(GATE["brake_days"]))
    credit_active = widening_brake(credit_spread_series(store_dir, panel.dates))
    union_active = tone_active | credit_active
    first = panel.first_index_with_data(("SPY",))

    finite_z = np.flatnonzero(np.isfinite(z))
    covered = np.flatnonzero(np.isfinite(tone))
    if not len(finite_z):
        raise RuntimeError("tone z never warms up; check FNSPID/sentiment caches")
    lo, hi = panel.dates[finite_z[0]], panel.dates[covered[-1]]
    run = _clipped_runner(lo, hi)

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "vol-target + tone brake": lambda: BrakeVolTarget(
            "SPY", tone_active, GATE["brake_scale"]
        ),
        "vol-target + credit brake (ref)": lambda: BrakeVolTarget(
            "SPY", credit_active, GATE["brake_scale"]
        ),
        "vol-target + tone∪credit brake": lambda: BrakeVolTarget(
            "SPY", union_active, GATE["brake_scale"]
        ),
    }
    runs = {label: run(panel, factory, first) for label, factory in contenders.items()}
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE, run=run)
        for label in CHALLENGERS
    }
    signal = signal_check(panel, z)
    in_span = slice(finite_z[0], covered[-1] + 1)
    return {
        "gate": dict(GATE),
        "oos_span": (lo, hi),
        "active_days": {
            "tone": int(tone_active[in_span].sum()),
            "credit": int(credit_active[in_span].sum()),
            "union": int(union_active[in_span].sum()),
            "span_days": int(in_span.stop - in_span.start),
        },
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "signal": signal,
        "verdict": {
            "per_challenger": verdicts,
            "race_pass": any(v["passes"] for v in verdicts.values()),
            "signal_pass": (
                signal["spearman"] >= GATE["signal_min_spearman"]
                and signal["p_value"] <= GATE["max_p_value"]
            ),
        },
    }
