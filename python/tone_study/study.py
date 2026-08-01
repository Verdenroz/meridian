"""Content-scored news event study: the branch the 2026-07-31 event study
left shelved, pre-registered 2026-08-01 before any result was computed.

That study's verdict was "volume isn't information, content might be":
news-burst *counts* showed no vol signal, and news features stay shelved
"unless content-scored variants pass this same gate". This is that test,
run through the same gate values (uplift >= 0.05, p <= 0.01, min 100
events) on the same 15-name panel, same availability clock (a headline is
known at the close of its UTC date), same EWMA-relative forward-vol
ratios pooled across names.

Content source: the FNSPID corpus (1.41M headlines, 482 symbols) scored
by FinBERT — signed P(pos)−P(neg) per headline — via the legacy
sentiment cache (sha256(model + text) -> f32; 100% hit rate verified on
a 9.6k sample, so the tone series involves no new model inference and is
deterministic). Thresholds fixed a priori, on FinBERT's natural [-1, 1]
scale, before any ratio was computed:

- negative day:  >= 3 scored headlines, mean tone <= -0.3  (primary)
- severe day:    >= 3 scored headlines, mean tone <= -0.5  (diagnostic)
- neutral-matched control: >= 3 scored headlines, mean tone > -0.1 —
  the discriminator the count study could not run: both groups are
  news-active, only the content differs. The content leg passes ONLY if
  negative days beat this matched control (uplift >= 0.05, p <= 0.01),
  not merely the all-days baseline a volume effect would also clear.
- market leg: aggregate mean tone across the corpus on >= 30-headline
  days; spearman(-tone, log residual SPY vol ratio) >= 0.10, p <= 0.01.

Known caveats, pre-registered: FNSPID carries publisher timestamps, not
`fetched_at` (the standard next-close convention is the availability
clock, same as the count study); overlapping forward windows make
p-values optimistic; FinBERT is a 2019-era model scoring older text
(backtest-legal: content-only, no future knowledge beyond language
drift). Changing any constant after seeing results invalidates the
study — rerun under a new pre-registration instead.
"""

from __future__ import annotations

import functools
import hashlib
import json
import struct
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from event_study.events import FNSPID_CACHE
from event_study.stats import spearman
from event_study.study import (
    HORIZON_DAYS,
    PANEL,
    SECONDARY_HORIZON,
    _symbol_series,
    _uplift,
    _vol_ratios,
)

SENTIMENT_CACHE = Path("data/sentiment_cache")
SENTIMENT_MODEL = "ProsusAI/finbert"
MAX_CHARS = 2000

GATE = {
    # Inherited unchanged from the count study's gate.
    "horizon_days": HORIZON_DAYS,
    "min_median_uplift": 0.05,
    "max_p_value": 0.01,
    "min_events": 100,
    "market_min_spearman": 0.10,
    # Tone thresholds, fixed a priori on FinBERT's [-1, 1] scale.
    "min_headlines": 3,
    "negative_tone": -0.3,
    "severe_tone": -0.5,
    "neutral_floor": -0.1,
    "market_min_headlines": 30,
}


def _tone(text: str) -> float | None:
    t = text.strip()[:MAX_CHARS]
    if not t:
        return None
    h = hashlib.sha256(f"{SENTIMENT_MODEL}\x00{t}".encode()).hexdigest()
    path = SENTIMENT_CACHE / h[:2] / h
    try:
        return struct.unpack("<f", path.read_bytes())[0]
    except (OSError, struct.error):
        return None


@functools.cache
def daily_tone(symbol: str) -> dict[str, tuple[int, float]]:
    """UTC date -> (scored-headline count, mean signed tone). Memoized: the
    market leg touches all 482 symbols once per horizon."""
    path = FNSPID_CACHE / f"{symbol}.jsonl"
    if not path.is_file():
        return {}
    sums: dict[str, list[float]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                ts, text = row["ts"], row["text"]
            except (ValueError, KeyError):
                continue
            score = _tone(text)
            if score is None:
                continue
            date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
            sums[date].append(score)
    return {d: (len(v), float(np.mean(v))) for d, v in sums.items()}


def _day_class(count: int, tone: float, threshold: float) -> str | None:
    if count < GATE["min_headlines"]:
        return None
    if tone <= threshold:
        return "event"
    if tone > GATE["neutral_floor"]:
        return "control"
    return None


def leg_tone_days(store_dir: Path, horizon: int, threshold: float) -> dict:
    """Pooled event-day ratios for tone <= threshold days, against BOTH the
    all-covered-days baseline and the neutral-matched news-active control."""
    events: list[float] = []
    control: list[float] = []
    baseline: list[float] = []
    for symbol in PANEL:
        tones = daily_tone(symbol)
        if not tones:
            continue
        dates, returns = _symbol_series(store_dir, symbol)
        ratios = _vol_ratios(returns, horizon)
        covered_lo, covered_hi = min(tones), max(tones)
        for t, date in enumerate(dates):
            if not (covered_lo <= date <= covered_hi and np.isfinite(ratios[t])):
                continue
            baseline.append(float(ratios[t]))
            count, tone = tones.get(date, (0, 0.0))
            cls = _day_class(count, tone, threshold)
            if cls == "event":
                events.append(float(ratios[t]))
            elif cls == "control":
                control.append(float(ratios[t]))

    out: dict = {"n_events": len(events), "n_control": len(control)}
    enough = len(events) >= GATE["min_events"]
    out["vs_baseline"] = (
        _uplift(np.array(events), np.array(baseline))
        if enough and len(baseline) >= GATE["min_events"]
        else None
    )
    out["vs_matched_control"] = (
        _uplift(np.array(events), np.array(control))
        if enough and len(control) >= GATE["min_events"]
        else None
    )
    return out


def leg_market_tone(store_dir: Path, horizon: int) -> dict:
    """Does market-wide *negative* tone rank-predict SPY residual vol?"""
    agg: dict[str, list[float]] = defaultdict(list)
    for path in sorted(FNSPID_CACHE.glob("*.jsonl")):
        for date, (count, tone) in daily_tone(path.stem).items():
            agg[date].extend([tone] * count)

    dates, returns = _symbol_series(store_dir, "SPY")
    ratios = _vol_ratios(returns, horizon)
    neg_tone = np.full(len(dates), np.nan)
    for t, date in enumerate(dates):
        scores = agg.get(date, [])
        if len(scores) >= GATE["market_min_headlines"]:
            neg_tone[t] = -float(np.mean(scores))
    mask = np.isfinite(neg_tone) & np.isfinite(ratios) & (ratios > 0)
    rho, p = spearman(neg_tone[mask], np.log(ratios[mask]))
    return {"n_days": int(mask.sum()), "spearman": rho, "p_value": p}


def _passes(result: dict | None) -> bool:
    return (
        result is not None
        and result["median_uplift"] >= GATE["min_median_uplift"]
        and result["p_value"] <= GATE["max_p_value"]
    )


def run_study(store_dir: Path) -> dict:
    report: dict = {"gate": dict(GATE), "horizons": {}}
    for horizon in (HORIZON_DAYS, SECONDARY_HORIZON):
        report["horizons"][horizon] = {
            "negative_days": leg_tone_days(store_dir, horizon, GATE["negative_tone"]),
            "severe_days": leg_tone_days(store_dir, horizon, GATE["severe_tone"]),
            "market_tone": leg_market_tone(store_dir, horizon),
        }

    primary = report["horizons"][HORIZON_DAYS]
    # The content claim: negative days beat the *matched* news-active control.
    content_pass = _passes(primary["negative_days"]["vs_matched_control"])
    market = primary["market_tone"]
    market_pass = (
        market["spearman"] >= GATE["market_min_spearman"]
        and market["p_value"] <= GATE["max_p_value"]
    )
    report["verdict"] = {
        "content_pass": content_pass,
        "market_tone_pass": market_pass,
        "overall_pass": content_pass or market_pass,
    }
    return report
