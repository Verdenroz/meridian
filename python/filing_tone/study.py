"""Deep filing-tone study: content at high power on filings,
pre-registered 2026-08-01 before any filing was scored.

The evidence matrix has one untested cell. LLM *severity* on recent 8-Ks
ranks nothing (score study: spearman +0.004, n=957). Filing *occurrence*
fails at high power (deep race: 2004+, brake loses). Headline *content*
carries the vol signal (tone study: negative days beat a matched
news-active control). This study tests content on filings at deep-race
power: FinBERT tone over the actual 8-K text, 15-name panel, full
edgar_text coverage.

Scoring pipeline, fixed a priori before any score was computed:
- Documents: 8-K filings from the edgar_index whose text exists in the
  legacy `data/edgar_text/` store (availability-driven coverage).
- Text: HTML-entity-unescaped, whitespace-collapsed; scored from the
  first occurrence of "Item " (8-K cover pages are boilerplate; the
  items are the content) or the start if absent; first 2000 chars —
  the sentiment cache's standing convention. FinBERT (ProsusAI/finbert,
  the tone study's exact model), signed P(pos) − P(neg), written to the
  same sha256(model\x00text) cache so scoring is deterministic and
  re-runs are free.
- Availability: a filing belongs to the first trading session on or
  after its SEC filingDate (the 8-K race convention). Day tone = mean
  over that day's scored 8-Ks for the name.

Legs and gate, inherited from the tone study unchanged (thresholds are
FinBERT-scale constants, not retuned): within *filing days only* —
- negative filing day: mean tone <= -0.3 (primary event)
- neutral-matched control: mean tone > -0.1 — both groups have an 8-K;
  only the language differs. The content leg passes ONLY if negative
  days beat this matched control: median uplift >= 0.05, p <= 0.01,
  min 100 events, at the 5-day horizon (21d diagnostic).
- rank leg: spearman(-tone, log residual vol ratio) >= 0.10, p <= 0.01
  across filing days — the exact test LLM severity failed.

Known caveats, pre-registered: FinBERT's 512-token window reads the
opening items only; 8-K cover-page structure varies across two decades
(the "Item " anchor is a single a-priori choice); scheduled earnings
8-Ks (Item 2.02) dominate counts and dilute both arms equally.
Changing any constant after seeing results invalidates the study.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
from event_study.stats import spearman
from event_study.study import (
    HORIZON_DAYS,
    PANEL,
    SECONDARY_HORIZON,
    _symbol_series,
    _uplift,
    _vol_ratios,
)

from filing_tone.fetch import load_filing_index, text_path
from filing_tone.score import processed_text, read_cached_tone

GATE = {
    "horizon_days": HORIZON_DAYS,
    "min_median_uplift": 0.05,
    "max_p_value": 0.01,
    "min_events": 100,
    "negative_tone": -0.3,
    "neutral_floor": -0.1,
    "min_spearman": 0.10,
}


def eightk_day_tone(store_dir: Path, symbol: str) -> dict[str, float]:
    """Filing date -> mean cached FinBERT tone over that date's 8-Ks."""
    index = load_filing_index(store_dir, symbol) or []
    sums: dict[str, list[float]] = defaultdict(list)
    for row in index:
        if row["form"] != "8-K":
            continue
        path = text_path(row["accession"])
        if not path.is_file():
            continue
        tone = read_cached_tone(processed_text(path))
        if tone is not None:
            sums[row["date"]].append(tone)
    return {d: float(np.mean(v)) for d, v in sums.items()}


def run_study(store_dir: Path) -> dict:
    report: dict = {"gate": dict(GATE), "horizons": {}}
    per_day: list[tuple[str, int, float]] = []  # (symbol, t, tone)
    series: dict[str, tuple[list[str], np.ndarray]] = {}
    for symbol in PANEL:
        dates, returns = _symbol_series(store_dir, symbol)
        series[symbol] = (dates, returns)
        for fdate, tone in eightk_day_tone(store_dir, symbol).items():
            t = bisect_left(dates, fdate)
            if t < len(dates):
                per_day.append((symbol, t, tone))

    for horizon in (HORIZON_DAYS, SECONDARY_HORIZON):
        ratios = {s: _vol_ratios(r, horizon) for s, (_, r) in series.items()}
        events, control, tones, logratios = [], [], [], []
        for symbol, t, tone in per_day:
            ratio = ratios[symbol][t]
            if not (np.isfinite(ratio) and ratio > 0):
                continue
            tones.append(-tone)
            logratios.append(float(np.log(ratio)))
            if tone <= GATE["negative_tone"]:
                events.append(float(ratio))
            elif tone > GATE["neutral_floor"]:
                control.append(float(ratio))
        enough = (
            len(events) >= GATE["min_events"]
            and len(control) >= GATE["min_events"]
        )
        rho, p = spearman(np.array(tones), np.array(logratios))
        report["horizons"][horizon] = {
            "n_filing_days": len(tones),
            "n_events": len(events),
            "n_control": len(control),
            "vs_matched_control": (
                _uplift(np.array(events), np.array(control)) if enough else None
            ),
            "rank": {"spearman": rho, "p_value": p},
        }

    primary = report["horizons"][HORIZON_DAYS]
    uplift = primary["vs_matched_control"]
    content_pass = (
        uplift is not None
        and uplift["median_uplift"] >= GATE["min_median_uplift"]
        and uplift["p_value"] <= GATE["max_p_value"]
    )
    rank = primary["rank"]
    rank_pass = (
        rank["spearman"] >= GATE["min_spearman"]
        and rank["p_value"] <= GATE["max_p_value"]
    )
    report["verdict"] = {
        "content_pass": bool(content_pass),
        "rank_pass": bool(rank_pass),
        "overall_pass": bool(content_pass or rank_pass),
    }
    return report
