"""Topic-salience race: the first *slow* news-derived regime feature,
pre-registered before any result was computed.

Every news signal raced so far was fast and short-horizon — tone fades
by 21 days, occurrence has no per-day ranking power — and the spec
records "never as a slow regime feature" for tone *levels*. Topic
*composition* is a different object: the share of news about inflation
and rates was elevated for two consecutive years in 2021–23. If the
news corpus's topical mix shifts slowly with the macro regime, it can
inform the HMM the way the credit level and rate vol do. This race
tests exactly one channel, chosen a priori for its overlap with the
proven rate story: inflation/rates salience.

Signal construction, fixed a priori:
- Embeddings: the legacy harrier-oss:0.6b cache (1024-dim, keyed
  sha256(headline); 100% hit rate verified on a 2,000-headline sample;
  the live endpoint reproduces cached vectors at cosine 0.9995, so the
  anchor and the corpus share one geometry).
- Anchor text (single a-priori choice, one channel, no scan):
  "Inflation and interest rates: Federal Reserve policy tightening,
  rising bond yields, CPI inflation data, rate hikes."
- Daily salience = per-appearance mean cosine(headline, anchor) across
  the FNSPID corpus (the tone study's market-leg weighting convention);
  a UTC date's headlines belong to the first session on or after it;
  sessions with < 30 embedded headlines are uncovered.
- Feature = causal trailing z (window 252, min 120) of the 21-session
  trailing mean salience; 0.0 (neutral: at trailing mean) where
  unavailable, so the walk-forward can fit pre-coverage windows.

Challengers, both fixed a priori: (1) spine 4f + salience (5f-sal) —
value alone; (2) spine 6f + salience (7f) — compounding on the leader's
feature set (registered risk: the corr race's 7f diluted). References
(non-gating): buy-hold, the untouched null, spine 6f. All spines
2-state for comparability with the feature-race series; state count is
an orthogonal axis adjudicated by the nstate races.

Scoring span is availability-driven, stated procedurally: from the
first session with a nonzero feature (z warm) through the last covered
session (FNSPID ends 2023-12); both bounds applied to every contender
equally. Gate: each challenger clears the amended bar (§1.3) against
the untouched null on the clipped span. Changing any constant after
seeing results invalidates the race.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.signals import causal_zscores
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from tone_study.study import FNSPID_CACHE

__all__ = ["daily_salience", "run_race", "salience_feature"]

GATE = {
    "min_headlines": 30,
    "smooth_days": 21,
    "z_window": 252,
    "z_min_obs": 120,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
ANCHOR_TEXT = (
    "Inflation and interest rates: Federal Reserve policy tightening, "
    "rising bond yields, CPI inflation data, rate hikes."
)
EMBED_CACHE = Path("data/embed_cache")
EMBED_ENDPOINT = "http://127.0.0.1:12434/v1/embeddings"
EMBED_MODEL = "harrier-oss:0.6b"
# Derived, rebuildable daily aggregate (5.6GB of cache reads otherwise).
SALIENCE_CACHE = Path("data/salience_race/daily.json")

RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = ("spine 5f-sal (+salience)", "spine 7f (6f +salience)")


def _cached_embedding(text: str) -> np.ndarray | None:
    h = hashlib.sha256(text.encode()).hexdigest()
    path = EMBED_CACHE / h[:2] / h
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return np.frombuffer(raw, dtype="<f4") if len(raw) == 4096 else None


def _anchor_embedding() -> np.ndarray:
    import urllib.request

    req = urllib.request.Request(
        EMBED_ENDPOINT,
        data=json.dumps({"model": EMBED_MODEL, "input": ANCHOR_TEXT}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        vec = json.load(resp)["data"][0]["embedding"]
    out = np.array(vec, dtype=np.float32)
    return out / np.linalg.norm(out)


def daily_salience() -> dict[str, tuple[int, float]]:
    """UTC date -> (headline count, mean anchor cosine); disk-cached."""
    if SALIENCE_CACHE.is_file():
        doc = json.loads(SALIENCE_CACHE.read_text())
        if doc.get("anchor") == ANCHOR_TEXT:
            return {d: (int(n), float(m)) for d, (n, m) in doc["days"].items()}
    anchor = _anchor_embedding()
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    from datetime import UTC, datetime

    for path in sorted(FNSPID_CACHE.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    ts, text = row["ts"], row["text"]
                except (ValueError, KeyError):
                    continue
                emb = _cached_embedding(text)
                if emb is None:
                    continue
                norm = float(np.linalg.norm(emb))
                if norm == 0:
                    continue
                cos = float(np.dot(emb, anchor) / norm)
                date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                box = totals[date]
                box[0] += 1
                box[1] += cos
    days = {d: (int(c), s / c) for d, (c, s) in totals.items() if c > 0}
    SALIENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SALIENCE_CACHE.write_text(json.dumps({"anchor": ANCHOR_TEXT, "days": days}))
    return days


def salience_feature(dates: list[str]) -> np.ndarray:
    """Causal z of the 21-session mean salience; 0.0 where unavailable."""
    count = np.zeros(len(dates))
    total = np.zeros(len(dates))
    for date, (n, mean) in daily_salience().items():
        i = bisect_left(dates, date)
        if i < len(dates):
            count[i] += n
            total[i] += n * mean
    raw = np.full(len(dates), np.nan)
    mask = count >= GATE["min_headlines"]
    raw[mask] = total[mask] / count[mask]

    smooth = np.full(len(dates), np.nan)
    w = int(GATE["smooth_days"])
    for t in range(len(dates)):
        window = raw[max(0, t - w + 1) : t + 1]
        finite = window[np.isfinite(window)]
        if len(finite):
            smooth[t] = float(finite.mean())
    z = causal_zscores(smooth, int(GATE["z_window"]), int(GATE["z_min_obs"]))
    return np.where(np.isfinite(z), z, 0.0)


def _clipped_runner(lo: str, hi: str) -> Callable[..., dict]:
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


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features4 = build_features(panel, store_dir)
    f6 = np.column_stack(
        [
            features4,
            credit_spread_series(store_dir, panel.dates),
            rate_vol_feature(store_dir, panel.dates),
        ]
    )
    sal = salience_feature(panel.dates)
    f5_sal = np.column_stack([features4, sal])
    f7 = np.column_stack([f6, sal])
    first = panel.first_index_with_data(("SPY",))

    nonzero = np.flatnonzero(sal != 0.0)
    if not len(nonzero):
        raise RuntimeError("salience feature never warms; check embed cache")
    lo, hi = panel.dates[nonzero[0]], panel.dates[nonzero[-1]]
    run = _clipped_runner(lo, hi)

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f (leader ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 5f-sal (+salience)": lambda: TunedSpineStrategy(panel, f5_sal),
        "spine 7f (6f +salience)": lambda: TunedSpineStrategy(panel, f7),
    }
    runs = {label: run(panel, factory, first) for label, factory in contenders.items()}
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE, run=run)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "anchor": ANCHOR_TEXT,
        "oos_span": (lo, hi),
        "n_days": panel.n_days,
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {
            "per_challenger": verdicts,
            "race_pass": any(v["passes"] for v in verdicts.values()),
        },
    }
