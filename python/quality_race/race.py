"""Quality-spread race: Baa−Aaa default-risk premium, pre-registered.

Registered 2026-08-01, before any result was computed. Provenance: the
original intent was ICE BofA high-yield OAS (BAMLH0A0HYM2) as a 6th regime
feature, but upstream truncates all ICE BofA series to ~3y (787 obs from
2023-08) — unusable for a decades-long walk-forward, same finding the
credit race recorded. The availability-driven substitute is the Moody's
quality spread BAA10Y − AAA10Y (both daily: AAA10Y from 1983, BAA10Y from
1986): subtracting the Aaa−10Y component strips the duration/liquidity
premium common to both, isolating the default-risk premium — a different
channel from the 5f spine's BAA10Y *level*. Series choice was made on
depth alone; no outcome was seen first.

Two challengers, both fixed a priori, mechanics inherited frozen from the
credit race (which inherited the brake from the occurrence-flag race):

1. Regime route — the tuned spine with a 6th feature, the quality spread
   level, through the frozen tuner candidate set. The 5f spine runs as a
   non-gating reference so the marginal contribution is attributable.
2. Brake route — the null vol-target with equity halved for 5 sessions
   after a quality-spread widening burst (causal trailing z of the 5-day
   change >= 2; z window 252, min 120).

GATE is the amended §1.3 bar, pre-registered here in full: a challenger
passes if, after costs, walk-forward OOS, (a) Sharpe > null outright, or
(b) all of: Sharpe >= null − 0.05, max drawdown <= 2/3 of the null's, and
Sharpe > a matched-vol placebo (the null statically de-levered to the
challenger's realized OOS vol). Per the prospective-only clause, a
(b)-pass is *held, never shipped* on this backtest alone — it additionally
requires re-races as OOS accrues and the step-9 behavior-match. Changing
any constant after seeing results invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import (
    build_features5,
    causal_zscores,
    widening_brake,
)
from eightk_race.race import BrakeVolTarget
from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.strategies import BuyHold, VolTarget
from harness.view import HistoryView
from harness.walkforward import run_walk_forward
from market_data.store import read_snapshot
from spine.features import _ffill_series, _macro_by_date
from spine.strategy import TunedSpineStrategy

SERIES_HIGH = "BAA10Y"
SERIES_LOW = "AAA10Y"
GATE = {
    # Brake mechanics — inherited frozen from credit_race / eightk_race.
    "change_days": 5,
    "burst_z": 2.0,
    "brake_scale": 0.5,
    "brake_days": 5,
    "z_window": 252,
    "z_min_obs": 120,
    # Amended §1.3 bar (frozen 2026-07-31 amendment, applied prospectively).
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2.0 / 3.0,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")

__all__ = ["causal_zscores", "run_race"]


def quality_spread_series(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Baa−Aaa spread: default-risk premium net of the common 10Y leg."""
    if read_snapshot(store_dir, "macro", SERIES_LOW) is None:
        raise FileNotFoundError(f"macro snapshot {SERIES_LOW} missing; run market_data fetch")
    baa = _ffill_series(dates, _macro_by_date(store_dir, SERIES_HIGH, "value"))
    aaa = _ffill_series(dates, _macro_by_date(store_dir, SERIES_LOW, "value"))
    return baa - aaa


def build_features6(panel: Panel, store_dir: Path) -> np.ndarray:
    """The credit race's 5 features plus the quality spread level."""
    quality = quality_spread_series(store_dir, panel.dates)
    return np.column_stack([build_features5(panel, store_dir), quality])


class ScaledVolTarget(VolTarget):
    """The null statically de-levered by a constant — the matched-vol
    placebo of §1.3(3). Any drawdown edge a constant de-lever reproduces
    is exposure, not skill."""

    def __init__(self, symbol: str, scale: float) -> None:
        super().__init__(symbol)
        self.scale = scale

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        vol = self._ewma_vol(view.returns_window(self.symbol, self.window))
        weight = 0.0 if vol is None else min(1.0, self.target_vol / vol) * self.scale
        if self._last_target is not None and abs(weight - self._last_target) < self.band:
            return None
        self._last_target = weight
        return {self.symbol: weight}


def _run(panel: Panel, factory: Callable[[], Strategy], first: int) -> dict:
    result = run_walk_forward(panel, factory, first)
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    metrics = compute_metrics(
        result["returns"], [cash_by_date[d] for d in result["dates"]]
    )
    years = len(result["returns"]) / 252
    metrics["turnover_oneway"] = (
        sum(w["turnover_oneway"] for w in result["windows"]) / years
    )
    return {"result": result, "metrics": metrics}


def _amended_bar(
    label: str, chall: dict, null: dict, panel: Panel, first: int
) -> dict:
    """Apply the amended §1.3 gate to one challenger; runs the matched-vol
    placebo only when the (b) branch needs it."""
    outright = chall["metrics"]["sharpe"] > null["metrics"]["sharpe"]
    verdict: dict[str, object] = {
        "outright_sharpe_win": bool(outright),
        "co_criterion": None,
        "passes": bool(outright),
        "route": "a" if outright else None,
    }
    if outright:
        return verdict

    noninferior = (
        chall["metrics"]["sharpe"]
        >= null["metrics"]["sharpe"] - GATE["noninferiority_margin"]
    )
    better_tail = (
        chall["metrics"]["max_drawdown"]
        <= GATE["maxdd_ratio"] * null["metrics"]["max_drawdown"]
    )
    co: dict[str, object] = {
        "noninferior_sharpe": bool(noninferior),
        "tail_two_thirds": bool(better_tail),
        "placebo": None,
    }
    if noninferior and better_tail:
        scale = min(
            1.0, chall["metrics"]["ann_vol"] / max(null["metrics"]["ann_vol"], 1e-9)
        )
        placebo = _run(panel, lambda: ScaledVolTarget("SPY", scale), first)
        co["placebo"] = {
            "scale": scale,
            "sharpe": placebo["metrics"]["sharpe"],
            "ann_vol": placebo["metrics"]["ann_vol"],
            "max_drawdown": placebo["metrics"]["max_drawdown"],
        }
        beats_placebo = chall["metrics"]["sharpe"] > placebo["metrics"]["sharpe"]
        co["beats_placebo"] = bool(beats_placebo)
        if beats_placebo:
            verdict["passes"] = True
            verdict["route"] = "b (held: re-race + behavior-match required)"
    verdict["co_criterion"] = co
    return verdict


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features5 = build_features5(panel, store_dir)
    features6 = build_features6(panel, store_dir)
    quality = quality_spread_series(store_dir, panel.dates)
    active = widening_brake(quality)
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15% (null)": lambda: VolTarget("SPY"),
        "vol-target + quality brake": lambda: BrakeVolTarget(
            "SPY", active, GATE["brake_scale"]
        ),
        "spine tuned 5f (reference)": lambda: TunedSpineStrategy(panel, features5),
        "spine tuned 6f (+quality)": lambda: TunedSpineStrategy(panel, features6),
    }
    runs = {label: _run(panel, factory, first) for label, factory in contenders.items()}

    null = runs["vol-target SPY 15% (null)"]
    challengers = ("vol-target + quality brake", "spine tuned 6f (+quality)")
    verdicts = {c: _amended_bar(c, runs[c], null, panel, first) for c in challengers}

    sample = runs["vol-target SPY 15% (null)"]["result"]
    return {
        "gate": dict(GATE),
        "series": f"{SERIES_HIGH} - {SERIES_LOW}",
        "brake_active_days": int(active.sum()),
        "oos_span": (sample["dates"][0], sample["dates"][-1]),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdicts": verdicts,
        "race_pass": any(v["passes"] for v in verdicts.values()),
    }
