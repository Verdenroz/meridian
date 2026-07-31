"""Vol-forecaster race: better forecasts attack the null through its own engine.

The null baseline *is* a vol-target on EWMA(0.94); a forecaster with more
information should beat it in the same walk-forward race with no new
mechanism. Two pre-registered stages:

Stage 1 (forecast quality): Pearson correlation with 21d forward realized
vol on SPY, all candidates scored on the common finite sample. A candidate
graduates only if it beats the incumbent EWMA's in-run correlation (the
established ~0.67 bar) by at least `min_corr_margin`.

Stage 2 (the race, graduates only): a vol-target strategy sized on the
challenger forecast, raced walk-forward after costs against the untouched
null. Scale calibration is part of the challenger and fitted per window on
training data only: forecasts are multiplied by mean(fwd RV)/mean(forecast)
over training days whose 21d forward window closes before the fit point.
Gate: after-cost OOS Sharpe > the null's, same span.

GATE and the candidate set were fixed before any result was computed;
changing either after seeing results invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from harness.engine import Strategy
from harness.metrics import compute_metrics
from harness.panel import build_panel
from harness.strategies import BuyHold, VolTarget
from harness.view import HistoryView
from harness.walkforward import run_walk_forward
from spine.vol import forecast_correlation, forward_realized_vol

from vol_race.forecasters import build_candidates

PRIMARY_HORIZON = 21
SECONDARY_HORIZON = 5
GATE = {
    "primary_horizon": PRIMARY_HORIZON,
    "secondary_horizon": SECONDARY_HORIZON,
    "min_corr_margin": 0.01,
}
TARGET_VOL = 0.15
REBALANCE_BAND = 0.02


class CalibratedVolTarget:
    """Vol-target sized on a precomputed causal forecast; scale fitted on
    training data only (forward-RV windows that close before the fit point)."""

    def __init__(
        self,
        symbol: str,
        forecast: np.ndarray,
        returns: np.ndarray,
        horizon: int = PRIMARY_HORIZON,
        target_vol: float = TARGET_VOL,
        band: float = REBALANCE_BAND,
    ) -> None:
        self.symbol = symbol
        self.forecast = forecast
        self.horizon = horizon
        self.target_vol = target_vol
        self.band = band
        self._forward = forward_realized_vol(returns, horizon)
        self._scale = 1.0
        self._last_target: float | None = None

    def fit(self, view: HistoryView) -> None:
        end = view.t - self.horizon + 1  # forward windows closing <= view.t
        fwd, fc = self._forward[:end], self.forecast[:end]
        mask = np.isfinite(fwd) & np.isfinite(fc) & (fc > 0)
        if mask.sum() < 60:
            raise ValueError("not enough training overlap to calibrate scale")
        self._scale = float(fwd[mask].mean() / fc[mask].mean())
        self._last_target = None

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        raw = self.forecast[view.t]
        vol = raw * self._scale if np.isfinite(raw) and raw > 0 else np.nan
        weight = 0.0 if not np.isfinite(vol) else min(1.0, self.target_vol / vol)
        if self._last_target is not None and abs(weight - self._last_target) < self.band:
            return None
        self._last_target = weight
        return {self.symbol: weight}


def stage1_correlations(
    candidates: dict[str, np.ndarray], returns: np.ndarray, horizon: int
) -> dict[str, float]:
    """All candidates scored on the common finite sample for fairness."""
    target = forward_realized_vol(returns, horizon)
    mask = np.isfinite(target)
    for series in candidates.values():
        mask &= np.isfinite(series)
    return {
        name: forecast_correlation(
            np.where(mask, series, np.nan), np.where(mask, target, np.nan)
        )
        for name, series in candidates.items()
    }


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, ("SPY",))
    returns = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
    candidates = build_candidates(store_dir, panel.dates, returns)

    corr = {
        h: stage1_correlations(candidates, returns, h)
        for h in (PRIMARY_HORIZON, SECONDARY_HORIZON)
    }
    primary = corr[PRIMARY_HORIZON]
    bar = primary["ewma"] + GATE["min_corr_margin"]
    graduates = [n for n, c in primary.items() if n != "ewma" and c >= bar]

    first = panel.first_index_with_data(("SPY",))
    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15% (null)": lambda: VolTarget("SPY"),
    }
    def challenger_factory(series: np.ndarray) -> Callable[[], Strategy]:
        return lambda: CalibratedVolTarget("SPY", series, returns)

    for name in graduates:
        contenders[f"vol-target on {name}"] = challenger_factory(candidates[name])
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    metrics = {}
    oos_span = ("", "")
    for label, factory in contenders.items():
        result = run_walk_forward(panel, factory, first)
        metrics[label] = compute_metrics(
            result["returns"], [cash_by_date[d] for d in result["dates"]]
        )
        oos_span = (result["dates"][0], result["dates"][-1])

    null_sharpe = metrics["vol-target SPY 15% (null)"]["sharpe"]
    winners = [
        n for n in graduates
        if metrics[f"vol-target on {n}"]["sharpe"] > null_sharpe
    ]
    return {
        "gate": dict(GATE),
        "correlations": corr,
        "graduates": graduates,
        "metrics": metrics,
        "oos_span": oos_span,
        "verdict": {
            "ewma_corr": primary["ewma"],
            "null_sharpe": null_sharpe,
            "winners": winners,
            "race_pass": bool(winners),
        },
    }
