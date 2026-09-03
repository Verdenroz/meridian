"""Stock-sleeve race: per-name risk forecasting and risk-weighted sizing
inside the equity bucket, pre-registered before any result was computed.

Motivation. The charter's mandate has two halves: regime-dependent exposure
(built, held) and a 20–30 name risk-weighted sleeve inside the equity bucket
(never built). The record says the sleeve is where learned components get
their first fair fight: every regime challenger so far learned from one
series with ~9 stress episodes, whereas per-name vol is a cross-sectional
problem with 25 names × ~30 years of (day, name) rows. The charter's line
holds throughout — names are sized by forecast risk only, never by any view
on their returns.

Universe, fixed a priori: `sleeve.universe.SLEEVE_UNIVERSE`, 25 liquid US
large caps with deep clean history; survivorship bias is disclosed with the
result. A name is eligible once it has `min_history` consecutive
finite returns and a finite forecast — inception honored causally. Below
`min_names` eligible names the equity bucket falls back to the index.

Stage 1 — forecast quality, out-of-sample. Candidates are per-name
annualized vol forecasts for day t+1 scored against `horizon`-day forward
realized vol on the union of walk-forward test spans, on the common finite
sample per name:
  • ewma      — RiskMetrics EWMA(0.94) per name, the incumbent and the bar;
  • har_lite  — equal-weight 1/5/22d realized-variance mean, parameter-free;
  • ridge     — the learned challenger: one pooled cross-sectional ridge
    (lambda `ridge_lambda` on standardized columns, unpenalized intercept)
    from six causal log-vol features (own EWMA, own realized 5/22/63d,
    market EWMA, market realized 22d) to log forward realized vol, fitted on
    every (day, name) training row whose forward window closes at or before
    the fit boundary and refit per walk-forward window.
Primary metric: mean over names (≥ `min_scored_days` scored days) of the
Pearson forecast/realized correlation — the spine's own vol bar statistic.
Secondary (reported, non-gating): mean daily cross-sectional Spearman across
eligible names on days with ≥ `min_cross_section` names — the ranking the
sizing rule actually consumes. A candidate graduates to stage 2 only if its
primary metric beats the incumbent's in-run value by ≥ `min_corr_margin`.

Stage 2 — the race. The null's vol-target shell is kept exactly (15% target,
2% band) and only the equity bucket changes: a basket of eligible names
weighted by inverse forecast vol under a `max_name_weight` cap, sized by the
basket's forecast vol (forecast vols on the diagonal, Ledoit–Wolf shrunk
correlation over `corr_window` days, refit every `corr_refit_days`). Stock
one-way cost is `stock_cost_bps`, conservative for large caps at the close.
Contenders, all fixed a priori:
  1. sleeve inverse-EWMA — the classical challenger, always raced;
  2. sleeve inverse-<graduate> — one per stage-1 graduate;
  reference (non-gating): sleeve equal-weight sized on the EWMA risk model,
  so risk-weighting is attributable against 1/N in the same shell; buy-hold
  and the untouched null.
Gate: each challenger clears the charter's amended bar against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown ≤ 2/3 of the null's + beat of the
matched-vol placebo. Full walk-forward span, same engine. A seeded block
bootstrap on the Sharpe difference vs the null is reported per challenger
(non-gating). Stacking the sleeve on the regime model is a separate, later
registration; nothing here ships — the roadmap's ordering (sleeve after the
ETF core wins) is untouched because this race only tests, never promotes.
Changing any constant after seeing results invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from harness.costs import DEFAULT_COST_BPS
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.racing import race_contenders
from harness.stats import sharpe_diff_bootstrap, spearman
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import walk_forward_splits
from spine.vol import forecast_correlation

from sleeve.forecasters import (
    PooledRidgeForecaster,
    StaticForecast,
    VolForecaster,
    ewma_vol_matrix,
    forward_realized_vol_matrix,
    har_lite_vol_matrix,
    ridge_features,
    walk_forward_forecasts,
)
from sleeve.panel import build_sleeve_panel
from sleeve.strategy import SleeveData, SleeveVolTarget, sleeve_data
from sleeve.universe import MARKET_SYMBOL, SLEEVE_UNIVERSE

__all__ = ["GATE", "run_race"]

GATE = {
    "horizon": 21,
    "min_history": 252,
    "corr_window": 252,
    "corr_refit_days": 21,
    "min_names": 8,
    "max_name_weight": 0.10,
    "target_vol": 0.15,
    "band": 0.02,
    "stock_cost_bps": 3.0,
    "ridge_lambda": 1.0,
    "min_corr_margin": 0.01,
    "min_scored_days": 252,
    "min_cross_section": 10,
    "bootstrap_block_len": 21,
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20260901,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
INCUMBENT = "ewma"
NULL = "vol-target SPY 15% (null)"
EQUAL_REF = "sleeve equal-weight (ref)"


def _challenger_label(name: str) -> str:
    return f"sleeve inverse-{name}"


def build_forecasters(data: SleeveData) -> dict[str, VolForecaster]:
    horizon = int(GATE["horizon"])
    features = ridge_features(data.returns, data.market_returns)
    forward = forward_realized_vol_matrix(data.returns, horizon)
    return {
        INCUMBENT: StaticForecast(ewma_vol_matrix(data.returns)),
        "har_lite": StaticForecast(har_lite_vol_matrix(data.returns)),
        "ridge": PooledRidgeForecaster(features, forward, horizon, float(GATE["ridge_lambda"])),
    }


def _per_name_correlation(forecast: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    scores = []
    for j in range(forecast.shape[1]):
        col = mask[:, j]
        if col.sum() < int(GATE["min_scored_days"]):
            continue
        scores.append(
            forecast_correlation(
                np.where(col, forecast[:, j], np.nan), np.where(col, target[:, j], np.nan)
            )
        )
    return float(np.mean(scores))


def _cross_sectional_spearman(forecast: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    scores = []
    for t in range(forecast.shape[0]):
        row = mask[t]
        if row.sum() < int(GATE["min_cross_section"]):
            continue
        scores.append(spearman(forecast[t, row], target[t, row])[0])
    return float(np.mean(scores)) if scores else float("nan")


def stage1(
    data: SleeveData, forecasters: dict[str, VolForecaster], splits: list[tuple[int, int, int]]
) -> dict:
    """OOS forecast-quality scores on the common finite sample per name."""
    shape = data.returns.shape
    target = forward_realized_vol_matrix(data.returns, int(GATE["horizon"]))
    stitched = {
        name: walk_forward_forecasts(f, splits, shape) for name, f in forecasters.items()
    }
    mask = np.isfinite(target) & (data.run_lengths >= int(GATE["min_history"]))
    for series in stitched.values():
        mask &= np.isfinite(series)
    primary = {n: _per_name_correlation(s, target, mask) for n, s in stitched.items()}
    secondary = {n: _cross_sectional_spearman(s, target, mask) for n, s in stitched.items()}
    bar = primary[INCUMBENT] + float(GATE["min_corr_margin"])
    graduates = [n for n, c in primary.items() if n != INCUMBENT and c >= bar]
    return {
        "primary_per_name_pearson": primary,
        "secondary_cross_sectional_spearman": secondary,
        "scored_rows": int(mask.sum()),
        "graduates": graduates,
    }


def _sleeve(data: SleeveData, forecaster: VolForecaster, sizing: str) -> SleeveVolTarget:
    return SleeveVolTarget(
        data,
        forecaster,
        sizing=sizing,
        target_vol=float(GATE["target_vol"]),
        band=float(GATE["band"]),
        max_name_weight=float(GATE["max_name_weight"]),
        min_names=int(GATE["min_names"]),
        min_history=int(GATE["min_history"]),
        corr_window=int(GATE["corr_window"]),
        corr_refit_days=int(GATE["corr_refit_days"]),
    )


def _inverse_vol_factory(data: SleeveData, forecaster: VolForecaster) -> Callable[[], Strategy]:
    return lambda: _sleeve(data, forecaster, "inverse_vol")


def _bootstrap(chall: dict, null: dict, cash_by_date: dict[str, float]) -> dict:
    dates = set(chall["result"]["dates"]) & set(null["result"]["dates"])

    def aligned(run: dict) -> np.ndarray:
        return np.array(
            [r for d, r in zip(run["result"]["dates"], run["result"]["returns"]) if d in dates]
        )

    cash = np.array([cash_by_date[d] for d in sorted(dates)])
    return sharpe_diff_bootstrap(
        aligned(chall),
        aligned(null),
        cash,
        block_len=int(GATE["bootstrap_block_len"]),
        n_resamples=int(GATE["bootstrap_resamples"]),
        seed=int(GATE["bootstrap_seed"]),
    )


def run_race(store_dir: Path) -> dict:
    panel, filled_gaps = build_sleeve_panel(store_dir, MARKET_SYMBOL, SLEEVE_UNIVERSE)
    data = sleeve_data(panel, MARKET_SYMBOL, SLEEVE_UNIVERSE)
    first = panel.first_index_with_data((MARKET_SYMBOL,))
    splits = walk_forward_splits(first, panel.n_days)
    forecasters = build_forecasters(data)
    quality = stage1(data, forecasters, splits)

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold(MARKET_SYMBOL),
        NULL: lambda: VolTarget(MARKET_SYMBOL),
        EQUAL_REF: lambda: _sleeve(data, forecasters[INCUMBENT], "equal"),
    }
    challengers = [INCUMBENT, *quality["graduates"]]
    for name in challengers:
        contenders[_challenger_label(name)] = _inverse_vol_factory(data, forecasters[name])
    cost_bps = {**DEFAULT_COST_BPS, **{s: float(GATE["stock_cost_bps"]) for s in SLEEVE_UNIVERSE}}
    runs = race_contenders(panel, contenders, first, cost_bps=cost_bps)
    cash_by_date = dict(zip(panel.dates, panel.cash_returns))
    verdicts = {}
    for name in challengers:
        label = _challenger_label(name)
        verdict = amended_bar(runs[label], runs[NULL], panel, first, GATE)
        verdict["bootstrap_vs_null"] = _bootstrap(runs[label], runs[NULL], cash_by_date)
        verdicts[label] = verdict
    return {
        "gate": dict(GATE),
        "universe": list(SLEEVE_UNIVERSE),
        "filled_gaps": filled_gaps,
        "n_days": panel.n_days,
        "oos_span": (runs[NULL]["result"]["dates"][0], runs[NULL]["result"]["dates"][-1]),
        "stage1": quality,
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {
            "per_challenger": verdicts,
            "race_pass": any(v["passes"] for v in verdicts.values()),
        },
    }
