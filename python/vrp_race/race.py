"""VRP race: premium-adjusted VIX sizing, pre-registered.

Registered 2026-08-01, before any result was computed. This is the *new*
pre-registration the vol race's postmortem said such a rule would require
— not a retune of that frozen race, whose FAIL verdict stands on its own
gate. Its diagnosis, recorded 2026-07-31 before this design: the VIX-sized
target lost to the null despite better forecast correlation because
(1) an unconditional mean calibration misprices the state-dependent
variance risk premium (calm vs stressed), and (2) it traded ~2x the
null's turnover (5.5 vs 3.0 one-way/yr). Each challenger below fixes one
or both, factorially, so the outcome attributes which defect mattered:

1. tercile premium — the per-window scale is fitted per VIX-level tercile
   (edges at the training window's 1/3, 2/3 quantiles) instead of one
   global mean ratio. Same rebalance band as the null (0.02).
2. wide band — the original global calibration, rebalance band widened to
   0.05 to cut trading toward the null's turnover.
3. tercile + wide band — both fixes.

All constants (terciles, band 0.05, horizon 21, min 180 training overlap
days) were fixed a priori. The failed global-scale challenger re-runs as a
non-gating reference for attribution. GATE is the amended §1.3 bar, as in
quality_race: outright Sharpe win, or non-inferiority (−0.05) + maxdd
<= 2/3 null + matched-vol placebo — with any (b)-route pass *held, never
shipped* pending re-races + behavior-match (prospective-only clause).
Changing any constant after seeing results invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from harness.view import HistoryView
from vol_race.forecasters import vix_vol_series
from vol_race.race import PRIMARY_HORIZON, CalibratedVolTarget

GATE = {
    "horizon": PRIMARY_HORIZON,
    "n_buckets": 3,
    "wide_band": 0.05,
    "min_train_obs": 180,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2.0 / 3.0,
}


class PremiumAdjustedVolTarget(CalibratedVolTarget):
    """VIX sizing with the risk premium calibrated per VIX-level tercile.

    fit() learns tercile edges and per-tercile mean(fwd RV)/mean(VIX)
    ratios from training days whose forward window closes before the fit
    point; target_weights() applies the scale of the bucket the standing
    VIX level falls in. Everything else is the shared vol-target math."""

    def __init__(
        self, symbol: str, forecast: np.ndarray, returns: np.ndarray, band: float
    ) -> None:
        super().__init__(symbol, forecast, returns, band=band)
        self._edges = np.array([])
        self._scales = np.array([1.0])

    def fit(self, view: HistoryView) -> None:
        end = view.t - self.horizon + 1
        fwd, fc = self._forward[:end], self.forecast[:end]
        mask = np.isfinite(fwd) & np.isfinite(fc) & (fc > 0)
        if mask.sum() < GATE["min_train_obs"]:
            raise ValueError("not enough training overlap to calibrate premium")
        fwd, fc = fwd[mask], fc[mask]
        k = int(GATE["n_buckets"])
        self._edges = np.quantile(fc, [i / k for i in range(1, k)])
        buckets = np.digitize(fc, self._edges)
        scales = np.empty(k)
        for b in range(k):
            sel = buckets == b
            # Quantile edges guarantee non-empty buckets on the train set.
            scales[b] = fwd[sel].mean() / fc[sel].mean()
        self._scales = scales
        self._last_target = None

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        raw = self.forecast[view.t]
        if not (np.isfinite(raw) and raw > 0):
            weight = 0.0
        else:
            scale = self._scales[int(np.digitize(raw, self._edges))]
            weight = min(1.0, self.target_vol / (raw * scale))
        if self._last_target is not None and abs(weight - self._last_target) < self.band:
            return None
        self._last_target = weight
        return {self.symbol: weight}


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, ("SPY",))
    returns = np.array([np.nan if r is None else r for r in panel.returns["SPY"]])
    vix = vix_vol_series(store_dir, panel.dates)
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        "vol-target SPY 15% (null)": lambda: VolTarget("SPY"),
        "vix global scale (reference)": lambda: CalibratedVolTarget(
            "SPY", vix, returns
        ),
        "vix tercile premium": lambda: PremiumAdjustedVolTarget(
            "SPY", vix, returns, band=0.02
        ),
        "vix global + wide band": lambda: CalibratedVolTarget(
            "SPY", vix, returns, band=GATE["wide_band"]
        ),
        "vix tercile + wide band": lambda: PremiumAdjustedVolTarget(
            "SPY", vix, returns, band=GATE["wide_band"]
        ),
    }
    runs = race_contenders(panel, contenders, first)

    null = runs["vol-target SPY 15% (null)"]
    challengers = (
        "vix tercile premium",
        "vix global + wide band",
        "vix tercile + wide band",
    )
    verdicts = {c: amended_bar(runs[c], null, panel, first, GATE) for c in challengers}

    sample = null["result"]
    return {
        "gate": dict(GATE),
        "oos_span": (sample["dates"][0], sample["dates"][-1]),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdicts": verdicts,
        "race_pass": any(v["passes"] for v in verdicts.values()),
    }
