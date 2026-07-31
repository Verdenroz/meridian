"""SpineStrategy: composes vol forecast, regime filter, safe-asset checks and
the allocator into a harness Strategy.

Causality: all precomputed series are causal by construction (row t uses data
≤ t; asserted by tests), the HMM is fitted in fit() on data before the test
span, and inference advances a forward filter one day at a time.
"""

from __future__ import annotations

import numpy as np
from harness.panel import Panel
from harness.view import HistoryView

from spine.allocator import TARGET_VOL, allocate
from spine.correlation import rolling_correlation
from spine.regime import GaussianHMM
from spine.vol import ewma_vol_series

SAFE_CANDIDATES = ("TLT", "GLD")
SAFE_CORR_WINDOW = 60
REBALANCE_BAND = 0.02


class SpineStrategy:
    def __init__(
        self,
        panel: Panel,
        features: np.ndarray,
        n_states: int = 2,
        target_vol: float = TARGET_VOL,
        band: float = REBALANCE_BAND,
    ) -> None:
        self.panel = panel
        self.features = features
        self.n_states = n_states
        self.target_vol = target_vol
        self.band = band

        returns = {
            s: np.array([np.nan if r is None else r for r in panel.returns[s]])
            for s in panel.returns
        }
        self._vol = ewma_vol_series(returns["SPY"])
        self._safe_corr = {
            s: rolling_correlation(returns[s], returns["SPY"], SAFE_CORR_WINDOW)
            for s in SAFE_CANDIDATES
            if s in returns
        }
        self._hmm: GaussianHMM | None = None
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._filter_t = -1
        self._last_target: dict[str, float] | None = None

    def _standardize(self, rows: np.ndarray) -> np.ndarray:
        assert self._mu is not None and self._sigma is not None
        return (rows - self._mu) / self._sigma

    def fit(self, view: HistoryView) -> None:
        t = view.t
        finite = np.isfinite(self.features[: t + 1]).all(axis=1)
        train = self.features[: t + 1][finite]
        if len(train) < 20 * self.n_states:
            raise ValueError("not enough finite feature rows to fit regime model")
        self._mu = train.mean(axis=0)
        self._sigma = np.maximum(train.std(axis=0), 1e-12)
        self._hmm = GaussianHMM(self.n_states).fit(self._standardize(train))
        # Warm the forward filter through the training span so the first
        # out-of-sample day starts from the filtered state, not the prior.
        alpha = self._hmm.filter_posterior(self._standardize(train))
        self._alpha = alpha[-1]
        self._filter_t = t
        self._last_target = None

    def _advance_filter(self, t: int) -> None:
        assert self._hmm is not None and self._alpha is not None
        while self._filter_t < t:
            self._filter_t += 1
            row = self.features[self._filter_t]
            z = self._standardize(row) if np.isfinite(row).all() else None
            self._alpha = self._hmm.filter_step(self._alpha, z)

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        t = view.t
        vol = self._vol[t]
        if not np.isfinite(vol):
            return None  # warm-up: stay as-is (cash at the start of a run)
        self._advance_filter(t)
        assert self._alpha is not None
        p_stress = float(self._alpha[-1])
        equity_vol_weight = min(1.0, self.target_vol / float(vol))
        safe = {
            s: bool(np.isfinite(c[t]) and c[t] < 0.0)
            for s, c in self._safe_corr.items()
        }
        target = allocate(p_stress, equity_vol_weight, safe)
        if self._last_target is not None:
            drift = max(
                abs(target.get(s, 0.0) - self._last_target.get(s, 0.0))
                for s in set(target) | set(self._last_target)
            )
            if drift < self.band:
                return None
        self._last_target = target
        return target
