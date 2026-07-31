"""SpineStrategy: composes vol forecast, regime filter, safe-asset checks and
the allocator into a harness Strategy — plus the walk-forward tuner.

Causality: all precomputed series are causal by construction (row t uses data
≤ t; asserted by tests), the HMM is fitted in fit() on data before the test
span, and inference advances a forward filter one day at a time. The tuner
selects an allocator mapping by racing candidates *on the training window* in
the same engine — out-of-sample data never informs the choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from harness.engine import run_backtest
from harness.panel import Panel
from harness.view import HistoryView

from spine.allocator import TARGET_VOL, allocate
from spine.correlation import rolling_correlation
from spine.regime import GaussianHMM
from spine.vol import ewma_vol_series

SAFE_CANDIDATES = ("TLT", "GLD")
SAFE_CORR_WINDOW = 60
REBALANCE_BAND = 0.02


@dataclass(frozen=True)
class AllocatorConfig:
    """How the regime posterior modulates the vol-target equity weight."""

    mapping: str = "product"  # product | min | threshold | vol_only
    threshold: float = 0.85
    stress_scale: float = 0.25

    def label(self) -> str:
        if self.mapping == "threshold":
            return f"threshold({self.threshold})"
        return self.mapping

    def equity_weight(self, p_stress: float, vol_weight: float) -> float:
        if self.mapping == "product":
            return vol_weight * (1.0 - p_stress)
        if self.mapping == "min":
            return min(vol_weight, 1.0 - p_stress)
        if self.mapping == "threshold":
            scale = self.stress_scale if p_stress >= self.threshold else 1.0
            return vol_weight * scale
        if self.mapping == "vol_only":
            return vol_weight
        raise ValueError(f"unknown mapping {self.mapping}")


# Fixed candidate set for the tuner. vol_only is the null-equivalent floor: if
# the regime signal adds nothing on the training window, the tuner falls back
# to plain vol-targeting rather than shipping a worse book.
TUNER_CANDIDATES: tuple[AllocatorConfig, ...] = (
    AllocatorConfig("product"),
    AllocatorConfig("min"),
    AllocatorConfig("threshold", threshold=0.70),
    AllocatorConfig("threshold", threshold=0.85),
    AllocatorConfig("threshold", threshold=0.95),
    AllocatorConfig("vol_only"),
)


@dataclass
class FittedRegime:
    hmm: GaussianHMM
    mu: np.ndarray
    sigma: np.ndarray


class SpineStrategy:
    def __init__(
        self,
        panel: Panel,
        features: np.ndarray,
        config: AllocatorConfig | None = None,
        fitted: FittedRegime | None = None,
        n_states: int = 2,
        target_vol: float = TARGET_VOL,
        band: float = REBALANCE_BAND,
    ) -> None:
        self.panel = panel
        self.features = features
        self.config = config or AllocatorConfig()
        self.n_states = n_states
        self.target_vol = target_vol
        self.band = band
        self._fitted = fitted

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
        self._alpha: np.ndarray | None = None
        self._filter_t = -1
        self._last_target: dict[str, float] | None = None

    def _standardize(self, rows: np.ndarray) -> np.ndarray:
        assert self._fitted is not None
        return (rows - self._fitted.mu) / self._fitted.sigma

    def fit_regime(self, t: int) -> FittedRegime:
        """Fit the HMM on all finite feature rows up to index t."""
        finite = np.isfinite(self.features[: t + 1]).all(axis=1)
        train = self.features[: t + 1][finite]
        if len(train) < 20 * self.n_states:
            raise ValueError("not enough finite feature rows to fit regime model")
        mu = train.mean(axis=0)
        sigma = np.maximum(train.std(axis=0), 1e-12)
        hmm = GaussianHMM(self.n_states).fit((train - mu) / sigma)
        return FittedRegime(hmm, mu, sigma)

    def fit(self, view: HistoryView) -> None:
        if self._fitted is None:
            self._fitted = self.fit_regime(view.t)
        # Warm the forward filter from the start of history so the first
        # decision day starts from the filtered state, not the prior.
        assert self._fitted.hmm.startprob_ is not None
        self._alpha = self._fitted.hmm.startprob_.copy()
        self._filter_t = -1
        self._advance_filter(view.t)
        self._last_target = None

    def _advance_filter(self, t: int) -> None:
        assert self._fitted is not None and self._alpha is not None
        while self._filter_t < t:
            self._filter_t += 1
            row = self.features[self._filter_t]
            z = self._standardize(row) if np.isfinite(row).all() else None
            self._alpha = self._fitted.hmm.filter_step(self._alpha, z)

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        t = view.t
        vol = self._vol[t]
        if not np.isfinite(vol):
            return None  # warm-up: stay as-is (cash at the start of a run)
        self._advance_filter(t)
        assert self._alpha is not None
        p_stress = float(self._alpha[-1])
        vol_weight = min(1.0, self.target_vol / float(vol))
        equity = self.config.equity_weight(p_stress, vol_weight)
        safe = {
            s: bool(np.isfinite(c[t]) and c[t] < 0.0)
            for s, c in self._safe_corr.items()
        }
        target = allocate(equity, safe)
        if self._last_target is not None:
            drift = max(
                abs(target.get(s, 0.0) - self._last_target.get(s, 0.0))
                for s in set(target) | set(self._last_target)
            )
            if drift < self.band:
                return None
        self._last_target = target
        return target


class TunedSpineStrategy:
    """Fits the regime model, then picks the allocator mapping by racing the
    fixed candidate set over the training span in the real engine (after-cost
    Sharpe vs cash). The chosen strategy runs the out-of-sample window."""

    def __init__(
        self,
        panel: Panel,
        features: np.ndarray,
        candidates: tuple[AllocatorConfig, ...] = TUNER_CANDIDATES,
        n_states: int = 2,
    ) -> None:
        self.panel = panel
        self.features = features
        self.candidates = candidates
        self.n_states = n_states
        self.chosen: AllocatorConfig | None = None
        self._inner: SpineStrategy | None = None

    def _train_sharpe(self, result: dict) -> float:
        returns = np.array(result["returns"])
        cash = dict(zip(self.panel.dates, self.panel.cash_returns))
        excess = returns - np.array([cash[d] for d in result["dates"]])
        std = float(excess.std(ddof=1))
        return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    def fit(self, view: HistoryView) -> None:
        t = view.t
        fitted = SpineStrategy(
            self.panel, self.features, n_states=self.n_states
        ).fit_regime(t)
        eval_start = int(np.argmax(np.isfinite(self.features[: t + 1]).all(axis=1)))
        best_score = -np.inf
        for config in self.candidates:
            candidate = SpineStrategy(
                self.panel, self.features, config=config, fitted=fitted
            )
            result = run_backtest(self.panel, candidate, start=eval_start, end=t)
            score = self._train_sharpe(result)
            if score > best_score:
                best_score, self.chosen = score, config
        assert self.chosen is not None
        self._inner = SpineStrategy(
            self.panel, self.features, config=self.chosen, fitted=fitted
        )
        self._inner.fit(view)

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        assert self._inner is not None
        return self._inner.target_weights(view)
