"""Student-t emission race: the first learned-regime challenger,
pre-registered before any result was computed.

Motivation, from the accumulated record: the four-state race closed
n-scanning — further regime-capacity work goes through learned challengers,
not n+1 registrations. The cheapest principled upgrade is the emission
family, not the state count: the 6f features are fat-tailed (log vols spike,
drawdown and credit have long stress tails), and a Gaussian emission model
must either inflate a state's variance to cover tail days or carve extra
states for them — the 4-state failure showed where the second path leads. A
Student-t emission keeps the validated state structure but makes each state
robust to its own tail, so state boundaries are set by the bulk of a
regime's days rather than dragged by its outliers.

Model, fixed a priori: diagonal-covariance Student-t emissions with a single
frozen dof nu = 5 for every state and feature (a standard heavy-tail choice;
estimating nu would be a tuning knob this registration deliberately
excludes). EM mirrors the incumbent GaussianHMM exactly — same deterministic
quantile-split init, same forward/backward, same state ordering by the vol
feature, same causal forward filter — with the classical latent-scale
u-weights in the M-step. The rate_race 6f feature matrix unchanged;
TunedSpineStrategy unchanged except the emission class.

Challengers, both fixed a priori: spine 6f t5 2-state and spine 6f t5
3-state (the validated granularity; n=3 is inherited, not scanned).
References (non-gating): buy-hold, the untouched null, the Gaussian 6f
2-state and 6f 3-state (the leading candidate) — so the emission family's
contribution is attributable at each granularity.

Gate: each challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of the
matched-vol placebo. Full walk-forward span, same engine, same panel
(SPY/TLT/GLD). Changing any constant after seeing results invalidates the
race.
"""

from __future__ import annotations

from collections.abc import Callable
from math import lgamma, pi
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.regime import GaussianHMM
from spine.strategy import TunedSpineStrategy

__all__ = ["StudentTHMM", "run_race"]

GATE = {
    "nu": 5.0,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGERS = ("spine 6f t5 2-state", "spine 6f t5 3-state")

_MIN_VAR = 1e-10


class StudentTHMM(GaussianHMM):
    """GaussianHMM with diagonal Student-t emissions at a frozen dof.

    Only the emission density and its M-step change; init, forward/backward,
    state ordering, and the causal filter are inherited unchanged."""

    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__(n_states, n_iter, tol)
        self.nu = float(GATE["nu"])

    def _emission_logprob(self, x: np.ndarray) -> np.ndarray:
        assert self.means_ is not None and self.vars_ is not None
        d = x.shape[1]
        diff = x[:, None, :] - self.means_[None, :, :]
        delta2 = (diff**2 / self.vars_[None, :, :]).sum(axis=2)
        const = (
            lgamma((self.nu + d) / 2.0)
            - lgamma(self.nu / 2.0)
            - 0.5 * d * np.log(self.nu * pi)
        )
        return (
            const
            - 0.5 * np.log(self.vars_).sum(axis=1)[None, :]
            - 0.5 * (self.nu + d) * np.log1p(delta2 / self.nu)
        )

    def fit(self, x: np.ndarray) -> StudentTHMM:
        if x.ndim != 2 or len(x) < 10 * self.n_states:
            raise ValueError("not enough observations to fit HMM")
        self._init_params(x)
        n, k = len(x), self.n_states
        d = x.shape[1]
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            assert self.means_ is not None and self.vars_ is not None
            log_b = self._emission_logprob(x)
            alpha, scale = self._forward(log_b)
            beta = self._backward(log_b, scale)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            assert self.transmat_ is not None
            b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
            xi_sum = np.zeros((k, k))
            for t in range(n - 1):
                xi = self.transmat_ * np.outer(alpha[t], b[t + 1] * beta[t + 1])
                total = xi.sum()
                if total > 0:
                    xi_sum += xi / total

            # Latent-scale weights from the E-step parameters: tail days get
            # down-weighted instead of dragging the state's mean/variance.
            diff = x[:, None, :] - self.means_[None, :, :]
            delta2 = (diff**2 / self.vars_[None, :, :]).sum(axis=2)
            u = (self.nu + d) / (self.nu + delta2)
            w = gamma * u
            wsum = np.maximum(w.sum(axis=0), 1e-300)
            weights = np.maximum(gamma.sum(axis=0), 1e-300)
            self.means_ = (w.T @ x) / wsum[:, None]
            diff2 = (x[:, None, :] - self.means_[None, :, :]) ** 2
            self.vars_ = np.maximum(
                np.einsum("tk,tkd->kd", w, diff2) / weights[:, None], _MIN_VAR
            )
            self.transmat_ = xi_sum / np.maximum(
                xi_sum.sum(axis=1, keepdims=True), 1e-300
            )
            self.startprob_ = gamma[0]

            ll = float(np.sum(np.log(scale)) + log_b.max(axis=1).sum())
            if abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll
        self._order_states()
        return self


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    f6 = np.column_stack(
        [
            build_features(panel, store_dir),
            credit_spread_series(store_dir, panel.dates),
            rate_vol_feature(store_dir, panel.dates),
        ]
    )
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 2-state (ref)": lambda: TunedSpineStrategy(panel, f6),
        "spine 6f 3-state (leader ref)": lambda: TunedSpineStrategy(
            panel, f6, n_states=3
        ),
        CHALLENGERS[0]: lambda: TunedSpineStrategy(panel, f6, hmm_cls=StudentTHMM),
        CHALLENGERS[1]: lambda: TunedSpineStrategy(
            panel, f6, n_states=3, hmm_cls=StudentTHMM
        ),
    }
    runs = race_contenders(panel, contenders, first)
    verdicts = {
        label: amended_bar(runs[label], runs[NULL], panel, first, GATE)
        for label in CHALLENGERS
    }
    return {
        "gate": dict(GATE),
        "n_days": panel.n_days,
        "oos_span": (
            runs[NULL]["result"]["dates"][0],
            runs[NULL]["result"]["dates"][-1],
        ),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {
            "per_challenger": verdicts,
            "race_pass": any(v["passes"] for v in verdicts.values()),
        },
    }
