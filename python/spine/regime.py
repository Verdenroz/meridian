"""Regime detection: Gaussian HMM over market features, filtered causally.

Fitting uses Baum-Welch on training data only. Inference uses the *forward
filter* — P(state_t | observations up to t) — never smoothing, which would
condition on the future. States are ordered by the first feature's mean
(ascending), so the last state is always "stress" when feature 0 is a vol
measure. Initialization is deterministic: no RNG, reproducible refits.
"""

from __future__ import annotations

import numpy as np

_MIN_VAR = 1e-10


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM with K states."""

    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-6) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None

    def _emission_logprob(self, x: np.ndarray) -> np.ndarray:
        assert self.means_ is not None and self.vars_ is not None
        diff = x[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.log(2.0 * np.pi * self.vars_)[None, :, :] + diff**2 / self.vars_[None, :, :]
        ).sum(axis=2)

    def _init_params(self, x: np.ndarray) -> None:
        # Quantile-split on feature 0 (a vol measure): deterministic and
        # already close to the calm/stress structure we expect.
        k = self.n_states
        order = np.argsort(x[:, 0])
        chunks = np.array_split(order, k)
        self.means_ = np.stack([x[c].mean(axis=0) for c in chunks])
        self.vars_ = np.maximum(
            np.stack([x[c].var(axis=0) for c in chunks]), _MIN_VAR
        )
        self.transmat_ = np.full((k, k), 0.05 / max(k - 1, 1))
        np.fill_diagonal(self.transmat_, 0.95)
        self.startprob_ = np.full(k, 1.0 / k)

    def fit(self, x: np.ndarray) -> GaussianHMM:
        if x.ndim != 2 or len(x) < 10 * self.n_states:
            raise ValueError("not enough observations to fit HMM")
        self._init_params(x)
        n, k = len(x), self.n_states
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            log_b = self._emission_logprob(x)
            alpha, scale = self._forward(log_b)
            beta = self._backward(log_b, scale)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            assert self.transmat_ is not None
            b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
            xi_sum = np.zeros((k, k))
            for t in range(n - 1):
                xi = (
                    self.transmat_
                    * np.outer(alpha[t], b[t + 1] * beta[t + 1])
                )
                total = xi.sum()
                if total > 0:
                    xi_sum += xi / total
            weights = gamma.sum(axis=0)
            self.means_ = (gamma.T @ x) / weights[:, None]
            diff2 = (x[:, None, :] - self.means_[None, :, :]) ** 2
            self.vars_ = np.maximum(
                np.einsum("tk,tkd->kd", gamma, diff2) / weights[:, None], _MIN_VAR
            )
            self.transmat_ = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
            self.startprob_ = gamma[0]

            # scale omits the per-row max subtracted from log_b; add it back
            ll = float(np.sum(np.log(scale)) + log_b.max(axis=1).sum())
            if abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll
        self._order_states()
        return self

    def _order_states(self) -> None:
        """Sort states by mean of feature 0 so index K-1 is highest-vol."""
        assert self.means_ is not None and self.vars_ is not None
        assert self.transmat_ is not None and self.startprob_ is not None
        order = np.argsort(self.means_[:, 0])
        self.means_ = self.means_[order]
        self.vars_ = self.vars_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]
        self.startprob_ = self.startprob_[order]

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled forward pass; alpha[t] is the filtered posterior at t."""
        assert self.transmat_ is not None and self.startprob_ is not None
        n, k = log_b.shape
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        alpha = np.zeros((n, k))
        scale = np.zeros(n)
        state = self.startprob_ * b[0]
        for t in range(n):
            if t > 0:
                state = (alpha[t - 1] @ self.transmat_) * b[t]
            scale[t] = max(state.sum(), 1e-300)
            alpha[t] = state / scale[t]
        return alpha, scale

    def _backward(self, log_b: np.ndarray, scale: np.ndarray) -> np.ndarray:
        assert self.transmat_ is not None
        n, k = log_b.shape
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (self.transmat_ @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    def filter_posterior(self, x: np.ndarray) -> np.ndarray:
        """Causal filtered posterior P(state_t | x_1..t) for each row."""
        alpha, _ = self._forward(self._emission_logprob(x))
        return alpha

    def filter_step(self, alpha: np.ndarray, x_row: np.ndarray | None) -> np.ndarray:
        """Advance the filtered posterior one day; a None observation (missing
        data) applies the transition only."""
        assert self.transmat_ is not None
        state = alpha @ self.transmat_
        if x_row is not None:
            log_b = self._emission_logprob(x_row[None, :])[0]
            state = state * np.exp(log_b - log_b.max())
        return state / max(state.sum(), 1e-300)
