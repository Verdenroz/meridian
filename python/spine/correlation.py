"""Correlation estimation: Ledoit–Wolf constant-correlation shrinkage and
causal rolling correlations for the safe-asset checks."""

from __future__ import annotations

import numpy as np


def ledoit_wolf_constant_corr(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward the constant-correlation target
    (Ledoit & Wolf 2004). `returns` is T×N; returns (covariance, intensity)."""
    t_obs, n = returns.shape
    if t_obs < 2 or n < 2:
        raise ValueError("need T>=2 and N>=2")
    x = returns - returns.mean(axis=0)
    sample = x.T @ x / t_obs
    std = np.sqrt(np.diag(sample))
    corr = sample / np.outer(std, std)
    mean_corr = (corr.sum() - n) / (n * (n - 1))
    target = mean_corr * np.outer(std, std)
    np.fill_diagonal(target, std**2)

    # pi-hat: sum of asymptotic variances of sample covariance entries
    y = np.einsum("ti,tj->tij", x, x)
    pi_matrix = ((y - sample) ** 2).mean(axis=0)
    pi_hat = pi_matrix.sum()
    # rho-hat: diagonal exactly, off-diagonal via the usual approximation
    theta_ii = np.einsum("tij->ij", (y - sample) * (x**2 - std**2)[:, :, None]) / t_obs
    rho_off = mean_corr * (np.outer(1.0 / std, std) * theta_ii)
    rho_hat = np.trace(pi_matrix) + (rho_off.sum() - np.trace(rho_off))
    gamma_hat = ((target - sample) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    intensity = float(np.clip(kappa / t_obs, 0.0, 1.0))
    return intensity * target + (1.0 - intensity) * sample, intensity


def rolling_correlation(a: np.ndarray, b: np.ndarray, window: int = 60) -> np.ndarray:
    """out[t] = corr(a, b) over t-window+1..t; NaN when the window has any NaN
    or either side is degenerate. Causal by construction."""
    n = len(a)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        wa = a[t - window + 1 : t + 1]
        wb = b[t - window + 1 : t + 1]
        if np.isnan(wa).any() or np.isnan(wb).any():
            continue
        if wa.std() == 0 or wb.std() == 0:
            continue
        out[t] = float(np.corrcoef(wa, wb)[0, 1])
    return out
