"""Shared statistics for races and event studies.

Rank statistics use normal approximations (sample sizes here are hundreds+).
The block bootstrap takes its block length, resample count, and seed as
required arguments — those are pre-registered per study, never defaulted.
"""

from __future__ import annotations

import math

import numpy as np


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """One-sided test that `a` is stochastically greater than `b`.
    Returns (rank-biserial effect size in [-1,1], p-value). Normal
    approximation with tie correction — fine for n >= ~20 per side."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("need at least 2 observations per sample")
    combined = np.concatenate([a, b])
    ranks = _rank_with_ties(combined)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    n = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float((counts**3 - counts).sum()) / (n * (n - 1))
    var_u = n1 * n2 / 12.0 * ((n + 1) - tie_term)
    if var_u <= 0:
        return 0.0, 1.0
    z = (u1 - mean_u) / math.sqrt(var_u)
    p = 1.0 - _normal_cdf(z)
    effect = 2.0 * u1 / (n1 * n2) - 1.0
    return effect, p


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation with one-sided (positive) p-value."""
    if len(x) != len(y) or len(x) < 10:
        raise ValueError("need matched samples of at least 10")
    rho = float(np.corrcoef(_rank_with_ties(x), _rank_with_ties(y))[0, 1])
    z = rho * math.sqrt(max(len(x) - 3, 1))  # Fisher approximation
    return rho, 1.0 - _normal_cdf(z)


def sharpe_diff_bootstrap(
    chall: np.ndarray,
    null: np.ndarray,
    cash: np.ndarray,
    *,
    block_len: int,
    n_resamples: int,
    seed: int,
) -> dict:
    """Moving-block bootstrap over *paired* challenger/null daily returns,
    so common market days stay aligned and only the strategy difference is
    resampled. Seeded — the diagnostic is reproducible."""
    n = len(chall)
    idx_blocks = np.arange(n - block_len + 1)
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.choice(idx_blocks, size=n_blocks)
        take = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        diffs[i] = _sharpe(chall[take] - cash[take]) - _sharpe(null[take] - cash[take])

    point = _sharpe(chall - cash) - _sharpe(null - cash)
    lo, hi = np.percentile(diffs, [5, 95])
    return {
        "sharpe_diff": point,
        "ci90": (float(lo), float(hi)),
        "p_leq_0": float((diffs <= 0).mean()),
        "block_len": block_len,
        "n_resamples": n_resamples,
    }


def _sharpe(excess: np.ndarray) -> float:
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
