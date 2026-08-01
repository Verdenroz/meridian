"""Block-bootstrap Sharpe-difference diagnostic (non-gating).

Moving-block bootstrap over *paired* challenger/null daily returns, so
common market days stay aligned and only the strategy difference is
resampled. Reports the Sharpe-diff distribution the spec quotes for the
original race's hold decision. Seeded — the diagnostic is reproducible.
"""

from __future__ import annotations

import numpy as np

BLOCK_LEN = 21
N_RESAMPLES = 2000
SEED = 20260801


def _sharpe(excess: np.ndarray) -> float:
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def sharpe_diff_bootstrap(
    chall: np.ndarray, null: np.ndarray, cash: np.ndarray
) -> dict:
    n = len(chall)
    idx_blocks = np.arange(n - BLOCK_LEN + 1)
    n_blocks = int(np.ceil(n / BLOCK_LEN))
    rng = np.random.default_rng(SEED)

    diffs = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        starts = rng.choice(idx_blocks, size=n_blocks)
        take = np.concatenate([np.arange(s, s + BLOCK_LEN) for s in starts])[:n]
        diffs[i] = _sharpe(chall[take] - cash[take]) - _sharpe(null[take] - cash[take])

    point = _sharpe(chall - cash) - _sharpe(null - cash)
    lo, hi = np.percentile(diffs, [5, 95])
    return {
        "sharpe_diff": point,
        "ci90": (float(lo), float(hi)),
        "p_leq_0": float((diffs <= 0).mean()),
        "block_len": BLOCK_LEN,
        "n_resamples": N_RESAMPLES,
    }
