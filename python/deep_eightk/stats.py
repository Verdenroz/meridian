"""Block-bootstrap Sharpe-difference diagnostic (non-gating).

Mechanics live in harness.stats; the constants below are this race's frozen
pre-registration (block length, resample count, seed) and never change.
"""

from __future__ import annotations

import numpy as np
from harness.stats import sharpe_diff_bootstrap as _bootstrap

BLOCK_LEN = 21
N_RESAMPLES = 2000
SEED = 20260801


def sharpe_diff_bootstrap(
    chall: np.ndarray, null: np.ndarray, cash: np.ndarray
) -> dict:
    return _bootstrap(
        chall, null, cash, block_len=BLOCK_LEN, n_resamples=N_RESAMPLES, seed=SEED
    )
