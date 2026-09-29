"""Calm-state posterior at the prior close, recomputed from the leader's
exact feature matrix. Never read from the shadow log, which records weights,
not posteriors."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from credit_race.race import RACE_SYMBOLS
from harness.panel import build_panel
from shadow.contenders import build_contenders
from spine.strategy import SpineStrategy, TunedSpineStrategy

LEADER = "spine7f_turn_3state"
N_STATES = 3


def calm_posteriors(store_dir: Path, sessions: list[date]) -> dict[date, float]:
    """The session's own daily bar must be in the panel, so the prior close
    is the previous trading day and never a stale earlier one."""
    panel = build_panel(store_dir, RACE_SYMBOLS)
    leader = build_contenders(panel, store_dir)[LEADER]
    assert isinstance(leader, TunedSpineStrategy)
    features = leader.features
    index = {d: i for i, d in enumerate(panel.dates)}
    out = {}
    for session in sessions:
        t = index[session.isoformat()] - 1
        out[session] = _calm(panel, features, t)
    return out


def _calm(panel, features: np.ndarray, t: int) -> float:
    fitted = SpineStrategy(panel, features, n_states=N_STATES).fit_regime(t)
    assert fitted.hmm.startprob_ is not None
    alpha = fitted.hmm.startprob_.copy()
    for row in features[: t + 1]:
        z = (row - fitted.mu) / fitted.sigma if np.isfinite(row).all() else None
        alpha = fitted.hmm.filter_step(alpha, z)
    return float(alpha[0])
