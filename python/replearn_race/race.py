"""Representation-learning race: the second neural challenger, with the
objective shape the first one's failure prescribed. Pre-registered before
any result was computed.

The GRU-exceedance challenger failed decisively while its classical twin
won on identical features — the registered lesson was that the *decision*
should stay with the unsupervised filter, and a learned component should
compete on a different job. This race keeps the entire winning machinery —
3-state Gaussian HMM, tuner, allocator — and lets the network do only
representation: an unsupervised recurrent encoder compresses the 7f-turn
history into a state summary the HMM reads instead of the raw channels.
Hypothesis: recurrent memory can expose slow build-ups (vol-of-vol,
channel co-movement) that a first-order HMM on instantaneous features
cannot condition on. It predicts nothing but its own inputs — no return
or even risk target — so the §1.7 surface is minimal.

Model, fixed a priori (every constant frozen): single-layer GRU, hidden
16, trained per walk-forward window to predict the *next day's* 7f-turn
feature row (MSE; standardized on the window, non-finite zero-filled),
Adam lr 1e-3, 200 full-sequence epochs, grad clip 1.0, seed 20260801,
deterministic CUDA (registered flags as the learned race), eGPU with CPU
fallback. Representation: hidden states over the full sequence (params
frozen at the fit boundary; recurrence causal), PCA to 3 components
fitted on the training rows (deterministic SVD, each component's
largest-|loading| dimension forced positive). The regime model then runs
unchanged on [log realized vol, pc1, pc2, pc3] — feature 0 stays the
incumbent vol measure, so state ordering and the stress read are
inherited, and the learned block replaces only the six engineered
context channels.

Challenger: learned rep-HMM 3-state. References (non-gating): buy-hold,
the untouched null, spine 6f 3-state, spine 7f-turn 3-state (the
classical incumbent whose features feed the encoder — attributable
representation-for-representation).

Gate: the challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of
the matched-vol placebo. Full walk-forward span, same engine, same panel
(SPY/TLT/GLD). Changing any constant after seeing results invalidates
the race.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import Panel, build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from harness.view import HistoryView
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy
from turnover_race.race import turnover_feature

__all__ = ["RepTunedStrategy", "run_race"]

GATE = {
    "hidden": 16,
    "lr": 1e-3,
    "epochs": 200,
    "grad_clip": 1.0,
    "pca_components": 3,
    "n_states": 3,
    "seed": 20260801,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "learned rep-HMM 3-state"


def _encode_hidden(x_std: np.ndarray, train_len: int) -> np.ndarray:
    """Train the frozen-spec next-step GRU on the prefix, return hidden
    states for every row (params fixed at the fit boundary; causal)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch  # type: ignore[import-not-found]

    torch.manual_seed(int(GATE["seed"]))
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = x_std.shape[1]
    gru = torch.nn.GRU(d, int(GATE["hidden"]), 1, batch_first=True).to(device)
    head = torch.nn.Linear(int(GATE["hidden"]), d).to(device)
    params = list(gru.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=float(GATE["lr"]))
    loss_fn = torch.nn.MSELoss()

    xt = torch.tensor(x_std[None, :train_len], dtype=torch.float32, device=device)
    target = xt[0, 1:]
    for _ in range(int(GATE["epochs"])):
        opt.zero_grad()
        out, _ = gru(xt)
        pred = head(out[0, :-1])
        loss = loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(GATE["grad_clip"]))
        opt.step()

    with torch.no_grad():
        xa = torch.tensor(x_std[None, :], dtype=torch.float32, device=device)
        out, _ = gru(xa)
        hidden = out[0].cpu().numpy()
    return np.asarray(hidden, dtype=float)


def _pca_project(hidden: np.ndarray, train_len: int, k: int) -> np.ndarray:
    """Project onto the top-k PCs of the training rows; deterministic sign
    (largest-|loading| dimension of each component forced positive)."""
    train = hidden[:train_len]
    mean = train.mean(axis=0)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    comps = vt[:k]
    for j in range(k):
        lead = int(np.argmax(np.abs(comps[j])))
        if comps[j, lead] < 0:
            comps[j] = -comps[j]
    return (hidden - mean) @ comps.T


class RepTunedStrategy:
    """Per window: encode history with the frozen GRU spec, then hand the
    learned state summary to the unchanged 3-state tuner machinery."""

    def __init__(self, panel: Panel, features: np.ndarray) -> None:
        self.panel = panel
        self.features = features
        self._inner: TunedSpineStrategy | None = None

    def _learned_features(self, t: int) -> np.ndarray:
        finite = np.isfinite(self.features[: t + 1]).all(axis=1)
        if finite.sum() < 252:
            raise ValueError("not enough finite feature rows to train encoder")
        train = self.features[: t + 1][finite]
        mu = train.mean(axis=0)
        sigma = np.maximum(train.std(axis=0), 1e-12)
        x_std = np.where(
            np.isfinite(self.features), (self.features - mu) / sigma, 0.0
        )
        hidden = _encode_hidden(x_std, t + 1)
        pcs = _pca_project(hidden, t + 1, int(GATE["pca_components"]))
        return np.column_stack([self.features[:, 0], pcs])

    def fit(self, view: HistoryView) -> None:
        learned = self._learned_features(view.t)
        self._inner = TunedSpineStrategy(
            self.panel, learned, n_states=int(GATE["n_states"])
        )
        self._inner.fit(view)

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        assert self._inner is not None
        return self._inner.target_weights(view)


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    f6 = np.column_stack(
        [
            build_features(panel, store_dir),
            credit_spread_series(store_dir, panel.dates),
            rate_vol_feature(store_dir, panel.dates),
        ]
    )
    f7_turn = np.column_stack([f6, turnover_feature(store_dir, panel.dates)])
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 6f 3-state (ref)": lambda: TunedSpineStrategy(panel, f6, n_states=3),
        "spine 7f-turn 3-state (incumbent ref)": lambda: TunedSpineStrategy(
            panel, f7_turn, n_states=3
        ),
        CHALLENGER: lambda: RepTunedStrategy(panel, f7_turn),
    }
    runs = race_contenders(panel, contenders, first)
    verdict = amended_bar(runs[CHALLENGER], runs[NULL], panel, first, GATE)
    return {
        "gate": dict(GATE),
        "n_days": panel.n_days,
        "oos_span": (
            runs[NULL]["result"]["dates"][0],
            runs[NULL]["result"]["dates"][-1],
        ),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {"challenger": verdict, "race_pass": bool(verdict["passes"])},
    }
