"""Learned-regime race: a recurrent stress filter as the first neural
challenger, pre-registered before any result was computed.

The four-state race closed n-scanning and routed regime-capacity work to
learned challengers; the t-emission race then showed the cheap in-family
upgrade does not displace the Gaussian 3-state. This race fields the first
genuinely learned filter, per the charter's challenger discipline (§1.7):
the classical spine stays production; the neural model must beat the null
in the same engine to even be held. The hypothesis: the HMM's linear-
Gaussian emissions and first-order state memory may miss slow build-ups
and nonlinear channel interactions; a small recurrent network reading the
same features can learn the mapping from history to near-term stress
directly. It predicts *risk* (forward realized vol exceedance), never
returns — the charter's line.

Model, fixed a priori (every constant frozen here): single-layer GRU,
hidden 32, over the leader's 7f-turn feature matrix standardized on the
training window (non-finite rows zero-filled post-standardization); linear
head → sigmoid = p_stress. Label: 1 if the *forward 21-session* SPY
realized vol exceeds the 70th percentile of the training window's forward
vols (risk exceedance, computed strictly inside the train span). Training:
BCE on labeled train rows, Adam lr 1e-3, 200 full-sequence epochs, grad
clip 1.0, seed 20260801, refit per walk-forward window exactly like the
HMM. Inference is causal by construction (left-to-right recurrence with
parameters frozen at the window's fit boundary). Determinism registered:
CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms; the
eGPU (RTX 3090) runs training, CPU is the registered fallback when CUDA is
absent. The allocator layer is inherited unchanged: p_stress feeds the
same tuner over the same frozen mapping candidates.

Challenger: learned GRU-stress on 7f-turn features. References
(non-gating): buy-hold, the untouched null, spine 6f 3-state, spine
7f-turn 3-state — the classical incumbent on identical features, so the
learned/classical comparison is attributable feature-for-feature.

Gate: the challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held route
(b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's + beat of
the matched-vol placebo. Full walk-forward span, same engine, same panel
(SPY/TLT/GLD). Changing any constant after seeing results invalidates the
race.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy, run_backtest
from harness.gates import amended_bar
from harness.panel import Panel, build_panel
from harness.racing import race_contenders
from harness.strategies import BuyHold, VolTarget
from harness.view import HistoryView
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import (
    TUNER_CANDIDATES,
    AllocatorConfig,
    SpineStrategy,
    TunedSpineStrategy,
)
from spine.vol import forward_realized_vol
from turnover_race.race import turnover_feature

__all__ = ["GruTunedStrategy", "run_race"]

GATE = {
    "hidden": 32,
    "layers": 1,
    "lr": 1e-3,
    "epochs": 200,
    "grad_clip": 1.0,
    "horizon": 21,
    "stress_pct": 70.0,
    "seed": 20260801,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "learned GRU-stress 7f-turn"


def _train_and_predict(x_std: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Train the frozen-spec GRU on the labeled prefix, return p_stress for
    every row (parameters fixed at the fit boundary; recurrence is causal)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch  # type: ignore[import-not-found]

    torch.manual_seed(int(GATE["seed"]))
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = x_std.shape[1]
    gru = torch.nn.GRU(
        n_features, int(GATE["hidden"]), int(GATE["layers"]), batch_first=True
    ).to(device)
    head = torch.nn.Linear(int(GATE["hidden"]), 1).to(device)
    params = list(gru.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=float(GATE["lr"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_len = int(np.flatnonzero(valid)[-1]) + 1
    xt = torch.tensor(x_std[None, :train_len], dtype=torch.float32, device=device)
    yt = torch.tensor(y[valid], dtype=torch.float32, device=device)
    idx = torch.tensor(np.flatnonzero(valid), dtype=torch.long, device=device)
    for _ in range(int(GATE["epochs"])):
        opt.zero_grad()
        out, _ = gru(xt)
        logits = head(out[0]).squeeze(-1)[idx]
        loss = loss_fn(logits, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(GATE["grad_clip"]))
        opt.step()

    with torch.no_grad():
        xa = torch.tensor(x_std[None, :], dtype=torch.float32, device=device)
        out, _ = gru(xa)
        p = torch.sigmoid(head(out[0]).squeeze(-1)).cpu().numpy()
    return np.asarray(p, dtype=float)


class PStressSpine(SpineStrategy):
    """SpineStrategy with the HMM posterior replaced by a precomputed
    causal p_stress series; allocator, safe-asset and band logic inherited."""

    def __init__(
        self,
        panel: Panel,
        features: np.ndarray,
        p_series: np.ndarray,
        config: AllocatorConfig | None = None,
    ) -> None:
        super().__init__(panel, features, config=config)
        self._p_series = p_series

    def fit(self, view: HistoryView) -> None:
        self._last_target = None

    def _advance_filter(self, t: int) -> None:
        p = float(self._p_series[t])
        self._alpha = np.array([1.0 - p, p])


class GruTunedStrategy:
    """Mirror of TunedSpineStrategy with the GRU stress filter: fit trains
    the network on the window's history, then races the frozen allocator
    candidates on the training span in the real engine."""

    def __init__(self, panel: Panel, features: np.ndarray) -> None:
        self.panel = panel
        self.features = features
        self.chosen: AllocatorConfig | None = None
        self._inner: PStressSpine | None = None

    def _fit_pstress(self, t: int) -> np.ndarray:
        finite = np.isfinite(self.features[: t + 1]).all(axis=1)
        if finite.sum() < 252:
            raise ValueError("not enough finite feature rows to train")
        train = self.features[: t + 1][finite]
        mu = train.mean(axis=0)
        sigma = np.maximum(train.std(axis=0), 1e-12)
        x_std = np.where(
            np.isfinite(self.features), (self.features - mu) / sigma, 0.0
        )
        returns = np.array(
            [np.nan if r is None else r for r in self.panel.returns["SPY"]]
        )
        fwd = forward_realized_vol(returns, int(GATE["horizon"]))
        valid = np.zeros(len(self.features), dtype=bool)
        horizon = int(GATE["horizon"])
        upto = max(0, t + 1 - horizon)
        valid[:upto] = np.isfinite(fwd[:upto]) & np.isfinite(
            self.features[:upto]
        ).all(axis=1)
        thr = float(np.percentile(fwd[valid], float(GATE["stress_pct"])))
        y = (fwd > thr).astype(float)
        return _train_and_predict(x_std, y, valid)

    def _train_sharpe(self, result: dict) -> float:
        returns = np.array(result["returns"])
        cash = dict(zip(self.panel.dates, self.panel.cash_returns))
        excess = returns - np.array([cash[d] for d in result["dates"]])
        std = float(excess.std(ddof=1))
        return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    def fit(self, view: HistoryView) -> None:
        t = view.t
        p_series = self._fit_pstress(t)
        eval_start = int(np.argmax(np.isfinite(self.features[: t + 1]).all(axis=1)))
        best_score = -np.inf
        for config in TUNER_CANDIDATES:
            candidate = PStressSpine(self.panel, self.features, p_series, config)
            result = run_backtest(self.panel, candidate, start=eval_start, end=t)
            score = self._train_sharpe(result)
            if score > best_score:
                best_score, self.chosen = score, config
        assert self.chosen is not None
        self._inner = PStressSpine(self.panel, self.features, p_series, self.chosen)
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
        CHALLENGER: lambda: GruTunedStrategy(panel, f7_turn),
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
