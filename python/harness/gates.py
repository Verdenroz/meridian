"""The charter's promotion bar as reusable mechanics.

`amended_bar` evaluates one challenger against the untouched null: an
outright after-cost Sharpe win passes on route (a); otherwise route (b)
requires non-inferior Sharpe, a max drawdown no worse than the registered
fraction of the null's, and a beat of the matched-vol placebo — the null
statically de-levered to the challenger's realized vol, so a drawdown edge
that a constant de-lever reproduces is exposed as exposure, not skill.
A (b)-route pass is *held, never shipped* on the motivating backtest alone.

The gate margins are required inputs: criteria live in each race's frozen
pre-registration, never as defaults here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from harness.panel import Panel
from harness.racing import run_contender
from harness.strategies import VolTarget
from harness.view import HistoryView


class ScaledVolTarget(VolTarget):
    """The null statically de-levered by a constant — the matched-vol
    placebo. Any drawdown edge a constant de-lever reproduces is exposure,
    not skill."""

    def __init__(self, symbol: str, scale: float) -> None:
        super().__init__(symbol)
        self.scale = scale

    def target_weights(self, view: HistoryView) -> dict[str, float] | None:
        vol = self._ewma_vol(view.returns_window(self.symbol, self.window))
        weight = 0.0 if vol is None else min(1.0, self.target_vol / vol) * self.scale
        if self._last_target is not None and abs(weight - self._last_target) < self.band:
            return None
        self._last_target = weight
        return {self.symbol: weight}


def amended_bar(
    chall: dict,
    null: dict,
    panel: Panel,
    first: int,
    gate: Mapping[str, float],
    oos_start: str | None = None,
    placebo_symbol: str = "SPY",
    run: Callable[..., dict] = run_contender,
) -> dict:
    """Apply the amended bar to one challenger; runs the matched-vol
    placebo only when the (b) branch needs it. `gate` must supply
    'noninferiority_margin' and 'maxdd_ratio'. `run` is injectable so the
    branch logic is testable without market data."""
    margin = float(gate["noninferiority_margin"])
    maxdd_ratio = float(gate["maxdd_ratio"])
    outright = chall["metrics"]["sharpe"] > null["metrics"]["sharpe"]
    verdict: dict[str, object] = {
        "outright_sharpe_win": bool(outright),
        "co_criterion": None,
        "passes": bool(outright),
        "route": "a" if outright else None,
    }
    if outright:
        return verdict

    noninferior = (
        chall["metrics"]["sharpe"] >= null["metrics"]["sharpe"] - margin
    )
    better_tail = (
        chall["metrics"]["max_drawdown"]
        <= maxdd_ratio * null["metrics"]["max_drawdown"]
    )
    co: dict[str, object] = {
        "noninferior_sharpe": bool(noninferior),
        "tail_two_thirds": bool(better_tail),
        "placebo": None,
    }
    if noninferior and better_tail:
        scale = min(
            1.0, chall["metrics"]["ann_vol"] / max(null["metrics"]["ann_vol"], 1e-9)
        )
        placebo = run(
            panel, lambda: ScaledVolTarget(placebo_symbol, scale), first, oos_start
        )
        co["placebo"] = {
            "scale": scale,
            "sharpe": placebo["metrics"]["sharpe"],
            "ann_vol": placebo["metrics"]["ann_vol"],
            "max_drawdown": placebo["metrics"]["max_drawdown"],
        }
        beats_placebo = chall["metrics"]["sharpe"] > placebo["metrics"]["sharpe"]
        co["beats_placebo"] = bool(beats_placebo)
        if beats_placebo:
            verdict["passes"] = True
            verdict["route"] = "b (held: re-race + behavior-match required)"
    verdict["co_criterion"] = co
    return verdict
