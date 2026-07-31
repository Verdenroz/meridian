"""Walk-forward backtest harness and the null baseline.

One harness for everything: the null baseline (SPY + vol-target), the classical
spine, and every challenger are evaluated by the same engine, same cost model,
same calendar. All headline claims are relative to the null run. Nothing else
in Meridian is evaluated until this exists.
"""

from harness.engine import GuardrailViolation, run_backtest
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward, walk_forward_splits

__all__ = [
    "BuyHold",
    "GuardrailViolation",
    "Panel",
    "VolTarget",
    "build_panel",
    "compute_metrics",
    "run_backtest",
    "run_walk_forward",
    "walk_forward_splits",
]
