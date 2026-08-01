"""Walk-forward backtest harness and the null baseline.

One harness for everything: the null baseline (SPY + vol-target), the classical
spine, and every challenger are evaluated by the same engine, same cost model,
same calendar. All headline claims are relative to the null run. Nothing else
in Meridian is evaluated until this exists.
"""

from harness.engine import GuardrailViolation, run_backtest
from harness.gates import ScaledVolTarget, amended_bar
from harness.metrics import compute_metrics
from harness.panel import Panel, build_panel
from harness.racing import metrics_table, plain, race_contenders, run_contender
from harness.signals import BrakeVolTarget, brake_series, causal_zscores
from harness.strategies import BuyHold, VolTarget
from harness.walkforward import run_walk_forward, walk_forward_splits

__all__ = [
    "BrakeVolTarget",
    "BuyHold",
    "GuardrailViolation",
    "Panel",
    "ScaledVolTarget",
    "VolTarget",
    "amended_bar",
    "brake_series",
    "build_panel",
    "causal_zscores",
    "compute_metrics",
    "metrics_table",
    "plain",
    "race_contenders",
    "run_backtest",
    "run_contender",
    "run_walk_forward",
    "walk_forward_splits",
]
