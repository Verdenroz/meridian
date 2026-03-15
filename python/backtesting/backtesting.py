"""Historical backtesting framework — Section 12 of meridian-spec.md.

Integrates with finance-query's backtesting engine to run Meridian as a strategy.

Components:
  MeridianStrategy       — strategy wrapper implementing finance-query's Strategy trait
  BenchmarkComparisons   — SPY buy-and-hold, equal-weight, risk-parity baselines (12.2)
  WalkForwardValidator   — expanding-window OOS validation (12.3)
  MonteCarloAnalyzer     — confidence interval estimation (12.4)
  RegimeSegmentAnalyzer  — per-regime performance breakdown (12.5)
"""

from __future__ import annotations

# TODO: implement backtesting framework (Section 12 of meridian-spec.md)
