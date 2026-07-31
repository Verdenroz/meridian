"""Paper trading engine — Section 13 of meridian-spec.md.

Simulates live trading on real-time finance-query data with realistic friction.

Components:
  PaperBroker            — executes simulated trades (5 bps slippage, 10 bps commission)
  PerformanceTracker     — SQLite-backed per-minute and daily metrics (13.2)
  AdaptationMonitor      — tracks 3-tier adaptation health (13.3)
  GraduationChecker      — evaluates paper→live graduation criteria (13.4)

Graduation criteria (all must pass over 6-month period):
  - Annualized Sharpe > 1.0
  - Max drawdown < 20%
  - Cumulative return > 0%
  - Walk-forward consistency ratio > 0.6
  - Adaptive Sharpe > Frozen Sharpe
  - State stability (no monotonic growth)
  - Uptime > 99.5%
  - Inference latency p99 < 100 ms
  - Data pipeline reliability < 1% missed minutes
"""

from __future__ import annotations

# TODO: implement paper trading engine (Section 13 of meridian-spec.md)
