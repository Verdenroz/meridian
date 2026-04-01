"""Loss functions — Section 10.3 of meridian-spec.md.

Return head training:
  L = -DSR(w, r) + λ_turn · TurnoverPenalty(w) + λ_vol · VolatilityAux(w, r)

  DSR (Differential Sharpe Ratio):
    Online estimate of d/dη [E[R] / σ[R]] using exponential moving moments.
    Differentiable surrogate for Sharpe ratio that works with gradient descent.

  TurnoverPenalty:
    λ_turn · Σ_t Σ_i |w_{i,t} - w_{i,t-1}|
    Discourages excessive rebalancing (target: < 10% daily turnover)

  VolatilityAux:
    MSE between predicted portfolio volatility and realized volatility.
    Auxiliary task that improves risk-awareness of the backbone.

Sharpe head training (frozen backbone):
  L = -SharpeRatio(w, r)_{60d_rolling}

  Backbone frozen; only Meridian-S head parameters updated.
  Uses 60-day rolling Sharpe to encourage stability over time.

Hyperparameters:
  λ_turn = 0.1   — turnover penalty weight
  λ_vol  = 0.05  — volatility auxiliary weight
"""

from __future__ import annotations

# TODO: implement loss functions (Section 10.3 of meridian-spec.md)
