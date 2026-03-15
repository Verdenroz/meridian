"""Portfolio output heads — Section 7.8 of meridian-spec.md.

Two heads share the same backbone; only one is active per inference call.

Meridian-R (return-maximizing head):
  Input:  cross-asset fusion output (100 × D)
  Output: raw logits (100,) → softmax → allocation weights w ∈ Δ^99
  Leverage: weights sum to 1.0, all non-negative (long-only)
  Parameters: ~197K, FP16

Meridian-S (Sharpe-maximizing head):
  Input:  same cross-asset fusion output
  Output: raw logits (100,) → softmax with temperature → allocation weights
  Temperature: learned scalar, initialized to 1.0
  Parameters: ~197K, FP16

Both heads include a leverage normalization layer that enforces:
  Σ w_i = 1.0,  w_i ≥ 0  ∀ i ∈ {1..100}

Auxiliary prediction head (Section 7.9):
  Predicts aggregate surprise at t+1 (used for Tier 1 adaptation calibration)
  Input: portfolio-level summary vector
  Output: scalar surprise estimate
  Parameters: ~100K, FP16
"""

from __future__ import annotations

# TODO: implement portfolio output heads (Section 7.8 of meridian-spec.md)
