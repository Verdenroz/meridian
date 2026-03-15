"""Regime-Aware Gated DeltaNet (RA-GDN) backbone — Section 7.3.

Architecture:
  8 RA-GDN blocks + 3 FMoE blocks (interleaved at positions 3, 6, 8)
  Shared weights across 100 assets; separate recurrent state per asset.
  Event tokens injected at their natural arrival time.

RA-GDN block recurrence (per asset, per timestep):
  S_t = α_t(regime) · S_{t-1} + β_t(regime) · k_t ⊗ (v_t − α_t · S_{t-1}ᵀ k_t)

  where:
    α_t = sigmoid(W_α · [h_t, regime_emb])   — regime-conditioned forgetting gate
    β_t = sigmoid(W_β · [h_t, regime_emb])   — regime-conditioned update strength
    k_t, v_t, q_t — linear projections of input
    h_t = S_t · q_t                           — output (O(1) recall)

FMoE block (Section 7.4):
  4 experts per block, top-2 active per forward pass.
  Expert routing: softmax over regime posterior (external routing, not learned).
  Total FMoE capacity: 3 × 4 = 12 expert FFN sub-networks.

VSN input gating (Section 7.1):
  gates = softmax(Linear([regime_emb], 288)) — per-feature importance
  x_selected = gates ⊙ features_288

Parameters:
  RA-GDN (8 blocks): ~17.4M, ternary
  FMoE (3 × 4 experts): ~42.5M total, ~21.3M active, ternary
"""

from __future__ import annotations

# TODO: implement RA-GDN backbone (Section 7.3 of meridian-spec.md)
# Extend FLA GatedDeltaNet with regime conditioning
