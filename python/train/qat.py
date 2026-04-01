"""Quantization-Aware Training (QAT) — Section 10.5 of meridian-spec.md.

Strategy: BitNet b1.58 — absmean ternary weights {-1, 0, +1} with FP16 scaling factors.

  Ternary quantization (per weight matrix W):
    scale = mean(|W|)
    W_ternary = round(W / scale).clamp(-1, 1)

  Straight-through estimator (STE):
    Forward:  use W_ternary for compute
    Backward: pass gradients through as if W_ternary == W (identity STE)

  Precision highway (FP16):
    Keeps full-precision for: output heads, RMSNorm params, regime detector,
    MarketReader, episodic memory projections.

  Schedule:
    - Start from the meta-learning checkpoint
    - ~50K steps on combined pre-training + portfolio data
    - Validate: quantized Sharpe within 5% of full-precision Sharpe on test set
"""

from __future__ import annotations

# TODO: implement QAT training loop (Section 10.5 of meridian-spec.md)
