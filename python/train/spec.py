"""Architecture hyperparameter contracts for Meridian v3.0.

Key hyperparameters (Appendix C of meridian-spec.md):

  Model dimensions:
    D = 768                         — hidden dimension
    N_ASSETS = 100                  — max portfolio assets
    N_REGIMES = 4                   — HMM regime states
    REGIME_EMB_DIM = 64             — regime embedding dimension

  RA-GDN backbone:
    N_GDN_BLOCKS = 8                — RA-GDN blocks
    N_FMOE_BLOCKS = 3               — FMoE blocks interleaved
    N_EXPERTS = 4                   — experts per FMoE block
    TOP_K_EXPERTS = 2               — active experts per forward pass

  Temporal cascade:
    CASCADE_LEVELS = 3              — hourly / daily / weekly GDN stacks

  Cross-asset fusion:
    N_FUSION_LAYERS = 2             — MHA layers
    N_FUSION_HEADS = 8              — attention heads
    N_EPISODIC_RETRIEVALS = 8       — FAISS k-NN retrievals per step

  Event encoder:
    N_EVENT_TYPES = 18              — discrete event categories
    EVENT_EMB_DIM = 64              — per-component embedding dim

  MarketReader (Section 7.10):
    HEADLINE_DIM = 4                — sentiment vector
    SUMMARY_DIM = 32                — topic + sentiment vector
    DEEP_DIM = 64                   — full text embedding

  Training:
    PHASE1_STEPS = 200_000          — self-supervised pre-training
    PHASE2A_STEPS = 50_000          — return head fine-tuning
    PHASE2B_STEPS = 30_000          — Sharpe head (frozen backbone)
    PHASE3_STEPS = 100_000          — meta-learning for plasticity
    PHASE4_STEPS = 50_000           — quantization-aware training
"""

from __future__ import annotations

# TODO: implement hyperparameter dataclass with validate() method
# (Appendix C of meridian-spec.md)
