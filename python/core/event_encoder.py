"""EventEncoder — discrete financial events → D-dimensional tokens (Section 6.4)."""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn


class EventKind(IntEnum):
    INSIDER_BUY             = 0
    INSIDER_SELL            = 1
    ANALYST_UPGRADE         = 2
    ANALYST_DOWNGRADE       = 3
    EARNINGS_BEAT           = 4
    EARNINGS_MISS           = 5
    SEC_FILING_10K          = 6
    SEC_FILING_10Q          = 7
    SEC_FILING_8K           = 8
    SEC_FILING_OTHER        = 9
    FOMC_DECISION           = 10
    MACRO_RELEASE           = 11
    NEWS_HIGH_IMPACT        = 12
    DIVIDEND_ANNOUNCED      = 13
    SPLIT_ANNOUNCED         = 14
    OPTIONS_UNUSUAL_ACTIVITY = 15
    TRENDING_ENTRY          = 16
    RATING_CHANGE           = 17


N_EVENT_TYPES = 18


class EventEncoder(nn.Module):
    """Encodes a batch of events into D-dimensional tokens with time decay.

    Architecture:
      type_emb:      Embedding(18, 64)
      magnitude_proj: Linear(1, 64)
      asset_emb:     Embedding(100, 64)
      out_proj:      Linear(192, D)
      decay:         token *= exp(-λ × minutes_since_event)

    ~50K parameters, kept in FP16 at inference.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_event_types: int = N_EVENT_TYPES,
        n_assets: int = 100,
        event_emb_dim: int = 64,
        decay_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.decay_lambda = decay_lambda
        self.type_emb = nn.Embedding(n_event_types, event_emb_dim)
        self.magnitude_proj = nn.Linear(1, event_emb_dim)
        self.asset_emb = nn.Embedding(n_assets, event_emb_dim)
        self.out_proj = nn.Linear(event_emb_dim * 3, d_model)

    def forward(
        self,
        event_type: torch.Tensor,    # (B,) int64
        magnitude: torch.Tensor,     # (B, 1) float32 — normalised event magnitude
        asset_id: torch.Tensor,      # (B,) int64 — 0-indexed asset in portfolio
        minutes_since: torch.Tensor, # (B,) float32 — minutes elapsed since event
    ) -> torch.Tensor:               # (B, D)
        type_emb  = self.type_emb(event_type)           # (B, 64)
        mag_emb   = self.magnitude_proj(magnitude)       # (B, 64)
        asset_emb = self.asset_emb(asset_id)             # (B, 64)
        combined  = torch.cat([type_emb, mag_emb, asset_emb], dim=-1)  # (B, 192)
        token     = self.out_proj(combined)              # (B, D)
        decay     = torch.exp(-self.decay_lambda * minutes_since).unsqueeze(-1)  # (B, 1)
        return token * decay
