"""Portfolio output heads — Section 7.8 of meridian-spec.md.

Two heads share the same backbone; only one is active per inference call.

Meridian-R (return-maximizing head):
  Input:  cross-asset fusion output (N_assets × D)
  Output: Linear(D → 1) per asset → softmax → long-only weights ∈ Δ^{N-1}

Meridian-S (Sharpe-maximizing head):
  Input:  same cross-asset fusion output
  Output: Linear(D → 1) per asset → softmax(T) with learned temperature scalar
  Temperature init = 1.0; lower T → more concentrated, higher T → more uniform

Both heads enforce:
  Σ w_i = 1.0,  w_i ≥ 0  ∀ i  (long-only, fully invested)

Auxiliary surprise head (Section 7.9):
  Input: market token from fusion output (B, D)
  Output: scalar aggregate-surprise estimate for Tier 1 adaptation calibration
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeridianR(nn.Module):
    """Return-maximizing portfolio head.

    Scores each asset independently via a linear projection over its
    fused D-dim representation, then applies softmax to produce long-only
    allocation weights that sum to 1.

    Parameters: D × 1 linear = 768 scalars per asset slot, shared across assets
    (~197K total: Linear(768, 256) + Linear(256, 1) with bias).
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, asset_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            asset_tokens: (B, N_assets, D) — fused per-asset representations

        Returns:
            weights: (B, N_assets) — allocation weights, sum to 1, all ≥ 0
        """
        logits = self.score(asset_tokens).squeeze(-1)  # (B, N)
        return F.softmax(logits, dim=-1)               # (B, N)


class MeridianS(nn.Module):
    """Sharpe-maximizing portfolio head.

    Same architecture as MeridianR but with a learned temperature scalar.
    Lower temperature concentrates weight on the highest-scoring assets,
    which the Sharpe objective rewards when those assets have low correlation
    and high risk-adjusted returns.

    Parameters: same as MeridianR plus one scalar (temperature).
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # log_temperature for numerical stability; exp gives temperature > 0
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.1, max=10.0)

    def forward(self, asset_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            asset_tokens: (B, N_assets, D)

        Returns:
            weights: (B, N_assets) — temperature-scaled softmax allocation
        """
        logits = self.score(asset_tokens).squeeze(-1)  # (B, N)
        return F.softmax(logits / self.temperature, dim=-1)


class AuxHeadSurprise(nn.Module):
    """Auxiliary surprise prediction head (Section 7.9).

    Predicts the aggregate news/event surprise scalar at t+1 from the
    market-wide fusion token. Used by the Tier 1 adaptation system to
    calibrate its surprise threshold.

    Input:  market token from CrossAssetFusion output (B, D)
    Output: scalar surprise estimate (B,)
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, market_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_token: (B, D) — market-wide token from fusion output

        Returns:
            surprise: (B,) — predicted aggregate surprise ≥ 0
        """
        return F.softplus(self.net(market_token)).squeeze(-1)
