"""Cross-Asset Fusion Module — 2-layer MHA with sector bias (Section 7.6)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regime_aware_backbone import RMSNorm

# 11 GICS sectors (used for the sector bias matrix)
N_GICS_SECTORS = 11


class SectorBiasPrior(nn.Module):
    """Learned 11×11 sector affinity matrix applied as an additive attention bias.

    For each pair of tokens (i, j), the bias is:
      bias_ij = sector_affinity[sector_i, sector_j]

    Same-sector pairs typically receive a larger positive bias, encoding the
    prior that companies in the same sector share correlated information flows.

    Parameters: 11×11 = 121 (FP32, ~484 bytes).
    """

    def __init__(self, n_sectors: int = N_GICS_SECTORS) -> None:
        super().__init__()
        # Initialised near zero; model learns the sector affinity structure
        self.affinity = nn.Parameter(torch.zeros(n_sectors, n_sectors))

    def bias_matrix(
        self,
        sector_ids: torch.Tensor,  # (N_assets,) int64, -1 for non-asset tokens
        n_tokens: int,
    ) -> torch.Tensor:             # (n_tokens, n_tokens)
        """Build the full n_tokens × n_tokens additive bias matrix."""
        bias = torch.zeros(n_tokens, n_tokens, device=self.affinity.device)
        n_assets = sector_ids.shape[0]
        valid = sector_ids >= 0                                      # (n_assets,)
        valid_ids = sector_ids.clamp(min=0)                          # safe index; masked below
        pair_bias = self.affinity[valid_ids][:, valid_ids]           # (n_assets, n_assets)
        mask = valid.unsqueeze(1) & valid.unsqueeze(0)               # (n_assets, n_assets)
        bias[:n_assets, :n_assets] = pair_bias * mask.float()
        return bias


class FusionMHALayer(nn.Module):
    """One layer of multi-head self-attention for cross-asset fusion."""

    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj   = nn.Linear(d_model, d_model, bias=False)
        self.norm     = RMSNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                    # (B, N, D)
        bias: torch.Tensor | None = None,   # (N, N) additive attention bias
    ) -> torch.Tensor:                      # (B, N, D)
        residual = x
        x = self.norm(x)
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.qkv_proj(x).view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, Dh)

        attn_bias = None
        if bias is not None:
            attn_bias = bias.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.drop.p if self.training else 0.0,
        )  # (B, H, N, Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return residual + self.drop(self.o_proj(out))


class CrossAssetFusion(nn.Module):
    """Full cross-asset fusion module (Section 7.6).

    Input tokens (110+ total):
      - 100 per-asset tokens from RA-GDN backbone
      - 1 market-wide token
      - 1 portfolio state token
      - 8 episodic memory retrieval tokens

    Output: same shape as input — per-asset tokens enriched with cross-asset
    context via 2-layer MHA with sector bias.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 2,
        n_heads: int = 8,
        n_sectors: int = N_GICS_SECTORS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FusionMHALayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.sector_bias = SectorBiasPrior(n_sectors)

    def forward(
        self,
        asset_tokens: torch.Tensor,      # (B, N_assets, D)
        market_token: torch.Tensor,      # (B, D)
        portfolio_token: torch.Tensor,   # (B, D)
        episodic_tokens: torch.Tensor,   # (B, k, D)  — k episodic retrievals
        sector_ids: torch.Tensor,        # (N_assets,) int64  — GICS sector index per asset
        precomputed_bias: torch.Tensor | None = None,  # (N_total, N_total) — cached bias
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenate all tokens: (B, N_assets + 1 + 1 + k, D)
        market_exp    = market_token.unsqueeze(1)      # (B, 1, D)
        portfolio_exp = portfolio_token.unsqueeze(1)   # (B, 1, D)
        tokens = torch.cat(
            [asset_tokens, market_exp, portfolio_exp, episodic_tokens], dim=1
        )  # (B, N_total, D)

        if precomputed_bias is not None:
            bias = precomputed_bias
        else:
            N_total = tokens.shape[1]
            n_extras = N_total - asset_tokens.shape[1]
            extra_ids = torch.full((n_extras,), -1, dtype=torch.long, device=sector_ids.device)
            full_ids = torch.cat([sector_ids, extra_ids], dim=0)  # (N_total,)
            bias = self.sector_bias.bias_matrix(full_ids, N_total)  # (N_total, N_total)

        for layer in self.layers:
            tokens = layer(tokens, bias)

        # Split output back into asset tokens and context tokens
        n_assets = asset_tokens.shape[1]
        enriched_assets = tokens[:, :n_assets]      # (B, N_assets, D)
        market_out      = tokens[:, n_assets]        # (B, D)
        return enriched_assets, market_out
