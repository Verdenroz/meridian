"""Cross-asset fusion module: spec §6.8.

Two layers of full multi-head attention (all asset/context tokens attend to each
other) with a trainable 11×11 sector affinity bias added to asset–asset attention
logits. Each layer also has a SwiGLU feed-forward network.

Token set fed into fusion (spec §6.8):
  [N_assets asset tokens | 1 market token | 1 portfolio token | k episodic tokens]
  = 100 + 1 + 1 + k  (k=8 default at inference; 0 during Phase 1 pre-training)

The sector bias prior encodes GICS sector relationships. Index 11 is a reserved
"no-sector" slot assigned to market/portfolio/episodic tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .regime_aware_backbone import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn

    from .regime_aware_backbone import RMSNorm

else:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "CrossAssetFusion requires torch. Install with: pip install -e '.[train]'"
        )


# GICS sector count; index n_sectors is the "no-sector" sentinel for non-asset tokens
_N_SECTORS = 11
_NO_SECTOR = _N_SECTORS


@dataclass(frozen=True)
class FusionConfig:
    model_dim: int = 768
    n_heads: int = 8     # 8 × 96 = 768
    d_head: int = 96
    n_layers: int = 2
    n_sectors: int = _N_SECTORS
    ffn_dim: int = 1536  # 2 × model_dim


if TORCH_AVAILABLE:

    class _SwiGLU(nn.Module):
        """SwiGLU(x) = SiLU(W_gate·x) ⊙ (W_up·x), projected down by W_down."""

        def __init__(self, in_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.gate = nn.Linear(in_dim, hidden_dim)
            self.up   = nn.Linear(in_dim, hidden_dim)
            self.down = nn.Linear(hidden_dim, in_dim)

        def forward(self, x: Tensor) -> Tensor:
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class _FusionLayer(nn.Module):
        """Single cross-asset fusion layer: MHA with sector bias + SwiGLU FFN."""

        def __init__(self, config: FusionConfig) -> None:
            super().__init__()
            assert config.n_heads * config.d_head == config.model_dim, (
                "n_heads * d_head must equal model_dim"
            )
            c = config
            self.n_heads = c.n_heads
            self.d_head  = c.d_head
            self.scale   = math.sqrt(c.d_head)

            self.q_proj = nn.Linear(c.model_dim, c.model_dim)
            self.k_proj = nn.Linear(c.model_dim, c.model_dim)
            self.v_proj = nn.Linear(c.model_dim, c.model_dim)
            self.o_proj = nn.Linear(c.model_dim, c.model_dim)

            self.attn_norm = RMSNorm(c.model_dim)
            self.ffn_norm  = RMSNorm(c.model_dim)
            self.ffn       = _SwiGLU(c.model_dim, c.ffn_dim)

        def forward(self, x: Tensor, sector_bias: Tensor | None) -> Tensor:
            """
            x:           [B, N_tokens, D]
            sector_bias: [B, N_tokens, N_tokens] or None — added to attn logits
            """
            B, N, _ = x.shape
            h = self.n_heads
            d = self.d_head

            # MHA
            residual = x
            x_norm = self.attn_norm(x)
            Q = self.q_proj(x_norm).view(B, N, h, d).transpose(1, 2)  # [B, h, N, d]
            K = self.k_proj(x_norm).view(B, N, h, d).transpose(1, 2)
            V = self.v_proj(x_norm).view(B, N, h, d).transpose(1, 2)

            logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, N, N]
            if sector_bias is not None:
                logits = logits + sector_bias.unsqueeze(1)  # broadcast over heads

            attn = F.softmax(logits, dim=-1)
            out  = torch.matmul(attn, V)                            # [B, h, N, d]
            out  = out.transpose(1, 2).reshape(B, N, h * d)
            x    = residual + self.o_proj(out)

            # SwiGLU FFN
            x = x + self.ffn(self.ffn_norm(x))
            return x

    class CrossAssetFusion(nn.Module):
        """Cross-asset fusion: full attention over asset + context tokens.

        Sector bias prior (spec §6.8): trainable 11×11 GICS affinity matrix S.
        attention_logits[i,j] += S[sector_i, sector_j] for asset token pairs.
        Non-asset tokens (market, portfolio, episodic) use sentinel sector index 11
        which maps to a separate row/col in a 12×12 extended matrix (zero-initialized).

        Usage:
            fusion = CrossAssetFusion()
            out_asset_tokens = fusion(
                asset_tokens,    # [B, N_assets, D]
                context_tokens,  # [B, N_ctx, D]  (market + portfolio + episodic)
                sector_ids,      # [B, N_assets]  int, values 0–10
            )
            # returns [B, N_assets, D] — only asset tokens, context stripped
        """

        def __init__(self, config: FusionConfig = FusionConfig()) -> None:
            super().__init__()
            self.config = config

            self.layers = nn.ModuleList([_FusionLayer(config) for _ in range(config.n_layers)])

            # Sector bias: (n_sectors+1) × (n_sectors+1), index n_sectors = no-sector.
            # Init near zero so the model starts from attention without sector prior.
            n = config.n_sectors + 1  # 12 × 12
            self.sector_bias = nn.Parameter(torch.zeros(n, n))

        def _build_sector_bias(self, sector_ids: Tensor, n_tokens: int) -> Tensor:
            """
            sector_ids: [B, N_assets] — asset token sector labels (0..n_sectors-1)
            Returns [B, n_tokens, n_tokens] bias (0 outside asset×asset block).
            """
            B, N = sector_ids.shape
            # All tokens default to no-sector (index 11); asset tokens get their label
            all_ids = sector_ids.new_full((B, n_tokens), _NO_SECTOR)
            all_ids[:, :N] = sector_ids

            i_ids = all_ids.unsqueeze(2).expand(-1, -1, n_tokens)  # [B, T, T]
            j_ids = all_ids.unsqueeze(1).expand(-1, n_tokens, -1)
            return self.sector_bias[i_ids, j_ids]                  # [B, T, T]

        def forward(
            self,
            asset_tokens: Tensor,           # [B, N_assets, D]
            context_tokens: Tensor | None,  # [B, N_ctx, D] or None
            sector_ids: Tensor | None,      # [B, N_assets] int or None
        ) -> Tensor:
            """Returns [B, N_assets, D] refined asset representations."""
            if context_tokens is not None:
                x = torch.cat([asset_tokens, context_tokens], dim=1)
            else:
                x = asset_tokens

            N_assets = asset_tokens.shape[1]
            n_tokens = x.shape[1]

            sector_bias = (
                self._build_sector_bias(sector_ids, n_tokens)
                if sector_ids is not None
                else None
            )

            for layer in self.layers:
                x = layer(x, sector_bias)

            # Return only asset token outputs
            return x[:, :N_assets, :]

    def run_fusion_smoke_check(
        n_assets: int = 10,
        n_ctx: int = 3,
        batch_size: int = 2,
        config: FusionConfig = FusionConfig(),
        device: str | None = None,
    ) -> dict[str, object]:
        _ensure_torch()
        assert torch is not None

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CrossAssetFusion(config).to(dev)
        D = config.model_dim

        asset_tokens   = torch.randn(batch_size, n_assets, D, device=dev)
        context_tokens = torch.randn(batch_size, n_ctx, D, device=dev)
        sector_ids     = torch.randint(0, config.n_sectors, (batch_size, n_assets), device=dev)

        out = model(asset_tokens, context_tokens, sector_ids)

        return {
            "device": dev,
            "input_asset_shape":   list(asset_tokens.shape),
            "input_context_shape": list(context_tokens.shape),
            "output_shape":        list(out.shape),
            "n_layers":            config.n_layers,
            "n_heads":             config.n_heads,
        }

else:  # pragma: no cover

    class CrossAssetFusion:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ensure_torch()

    def run_fusion_smoke_check(*args: Any, **kwargs: Any) -> dict[str, object]:  # type: ignore[misc]
        _ensure_torch()
        return {}
