"""Temporal cascade: hourly/daily/weekly GDN stacks with surprise-gated injection.

Spec §6.7. Three GDN blocks aggregate per-asset minute representations at
progressively slower timescales. Their outputs are:
  - projected back as injection contexts for the minute backbone (downward pass)
  - concatenated with minute output and fused into a single representation

Training simplification: all GDNs run at every step. Cadence (60/390/5×390 min)
is respected at inference time by the runtime; during training the GDNs learn the
appropriate temporal scales through their recurrent state dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .regime_aware_backbone import BackboneCoreConfig, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch import Tensor, nn

    from .regime_aware_backbone import RAGDNBlock, RMSNorm

else:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "TemporalCascade requires torch. Install with: pip install -e '.[train]'"
        )


@dataclass(frozen=True)
class CascadeConfig:
    minute_dim: int = 768
    hourly_dim: int = 512   # 4 heads × d_v=128
    daily_dim: int = 512
    weekly_dim: int = 384   # 4 heads × d_v=96, d_head=48 (spec §6.7 state budget)
    regime_dim: int = 64
    conv_kernel: int = 4
    # Backbone block indices after which each context is injected (spec §6.7 table)
    hourly_injection_block: int = 1
    daily_injection_block: int = 4
    weekly_injection_block: int = 7


def _block_config(model_dim: int, regime_dim: int, d_head: int, conv_kernel: int) -> BackboneCoreConfig:
    """BackboneCoreConfig for a single cascade GDN (n_heads=4, d_v=model_dim//4)."""
    return BackboneCoreConfig(
        model_dim=model_dim,
        regime_input_dim=regime_dim,
        regime_dim=regime_dim,
        n_heads=4,
        d_head=d_head,
        d_v=model_dim // 4,
        conv_kernel=conv_kernel,
    )


if TORCH_AVAILABLE:

    class TemporalCascade(nn.Module):
        """Multi-timescale context cascade for Meridian backbone.

        Architecture (per forward call):
          minute_repr → up_hourly → hourly_gdn → h_hourly
          h_hourly    → up_daily  → daily_gdn  → h_daily
          h_daily     → up_weekly → weekly_gdn → h_weekly

          downward: gate_i * down_i(h_i) → backbone injection at block_i
          fusion:   Linear(cat([minute; hourly; daily; weekly])) → fused_repr
        """

        def __init__(self, config: CascadeConfig = CascadeConfig()) -> None:
            super().__init__()
            self.config = config
            c = config

            # Upward projections into cascade dims
            self.up_hourly = nn.Linear(c.minute_dim, c.hourly_dim)
            self.up_daily  = nn.Linear(c.hourly_dim, c.daily_dim)
            self.up_weekly = nn.Linear(c.daily_dim,  c.weekly_dim)

            # Cascade GDN blocks (one per level, spec §6.7)
            self.hourly_gdn = RAGDNBlock(_block_config(c.hourly_dim, c.regime_dim, 64, c.conv_kernel))
            self.daily_gdn  = RAGDNBlock(_block_config(c.daily_dim,  c.regime_dim, 64, c.conv_kernel))
            self.weekly_gdn = RAGDNBlock(_block_config(c.weekly_dim, c.regime_dim, 48, c.conv_kernel))

            # Downward projections to minute_dim for backbone injection
            self.down_hourly = nn.Linear(c.hourly_dim, c.minute_dim)
            self.down_daily  = nn.Linear(c.daily_dim,  c.minute_dim)
            self.down_weekly = nn.Linear(c.weekly_dim, c.minute_dim)

            # Surprise gate params: gate = sigmoid(w[i]*surprise + b[i]), packed as [h,d,w]
            self.gate_w = nn.Parameter(torch.ones(3))
            self.gate_b = nn.Parameter(torch.zeros(3))

            # Temporal fusion: Linear(cat([min; hourly; daily; weekly]) → minute_dim)
            fused_in = c.minute_dim + c.hourly_dim + c.daily_dim + c.weekly_dim
            self.fusion      = nn.Linear(fused_in, c.minute_dim)
            self.fusion_norm = RMSNorm(c.minute_dim)

        def _gate(self, level: int, surprise: Tensor | None, B: int, T: int) -> Tensor:
            """Surprise-amplification gate broadcastable to [B, T, 1]."""
            if surprise is not None:
                return torch.sigmoid(self.gate_w[level] * surprise.unsqueeze(-1) + self.gate_b[level])
            return torch.sigmoid(self.gate_b[level]).view(1, 1, 1).expand(B, T, 1)

        def forward(
            self,
            minute_repr: Tensor,       # [B, T, minute_dim]
            regime_embedding: Tensor,  # [B, T, regime_dim]
            surprise_ema: Tensor | None = None,  # [B, T]
        ) -> tuple[dict[int, Tensor], Tensor]:
            """
            Returns:
              injection_contexts: {block_idx: [B, T, minute_dim]}  — add to backbone output
              fused_repr:         [B, T, minute_dim]  — input for cross-asset fusion
            """
            c = self.config
            B, T, _ = minute_repr.shape

            # Upward: each level feeds the next
            h_hourly, _, _ = self.hourly_gdn(self.up_hourly(minute_repr), regime_embedding)
            h_daily,  _, _ = self.daily_gdn( self.up_daily(h_hourly),     regime_embedding)
            h_weekly, _, _ = self.weekly_gdn(self.up_weekly(h_daily),     regime_embedding)

            # Downward: surprise-gated contexts for backbone injection points
            ctx = {
                c.hourly_injection_block: self._gate(0, surprise_ema, B, T) * self.down_hourly(h_hourly),
                c.daily_injection_block:  self._gate(1, surprise_ema, B, T) * self.down_daily(h_daily),
                c.weekly_injection_block: self._gate(2, surprise_ema, B, T) * self.down_weekly(h_weekly),
            }

            fused = self.fusion_norm(
                self.fusion(torch.cat([minute_repr, h_hourly, h_daily, h_weekly], dim=-1))
            )

            return ctx, fused

    def run_cascade_smoke_check(
        batch_size: int = 2,
        seq_len: int = 16,
        config: CascadeConfig = CascadeConfig(),
        device: str | None = None,
    ) -> dict[str, object]:
        _ensure_torch()
        assert torch is not None

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalCascade(config).to(dev)
        c = config

        ctx, fused = model(
            torch.randn(batch_size, seq_len, c.minute_dim, device=dev),
            torch.randn(batch_size, seq_len, c.regime_dim, device=dev),
            surprise_ema=torch.rand(batch_size, seq_len, device=dev),
        )

        return {
            "device": dev,
            "injection_blocks": sorted(ctx.keys()),
            "hourly_ctx_shape":  list(ctx[c.hourly_injection_block].shape),
            "daily_ctx_shape":   list(ctx[c.daily_injection_block].shape),
            "weekly_ctx_shape":  list(ctx[c.weekly_injection_block].shape),
            "fused_shape":       list(fused.shape),
        }

else:  # pragma: no cover

    class TemporalCascade:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ensure_torch()

    def run_cascade_smoke_check(*args: Any, **kwargs: Any) -> dict[str, object]:  # type: ignore[misc]
        _ensure_torch()
        return {}
