"""Temporal Cascade — multi-resolution GDN stack with surprise injection (Section 7.5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regime_aware_backbone import RMSNorm


class CascadeGDN(nn.Module):
    """Lightweight single-head GDN for a cascade level (no regime conditioning).

    Standard GDN recurrence:
      S_t = α · S_{t-1} + β · (k_t ⊗ (v_t − S_{t-1}ᵀ k_t))
      h_t = S_t · q_t

    α and β are learned scalar parameters shared across all steps.
    """

    def __init__(self, dim: int, d_head: int = 64) -> None:
        super().__init__()
        self.dim    = dim
        self.d_head = d_head

        self.q_proj = nn.Linear(dim, d_head, bias=False)
        self.k_proj = nn.Linear(dim, d_head, bias=False)
        self.v_proj = nn.Linear(dim, dim,    bias=False)
        self.o_proj = nn.Linear(dim, dim,    bias=False)

        # Learned forget / update scalars (not regime-conditioned at this level)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_beta  = nn.Parameter(torch.zeros(1))

        self.norm = RMSNorm(dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_head, self.dim, device=device)

    def step(
        self,
        x: torch.Tensor,    # (B, dim)
        state: torch.Tensor,  # (B, d_head, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.log_alpha)
        beta  = torch.sigmoid(self.log_beta)

        q = self.q_proj(x)  # (B, d_head)
        k = self.k_proj(x)  # (B, d_head)
        v = self.v_proj(x)  # (B, dim)

        s_k = torch.einsum("bdc,bd->bc", state, k)          # (B, dim)
        correction = v - alpha * s_k                          # (B, dim)
        outer = torch.einsum("bd,bc->bdc", k, correction)    # (B, d_head, dim)
        new_state = alpha * state + beta * outer              # (B, d_head, dim)

        h = torch.einsum("bdc,bd->bc", new_state, q)         # (B, dim)
        return self.o_proj(self.norm(h)), new_state


class CascadeLevel(nn.Module):
    """One temporal cascade level with downward injection from a slower level.

    Receives a compressed representation of the faster (minute-scale) stream
    every `stride` steps, and injects its own context back into the faster
    stream gated by the surprise signal.
    """

    def __init__(
        self,
        in_dim: int,   # dimension of the faster stream (D=768 for level 1)
        dim: int,      # this level's working dimension
        stride: int,   # how many minute steps between updates
    ) -> None:
        super().__init__()
        self.dim    = dim
        self.stride = stride

        # Downsample the faster stream input to this level's dim
        self.down_proj  = nn.Linear(in_dim, dim)
        # Upsample this level's state back to the faster stream for injection
        self.up_proj    = nn.Linear(dim, in_dim)

        self.gdn = CascadeGDN(dim)

        # Surprise-gated injection: gate = sigmoid(W_gate · surprise_scalar)
        self.inject_gate = nn.Linear(1, 1)

        self.norm = RMSNorm(in_dim)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gdn_state, level_output)."""
        gdn_state    = self.gdn.init_state(batch_size, device)
        level_output = torch.zeros(batch_size, self.dim, device=device)
        return gdn_state, level_output

    def update(
        self,
        pooled_input: torch.Tensor,  # (B, in_dim) — mean-pooled faster stream over stride steps
        state: torch.Tensor,         # (B, d_head, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance this level by one stride and return new state + output."""
        x = self.down_proj(pooled_input)      # (B, dim)
        output, new_state = self.gdn.step(x, state)
        return new_state, output

    def inject(
        self,
        faster_token: torch.Tensor,  # (B, in_dim)
        level_output: torch.Tensor,  # (B, dim)
        surprise: torch.Tensor,      # (B, 1) scalar surprise signal
    ) -> torch.Tensor:               # (B, in_dim) — updated faster token
        gate = torch.sigmoid(self.inject_gate(surprise))   # (B, 1)
        injection = self.up_proj(level_output)              # (B, in_dim)
        return self.norm(faster_token + gate * injection)


class TemporalCascade(nn.Module):
    """Three-level cascade (hourly/daily/weekly) above the minute-scale backbone.

    Cascade dims (from Appendix C): 512 (hourly), 512 (daily), 384 (weekly).
    Strides: 60 min, 390 min (full trading day), 1950 min (full trading week).

    At each stride boundary the cascade level:
      1. Pools the backbone outputs since the last update
      2. Advances its own GDN state
      3. Injects its context back into the backbone stream (surprise-gated)

    State tensors are maintained externally so they can persist across
    training sequences and be independently managed for each asset at inference.
    """

    def __init__(
        self,
        d_model: int = 768,
        cascade_dims: tuple[int, ...] = (512, 512, 384),
        cascade_strides: tuple[int, ...] = (60, 390, 1950),
    ) -> None:
        super().__init__()
        assert len(cascade_dims) == len(cascade_strides)
        dims    = [d_model] + list(cascade_dims)
        self.levels = nn.ModuleList([
            CascadeLevel(dims[i], cascade_dims[i], cascade_strides[i])
            for i in range(len(cascade_dims))
        ])
        self.strides = cascade_strides

    def init_states(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [level.init_state(batch_size, device) for level in self.levels]

    def forward(
        self,
        backbone_outputs: torch.Tensor,    # (B, T, D) — per-minute backbone output
        surprise_signals: torch.Tensor,    # (B, T, 1) — per-minute surprise scalars
        states: list[tuple[torch.Tensor, torch.Tensor]],
        global_step: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Apply cascade injection to each timestep and return enriched outputs."""
        B, T, D = backbone_outputs.shape
        outputs = backbone_outputs.clone()
        new_states = [list(s) for s in states]  # mutable copies

        # Track pooling buffers for each level
        pool_bufs: list[list[torch.Tensor]] = [[] for _ in self.levels]

        for t in range(T):
            step = global_step + t
            token_t = outputs[:, t]  # (B, D)

            for li, level in enumerate(self.levels):
                pool_bufs[li].append(token_t.detach())
                stride = self.strides[li]

                if (step + 1) % stride == 0 and len(pool_bufs[li]) > 0:
                    # Advance this cascade level
                    pooled = torch.stack(pool_bufs[li], dim=1).mean(dim=1)  # (B, D)
                    gdn_state, level_out = new_states[li]
                    new_gdn_state, new_level_out = level.update(pooled, gdn_state)
                    new_states[li] = [new_gdn_state, new_level_out]
                    pool_bufs[li].clear()

                # Downward injection at every step using current level output
                _, level_out = new_states[li]
                surprise_t = surprise_signals[:, t]  # (B, 1)
                token_t = level.inject(token_t, level_out, surprise_t)

            outputs[:, t] = token_t

        return outputs, [tuple(s) for s in new_states]  # type: ignore[return-value]
