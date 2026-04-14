"""Regime-Aware Gated DeltaNet (RA-GDN) backbone — Sections 7.1–7.4."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ── Primitives ───────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ── Variable Selection Network (Section 7.1) ────────────────────────────────


class VSN(nn.Module):
    """Regime-conditioned per-feature gating over the 184-dim input vector.

    gates = softmax(Linear(regime_emb → in_features))
    x_selected = gates ⊙ x
    """

    def __init__(self, in_features: int = 184, regime_emb_dim: int = 64) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(regime_emb_dim, in_features)

    def forward(self, x: torch.Tensor, regime_emb: torch.Tensor) -> torch.Tensor:
        # x:          (B, in_features)
        # regime_emb: (B, regime_emb_dim)
        gates = F.softmax(self.gate_proj(regime_emb), dim=-1)  # (B, in_features)
        return gates * x


# ── Regime Detector (Section 7.2) ────────────────────────────────────────────


class RegimeDetector(nn.Module):
    """MLP that maps market context → regime embedding + posterior.

    Input (163 dims):
      market_wide (96) + portfolio_summary (3) + rolling_regime_ema (64)

    Output:
      regime_emb (64) — used to condition RA-GDN gates and VSN
      posteriors (4)  — used to route FMoE and update HMM prior
      velocity (1)    — ||regime_t − regime_{t-1}||_2
    """

    def __init__(
        self,
        input_dim: int = 163,
        hidden_dim: int = 256,
        mid_dim: int = 128,
        regime_emb_dim: int = 64,
        n_regimes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.emb_head = nn.Linear(mid_dim, regime_emb_dim)
        self.post_head = nn.Linear(mid_dim, n_regimes)

    def forward(
        self,
        market: torch.Tensor,  # (B, 96)
        portfolio_summary: torch.Tensor,  # (B, 3)
        regime_ema: torch.Tensor,  # (B, 64)
        prev_emb: torch.Tensor | None = None,  # (B, 64) — for velocity
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([market, portfolio_summary, regime_ema], dim=-1)  # (B, 163)
        h = self.net(x)
        regime_emb = self.emb_head(h)  # (B, 64)
        posteriors = F.softmax(self.post_head(h), dim=-1)  # (B, 4)
        if prev_emb is not None:
            velocity = torch.linalg.norm(regime_emb - prev_emb, dim=-1, keepdim=True)
        else:
            velocity = torch.zeros(regime_emb.shape[0], 1, device=regime_emb.device)
        return regime_emb, posteriors, velocity


# ── RA-GDN Layer (Section 7.3) ───────────────────────────────────────────────


class RAGDNLayer(nn.Module):
    """Single RA-GDN block with regime-conditioned forgetting/update gates.

    Recurrence (per head):
      S_t = α_t · S_{t-1} + β_t · (k_t ⊗ (v_t − α_t · S_{t-1}ᵀ k_t))
      h_t = S_t · q_t

    α_t = sigmoid(W_α · [h_{t-1}, regime_emb])  — forget gate
    β_t = sigmoid(W_β · [h_{t-1}, regime_emb])  — update strength

    State S: (B, n_heads, d_head, d_v) — separate per asset at inference.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 4,
        d_head: int = 64,
        d_v: int = 192,
        regime_emb_dim: int = 64,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_v = d_v
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

        # Regime-conditioned gates: input = [prev_output (D) + regime_emb (64)]
        gate_in = d_model + regime_emb_dim
        self.alpha_proj = nn.Linear(gate_in, n_heads)
        self.beta_proj = nn.Linear(gate_in, n_heads)

        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Optional hook for Tier 1 surprise-gated adaptation (Section 8.1).
        # Signature: (surprise: Tensor[B], beta: Tensor[B,H,1,1]) -> Tensor[B,H,1,1]
        self.surprise_hook: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None

    def init_state(
        self, batch_size: int, n_assets: int, device: torch.device
    ) -> torch.Tensor:
        """Zero initial state (B, N, n_heads, d_head, d_v) — one state per asset.

        Uses bf16 on CUDA to match autocast dtype, halving state memory and
        avoiding float32 upcasting in the large (B,N,H,Dh,Dv) einsum ops.
        """
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        return torch.zeros(
            batch_size,
            n_assets,
            self.n_heads,
            self.d_head,
            self.d_v,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        state: torch.Tensor,  # (B, N, n_heads, d_head, d_v)
        regime_emb: torch.Tensor,  # (B, regime_emb_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fully vectorised over N assets — no serial loop.

        Each asset maintains its own independent GDN state across portfolio
        timesteps. Gates are conditioned on the current asset token + regime
        embedding (replacing the old prev_out cross-asset dependency, which
        imposed an arbitrary ordering on assets). Cross-asset information flows
        through CrossAssetFusion, not through the GDN state.

        Weight shapes are unchanged from the serial version so pretrained
        checkpoints load without modification.
        """
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        H, Dh, Dv = self.n_heads, self.d_head, self.d_v

        q = self.q_proj(x).view(B, N, H, Dh)
        k = F.normalize(self.k_proj(x).view(B, N, H, Dh), dim=-1)
        v = self.v_proj(x).view(B, N, H, Dv)

        # Gates: condition on current token + regime (same weight shapes as before)
        regime_exp = regime_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, re)
        gate_input = torch.cat([x, regime_exp], dim=-1)  # (B, N, D+re)
        alpha = torch.sigmoid(self.alpha_proj(gate_input)).view(B, N, H, 1, 1)
        beta = torch.sigmoid(self.beta_proj(gate_input)).view(B, N, H, 1, 1)

        compute_dtype = k.dtype
        if state.dtype != compute_dtype:
            state = state.to(compute_dtype)
        if alpha.dtype != compute_dtype:
            alpha = alpha.to(compute_dtype)
        if beta.dtype != compute_dtype:
            beta = beta.to(compute_dtype)

        # DeltaNet update — vectorised over N
        # state: (B, N, H, Dh, Dv),  k: (B, N, H, Dh)
        s_k = torch.einsum("bnhdc,bnhd->bnhc", state, k)  # (B, N, H, Dv)
        correction = v - alpha.squeeze(-1) * s_k  # (B, N, H, Dv)

        if self.surprise_hook is not None:
            surprise_t = correction.detach().pow(2).sum(dim=(-2, -1))  # (B, N)
            beta = self.surprise_hook(surprise_t, beta)

        outer = torch.einsum("bnhd,bnhc->bnhdc", k, correction)  # (B, N, H, Dh, Dv)
        new_state = alpha * state + beta * outer  # (B, N, H, Dh, Dv)

        h = torch.einsum("bnhdc,bnhd->bnhc", new_state, q)  # (B, N, H, Dv)
        h = h.reshape(B, N, H * Dv)
        out = self.drop(self.o_proj(h))  # (B, N, D)
        return residual + out, new_state


# ── Sparse Sliding Window Attention (Section 7.3.2) ─────────────────────────


class SlidingWindowAttention(nn.Module):
    """Per-asset temporal SWA with fixed-size cache.

    The backbone is stepped one market timestep at a time. For each asset token
    x_t, SWA attends over a fixed cache of the previous W token states plus x_t.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        window: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert window > 0
        assert d_model % n_heads == 0
        self.window = window
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def init_state(
        self, batch_size: int, n_assets: int, device: torch.device
    ) -> torch.Tensor:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        return torch.zeros(
            batch_size, n_assets, self.window, self.d_model, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        cache: torch.Tensor,  # (B, N, W, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        B, N, D = x.shape

        seq = torch.cat([cache, x.unsqueeze(2)], dim=2)  # (B, N, W+1, D)
        seq_flat = seq.reshape(B * N, self.window + 1, D)
        q = x.reshape(B * N, 1, D)

        # Early timesteps have zero-padded cache entries. Mask these so attention
        # uses only populated history + current token.
        key_padding_mask = seq_flat.abs().sum(dim=-1).eq(0.0)
        out, _ = self.attn(
            q,
            seq_flat,
            seq_flat,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = out.reshape(B, N, D)
        new_cache = torch.cat([cache[:, :, 1:, :], x.unsqueeze(2)], dim=2)
        return residual + self.drop(out), new_cache


# ── FMoE Block (Section 7.4) ─────────────────────────────────────────────────


class ExpertFFN(nn.Module):
    """Single FFN expert: Linear(D → expert_ffn_dim) → SiLU → Linear(expert_ffn_dim → D)."""

    def __init__(self, d_model: int = 768, ffn_dim: int = 2304) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class FMoEBlock(nn.Module):
    """Regime-routed Sparse Mixture of Experts (Section 7.4).

    Routing is external (no learned router): top-k regimes by posterior activate
    the corresponding k experts; gate weights = softmax over selected posteriors.
    This directly ties expert selection to the HMM regime rather than learning
    a separate routing policy.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_experts: int = 4,
        top_k: int = 2,
        ffn_dim: int = 2304,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert top_k <= n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, ffn_dim) for _ in range(n_experts)]
        )
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        posteriors: torch.Tensor,  # (B, n_regimes) — from RegimeDetector
    ) -> torch.Tensor:  # (B, T, D)
        residual = x
        x = self.norm(x)

        # Select top-k experts per batch element
        top_vals, top_idx = posteriors.topk(self.top_k, dim=-1)  # (B, k)
        gate_weights = F.softmax(top_vals, dim=-1)  # (B, k)

        B, T, D = x.shape
        n_experts = len(self.experts)
        # Run all experts on all inputs — eliminates conditional branching.
        # Extra compute (4 vs 2 experts) is negligible at B=2, N=100 vs. kernel launch savings.
        all_expert_out = torch.stack(
            [expert(x) for expert in self.experts], dim=0
        )  # (n_experts, B, T, D)
        # Scatter gate weights into per-expert weight tensor
        expert_weights = torch.zeros(B, n_experts, device=x.device, dtype=x.dtype)
        expert_weights.scatter_(1, top_idx, gate_weights)  # (B, n_experts)
        # Weighted sum: (n_experts, B, 1, 1) * (n_experts, B, T, D)
        w = (
            expert_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        )  # (n_experts, B, 1, 1)
        out = (all_expert_out * w).sum(dim=0)  # (B, T, D)

        return residual + self.drop(out)


# ── Input Projection (Section 7.1) ──────────────────────────────────────────


class InputProjection(nn.Module):
    """Projects all input modalities to D-dimensional token space.

    Per-asset:  VSN → Linear(184 → D) → RMSNorm
    Market:     Linear(96 → D) → RMSNorm
    Portfolio:  Linear(201 → D) → RMSNorm
    Calendar:   Linear(20 → D)

    token_i = asset_proj_i + market_proj + portfolio_proj + calendar_proj
    """

    def __init__(
        self,
        d_model: int = 768,
        per_asset_dim: int = 184,
        market_dim: int = 96,
        portfolio_dim: int = 201,
        calendar_dim: int = 20,
        regime_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vsn = VSN(per_asset_dim, regime_emb_dim)
        self.asset_proj = nn.Linear(per_asset_dim, d_model)
        self.market_proj = nn.Linear(market_dim, d_model)
        self.portfolio_proj = nn.Linear(portfolio_dim, d_model)
        self.calendar_proj = nn.Linear(calendar_dim, d_model)
        self.asset_norm = RMSNorm(d_model)
        self.market_norm = RMSNorm(d_model)
        self.portfolio_norm = RMSNorm(d_model)

    def forward(
        self,
        asset_features: torch.Tensor,  # (B, N_assets, 184)
        market_features: torch.Tensor,  # (B, 96)
        portfolio_state: torch.Tensor,  # (B, 201)
        calendar: torch.Tensor,  # (B, 20)
        regime_emb: torch.Tensor,  # (B, 64)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = asset_features.shape

        # VSN: apply per-asset with same regime embedding
        regime_exp = regime_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, 64)
        asset_gated = self.vsn(
            asset_features.reshape(B * N, -1),
            regime_exp.reshape(B * N, -1),
        ).reshape(B, N, -1)  # (B, N, 184)

        asset_tok = self.asset_norm(self.asset_proj(asset_gated))  # (B, N, D)
        market_tok = self.market_norm(self.market_proj(market_features))  # (B, D)
        portfolio_tok = self.portfolio_norm(
            self.portfolio_proj(portfolio_state)
        )  # (B, D)
        calendar_tok = self.calendar_proj(calendar)  # (B, D)

        # Compose per-asset token: Section 7.1
        shared = market_tok + portfolio_tok + calendar_tok  # (B, D)
        combined = asset_tok + shared.unsqueeze(1)  # (B, N, D)
        return combined, market_tok, portfolio_tok, calendar_tok


# ── Full RA-GDN Backbone ─────────────────────────────────────────────────────


class RAGDNBackbone(nn.Module):
    """Hybrid backbone: RA-GDN + interleaved SWA + regime-routed FMoE."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 4,
        d_head: int = 64,
        d_v: int = 192,
        n_gdn_blocks: int = 8,
        n_experts: int = 4,
        top_k_experts: int = 2,
        expert_ffn_dim: int = 2304,
        fmoe_after_gdn: tuple[int, ...] = (2, 5, 8),
        swa_after_gdn: tuple[int, ...] = (3, 7),
        swa_window: int = 64,
        swa_heads: int = 8,
        regime_emb_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        gdn_positions = tuple(range(1, n_gdn_blocks + 1))
        assert all(p in gdn_positions for p in fmoe_after_gdn)
        assert all(p in gdn_positions for p in swa_after_gdn)

        self.gdn_blocks = nn.ModuleList(
            [
                RAGDNLayer(d_model, n_heads, d_head, d_v, regime_emb_dim, dropout)
                for _ in range(n_gdn_blocks)
            ]
        )
        self.fmoe_blocks = nn.ModuleList(
            [
                FMoEBlock(d_model, n_experts, top_k_experts, expert_ffn_dim, dropout)
                for _ in fmoe_after_gdn
            ]
        )
        self.swa_blocks = nn.ModuleList(
            [
                SlidingWindowAttention(
                    d_model=d_model,
                    n_heads=swa_heads,
                    window=swa_window,
                    dropout=dropout,
                )
                for _ in swa_after_gdn
            ]
        )

        block_plan: list[str] = []
        for gdn_pos in gdn_positions:
            block_plan.append("gdn")
            if gdn_pos in fmoe_after_gdn:
                block_plan.append("fmoe")
            if gdn_pos in swa_after_gdn:
                block_plan.append("swa")
        self.block_plan = tuple(block_plan)

        self._n_gdn_states = len(self.gdn_blocks)
        self._n_swa_states = len(self.swa_blocks)
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_v = d_v
        self.swa_window = swa_window
        self.d_model = d_model

        self._runtime_instrumentation = False
        self._last_runtime_metrics: dict[str, float] = {}
        self._swa_cache_budget_bytes: int | None = None
        self._max_forward_ms: float | None = None

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing on forward() to reduce VRAM.

        When enabled, intermediate activations within each backbone call are
        discarded during the forward pass and recomputed during backprop.
        Reduces peak VRAM from O(T*N) to O(N) at ~33% extra compute cost.
        """
        self._gradient_checkpointing = True

    def enable_runtime_instrumentation(self, enabled: bool = True) -> None:
        self._runtime_instrumentation = enabled

    def set_runtime_guardrails(
        self,
        *,
        swa_cache_budget_mb: float | None = None,
        max_forward_ms: float | None = None,
    ) -> None:
        self._swa_cache_budget_bytes = (
            int(swa_cache_budget_mb * 1024 * 1024)
            if swa_cache_budget_mb is not None
            else None
        )
        self._max_forward_ms = max_forward_ms

    def estimate_swa_cache_bytes(
        self,
        batch_size: int,
        n_assets: int,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        bytes_per_elem = torch.empty((), dtype=dtype).element_size()
        return (
            batch_size
            * n_assets
            * self._n_swa_states
            * self.swa_window
            * self.d_model
            * bytes_per_elem
        )

    def last_runtime_metrics(self) -> dict[str, float]:
        return dict(self._last_runtime_metrics)

    def _checkpointed_forward(
        self,
        x: torch.Tensor,
        regime_emb: torch.Tensor,
        posteriors: torch.Tensor,
        *flat_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Flat-tensor wrapper for use with torch.utils.checkpoint."""
        states = list(flat_states)
        x_out, new_states = self._forward_impl(x, regime_emb, posteriors, states)
        return (x_out, *new_states)

    def _forward_impl(
        self,
        x: torch.Tensor,
        regime_emb: torch.Tensor,
        posteriors: torch.Tensor,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert len(states) == self._n_gdn_states + self._n_swa_states
        gdn_states = states[: self._n_gdn_states]
        swa_states = states[self._n_gdn_states :]

        gdn_idx = fmoe_idx = swa_idx = 0
        new_gdn_states: list[torch.Tensor] = []
        new_swa_states: list[torch.Tensor] = []
        gdn_ms = fmoe_ms = swa_ms = 0.0
        total_t0 = time.perf_counter()

        for block_type in self.block_plan:
            t0 = time.perf_counter()
            if block_type == "gdn":
                x, new_s = self.gdn_blocks[gdn_idx](x, gdn_states[gdn_idx], regime_emb)
                new_gdn_states.append(new_s)
                gdn_idx += 1
                gdn_ms += (time.perf_counter() - t0) * 1e3
            elif block_type == "fmoe":
                x = self.fmoe_blocks[fmoe_idx](x, posteriors)
                fmoe_idx += 1
                fmoe_ms += (time.perf_counter() - t0) * 1e3
            else:  # swa
                x, new_cache = self.swa_blocks[swa_idx](x, swa_states[swa_idx])
                new_swa_states.append(new_cache)
                swa_idx += 1
                swa_ms += (time.perf_counter() - t0) * 1e3

        total_ms = (time.perf_counter() - total_t0) * 1e3
        if self._runtime_instrumentation:
            self._last_runtime_metrics = {
                "gdn_ms": gdn_ms,
                "fmoe_ms": fmoe_ms,
                "swa_ms": swa_ms,
                "total_ms": total_ms,
                "n_gdn_blocks": float(self._n_gdn_states),
                "n_swa_blocks": float(self._n_swa_states),
            }
        if self._max_forward_ms is not None and total_ms > self._max_forward_ms:
            raise RuntimeError(
                f"backbone forward time {total_ms:.3f} ms exceeded guardrail {self._max_forward_ms:.3f} ms"
            )
        return x, new_gdn_states + new_swa_states

    def init_state(
        self, batch_size: int, n_assets: int, device: torch.device
    ) -> list[torch.Tensor]:
        """Create zero states for GDN and SWA blocks.

        State layout:
          [GDN block 0..n_gdn-1 states, SWA block 0..n_swa-1 caches]
        """
        if self._swa_cache_budget_bytes is not None and self._n_swa_states > 0:
            cache_bytes = self.estimate_swa_cache_bytes(batch_size, n_assets)
            if cache_bytes > self._swa_cache_budget_bytes:
                max_mb = self._swa_cache_budget_bytes / (1024 * 1024)
                got_mb = cache_bytes / (1024 * 1024)
                raise ValueError(
                    f"SWA cache budget exceeded: {got_mb:.2f} MB > {max_mb:.2f} MB"
                )
        gdn_states = [
            b.init_state(batch_size, n_assets, device) for b in self.gdn_blocks
        ]
        swa_states = [
            b.init_state(batch_size, n_assets, device) for b in self.swa_blocks
        ]
        return gdn_states + swa_states

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        regime_emb: torch.Tensor,  # (B, regime_emb_dim)
        posteriors: torch.Tensor,  # (B, n_regimes)
        states: list[torch.Tensor],  # len = n_gdn_blocks + n_swa_layers
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training
        if use_ckpt:
            out = grad_checkpoint(
                self._checkpointed_forward,
                x,
                regime_emb,
                posteriors,
                *states,
                use_reentrant=False,
            )
            x_out = out[0]
            new_states = list(out[1:])
            return x_out, new_states
        return self._forward_impl(x, regime_emb, posteriors, states)
