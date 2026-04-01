"""Regime-Aware Gated DeltaNet (RA-GDN) backbone — Sections 7.1–7.4."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.emb_head  = nn.Linear(mid_dim, regime_emb_dim)
        self.post_head = nn.Linear(mid_dim, n_regimes)

    def forward(
        self,
        market: torch.Tensor,       # (B, 96)
        portfolio_summary: torch.Tensor,  # (B, 3)
        regime_ema: torch.Tensor,   # (B, 64)
        prev_emb: torch.Tensor | None = None,  # (B, 64) — for velocity
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([market, portfolio_summary, regime_ema], dim=-1)  # (B, 163)
        h = self.net(x)
        regime_emb  = self.emb_head(h)                           # (B, 64)
        posteriors  = F.softmax(self.post_head(h), dim=-1)       # (B, 4)
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
    ) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_head
        self.d_v      = d_v
        self.d_model  = d_model

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_v,   bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model,   bias=False)

        # Regime-conditioned gates: input = [prev_output (D) + regime_emb (64)]
        gate_in = d_model + regime_emb_dim
        self.alpha_proj = nn.Linear(gate_in, n_heads)
        self.beta_proj  = nn.Linear(gate_in, n_heads)

        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero initial state (B, n_heads, d_head, d_v)."""
        return torch.zeros(batch_size, self.n_heads, self.d_head, self.d_v, device=device)

    def _step(
        self,
        x: torch.Tensor,          # (B, D)
        state: torch.Tensor,       # (B, n_heads, d_head, d_v)
        regime_emb: torch.Tensor,  # (B, regime_emb_dim)
        prev_out: torch.Tensor,    # (B, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        H, Dh, Dv = self.n_heads, self.d_head, self.d_v

        q = self.q_proj(x).view(B, H, Dh)  # (B, H, Dh)
        k = self.k_proj(x).view(B, H, Dh)  # (B, H, Dh)
        v = self.v_proj(x).view(B, H, Dv)  # (B, H, Dv)

        # DeltaNet requires unit-norm keys to prevent state norm explosion.
        # Without this, ||S_t|| grows as O(t) and the recurrence diverges.
        k = F.normalize(k, dim=-1)

        gate_input = torch.cat([prev_out, regime_emb], dim=-1)  # (B, D+64)
        # α, β: (B, H, 1, 1) for broadcasting against state
        alpha = torch.sigmoid(self.alpha_proj(gate_input)).view(B, H, 1, 1)
        beta  = torch.sigmoid(self.beta_proj(gate_input)).view(B, H, 1, 1)

        # Retrieve: h = S · q → (B, H, Dv)
        # state: (B, H, Dh, Dv), q: (B, H, Dh)
        s_q = torch.einsum("bhdc,bhd->bhc", state, k)  # (B, H, Dv) — correction term

        # Update state: S_new = α·S + β · k ⊗ (v − α·S^T k)
        correction = v - alpha.squeeze(-1) * s_q        # (B, H, Dv)
        outer = torch.einsum("bhd,bhc->bhdc", k, correction)  # (B, H, Dh, Dv)
        new_state = alpha * state + beta * outer         # (B, H, Dh, Dv)

        # Output: h = new_S · q
        h = torch.einsum("bhdc,bhd->bhc", new_state, q)  # (B, H, Dv)
        h = h.reshape(B, H * Dv)                          # (B, H*Dv)
        out = self.drop(self.o_proj(h))                    # (B, D)
        return out, new_state

    def forward(
        self,
        x: torch.Tensor,          # (B, T, D)
        state: torch.Tensor,       # (B, n_heads, d_head, d_v)
        regime_emb: torch.Tensor,  # (B, regime_emb_dim)  — constant over sequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        device = x.device

        prev_out = torch.zeros(B, self.d_model, device=device)
        outputs = []
        for t in range(T):
            out_t, state = self._step(x[:, t], state, regime_emb, prev_out)
            prev_out = out_t
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)  # (B, T, D)
        return residual + out, state


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
        self.experts = nn.ModuleList([ExpertFFN(d_model, ffn_dim) for _ in range(n_experts)])
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, D)
        posteriors: torch.Tensor,  # (B, n_regimes) — from RegimeDetector
    ) -> torch.Tensor:             # (B, T, D)
        residual = x
        x = self.norm(x)

        # Select top-k experts per batch element
        top_vals, top_idx = posteriors.topk(self.top_k, dim=-1)  # (B, k)
        gate_weights = F.softmax(top_vals, dim=-1)               # (B, k)

        B, T, D = x.shape
        out = torch.zeros_like(x)
        for ki in range(self.top_k):
            expert_ids = top_idx[:, ki]  # (B,) — which expert is active for each item
            weights    = gate_weights[:, ki].view(B, 1, 1)  # (B, 1, 1)
            # Route each batch element to its assigned expert
            # Group by expert_id to minimise expert forward passes
            for eid, expert in enumerate(self.experts):
                mask = (expert_ids == eid)  # (B,)
                if not mask.any():
                    continue
                expert_out = expert(x[mask])   # (n_matched, T, D)
                out[mask] += weights[mask] * expert_out

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
        self.vsn          = VSN(per_asset_dim, regime_emb_dim)
        self.asset_proj   = nn.Linear(per_asset_dim, d_model)
        self.market_proj  = nn.Linear(market_dim, d_model)
        self.portfolio_proj = nn.Linear(portfolio_dim, d_model)
        self.calendar_proj = nn.Linear(calendar_dim, d_model)
        self.asset_norm    = RMSNorm(d_model)
        self.market_norm   = RMSNorm(d_model)
        self.portfolio_norm = RMSNorm(d_model)

    def forward(
        self,
        asset_features: torch.Tensor,   # (B, N_assets, 184)
        market_features: torch.Tensor,  # (B, 96)
        portfolio_state: torch.Tensor,  # (B, 201)
        calendar: torch.Tensor,         # (B, 20)
        regime_emb: torch.Tensor,       # (B, 64)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = asset_features.shape

        # VSN: apply per-asset with same regime embedding
        regime_exp = regime_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, 64)
        asset_gated = self.vsn(
            asset_features.view(B * N, -1),
            regime_exp.contiguous().view(B * N, -1),
        ).view(B, N, -1)                                         # (B, N, 184)

        asset_tok     = self.asset_norm(self.asset_proj(asset_gated))          # (B, N, D)
        market_tok    = self.market_norm(self.market_proj(market_features))    # (B, D)
        portfolio_tok = self.portfolio_norm(self.portfolio_proj(portfolio_state))  # (B, D)
        calendar_tok  = self.calendar_proj(calendar)                           # (B, D)

        # Compose per-asset token: Section 7.1
        shared = market_tok + portfolio_tok + calendar_tok  # (B, D)
        combined = asset_tok + shared.unsqueeze(1)          # (B, N, D)
        return combined, market_tok, portfolio_tok, calendar_tok


# ── Full RA-GDN Backbone ─────────────────────────────────────────────────────

class RAGDNBackbone(nn.Module):
    """Complete backbone: 8 RA-GDN blocks + 3 FMoE blocks (Sections 7.1–7.4).

    Block layout (1-indexed GDN positions where FMoE is inserted after):
      GDN 1 → GDN 2 → FMoE → GDN 3 → GDN 4 → GDN 5 → FMoE
      → GDN 6 → GDN 7 → GDN 8 → FMoE

    State is maintained externally to support separate per-asset states at
    inference time and reset semantics during training.
    """

    # Sequence of block types: True = FMoE, False = GDN
    _BLOCK_IS_FMOE = (False, False, True, False, False, False, True, False, False, False, True)

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
        regime_emb_dim: int = 64,
        n_regimes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert sum(1 for f in self._BLOCK_IS_FMOE if not f) == n_gdn_blocks

        gdn_blocks: list[nn.Module] = []
        fmoe_blocks: list[nn.Module] = []
        for is_fmoe in self._BLOCK_IS_FMOE:
            if is_fmoe:
                fmoe_blocks.append(
                    FMoEBlock(d_model, n_experts, top_k_experts, expert_ffn_dim, dropout)
                )
            else:
                gdn_blocks.append(
                    RAGDNLayer(d_model, n_heads, d_head, d_v, regime_emb_dim, dropout)
                )

        self.gdn_blocks  = nn.ModuleList(gdn_blocks)
        self.fmoe_blocks = nn.ModuleList(fmoe_blocks)
        self.n_heads = n_heads
        self.d_head  = d_head
        self.d_v     = d_v

    def init_state(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Create zero GDN states for a new sequence."""
        return [b.init_state(batch_size, device) for b in self.gdn_blocks]

    def forward(
        self,
        x: torch.Tensor,             # (B, T, D)
        regime_emb: torch.Tensor,    # (B, regime_emb_dim)
        posteriors: torch.Tensor,    # (B, n_regimes)
        states: list[torch.Tensor],  # len = n_gdn_blocks, each (B, H, Dh, Dv)
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        gdn_idx = fmoe_idx = 0
        new_states: list[torch.Tensor] = []
        for is_fmoe in self._BLOCK_IS_FMOE:
            if is_fmoe:
                x = self.fmoe_blocks[fmoe_idx](x, posteriors)
                fmoe_idx += 1
            else:
                x, new_s = self.gdn_blocks[gdn_idx](x, states[gdn_idx], regime_emb)
                new_states.append(new_s)
                gdn_idx += 1
        return x, new_states
