from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in envs without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "Phase 1 core requires torch. Install with: pip install -e '.[train]'"
        )


@dataclass(frozen=True)
class Phase1CoreConfig:
    model_dim: int = 768
    regime_input_dim: int = 91
    regime_dim: int = 64
    n_regimes: int = 4
    n_heads: int = 4
    d_head: int = 64
    d_v: int = 192
    conv_kernel: int = 4


if TORCH_AVAILABLE:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: Tensor) -> Tensor:
            norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            return x / norm * self.weight


    class RegimeDetector(nn.Module):
        def __init__(
            self,
            input_dim: int = 91,
            hidden_dim: int = 128,
            regime_dim: int = 64,
            n_regimes: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.regime_head = nn.Linear(hidden_dim, regime_dim)
            self.posterior_head = nn.Linear(hidden_dim, n_regimes)

        def forward(
            self,
            x: Tensor,
            hmm_posteriors: Tensor | None = None,
        ) -> dict[str, Tensor]:
            h = self.dropout(F.silu(self.fc1(x)))
            h = self.dropout(F.silu(self.fc2(h)))

            regime_embedding = self.regime_head(h)
            neural_posteriors = F.softmax(self.posterior_head(h), dim=-1)

            if hmm_posteriors is not None:
                final_posteriors = 0.7 * neural_posteriors + 0.3 * hmm_posteriors
                final_posteriors = final_posteriors / final_posteriors.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-9)
            else:
                final_posteriors = neural_posteriors

            previous_embedding = torch.cat(
                [regime_embedding[:, :1, :], regime_embedding[:, :-1, :]], dim=1
            )
            velocity = torch.norm(regime_embedding - previous_embedding, dim=-1)

            return {
                "regime_embedding": regime_embedding,
                "neural_posteriors": neural_posteriors,
                "final_posteriors": final_posteriors,
                "regime_change_velocity": velocity,
            }


    class RAGDNBlock(nn.Module):
        def __init__(self, config: Phase1CoreConfig = Phase1CoreConfig()) -> None:
            super().__init__()
            if config.n_heads * config.d_v != config.model_dim:
                raise ValueError("n_heads * d_v must equal model_dim for concat projection")

            self.config = config
            self.token_shift = nn.Parameter(torch.zeros(config.model_dim))
            self.pre_norm = RMSNorm(config.model_dim)
            self.post_norm = RMSNorm(config.model_dim)

            self.q_proj = nn.Linear(config.model_dim, config.n_heads * config.d_head)
            self.k_proj = nn.Linear(config.model_dim, config.n_heads * config.d_head)
            self.v_proj = nn.Linear(config.model_dim, config.n_heads * config.d_v)

            self.q_conv = nn.Conv1d(
                config.d_head,
                config.d_head,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel - 1,
                groups=config.d_head,
            )
            self.k_conv = nn.Conv1d(
                config.d_head,
                config.d_head,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel - 1,
                groups=config.d_head,
            )
            self.v_conv = nn.Conv1d(
                config.d_v,
                config.d_v,
                kernel_size=config.conv_kernel,
                padding=config.conv_kernel - 1,
                groups=config.d_v,
            )

            self.alpha_proj = nn.Linear(config.model_dim, config.n_heads)
            self.alpha_regime = nn.Linear(config.regime_dim, config.n_heads)
            self.beta_proj = nn.Linear(config.model_dim, config.n_heads)
            self.beta_regime = nn.Linear(config.regime_dim, config.n_heads)

            self.output_gate = nn.Linear(config.model_dim, config.model_dim)
            self.out_proj = nn.Linear(config.model_dim, config.model_dim)

        def _apply_temporal_conv(self, x: Tensor, conv: nn.Conv1d, channels: int) -> Tensor:
            batch, seq_len, n_heads, _ = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(batch * n_heads, channels, seq_len)
            y = F.silu(conv(x_flat))
            y = y[:, :, :seq_len]
            return y.reshape(batch, n_heads, channels, seq_len).permute(0, 3, 1, 2)

        def forward(
            self,
            x: Tensor,
            regime_embedding: Tensor,
            state: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
            cfg = self.config
            batch, seq_len, _ = x.shape

            shift = torch.sigmoid(self.token_shift).view(1, 1, -1)
            x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
            x_shifted = (1.0 - shift) * x + shift * x_prev
            x_norm = self.pre_norm(x_shifted)

            q = self.q_proj(x_norm).view(batch, seq_len, cfg.n_heads, cfg.d_head)
            k = self.k_proj(x_norm).view(batch, seq_len, cfg.n_heads, cfg.d_head)
            v = self.v_proj(x_norm).view(batch, seq_len, cfg.n_heads, cfg.d_v)

            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            q = self._apply_temporal_conv(q, self.q_conv, cfg.d_head)
            k = self._apply_temporal_conv(k, self.k_conv, cfg.d_head)
            v = self._apply_temporal_conv(v, self.v_conv, cfg.d_v)

            alpha = torch.sigmoid(self.alpha_proj(x_norm) + self.alpha_regime(regime_embedding))
            beta = torch.sigmoid(self.beta_proj(x_norm)) * torch.sigmoid(
                self.beta_regime(regime_embedding)
            )

            if state is None:
                state = torch.zeros(
                    batch,
                    cfg.n_heads,
                    cfg.d_head,
                    cfg.d_v,
                    dtype=x.dtype,
                    device=x.device,
                )

            outputs: list[Tensor] = []
            s_t = state
            for step in range(seq_len):
                q_t = q[:, step]
                k_t = k[:, step]
                v_t = v[:, step]
                alpha_t = alpha[:, step]
                beta_t = beta[:, step]

                prediction = torch.einsum("bhdv,bhd->bhv", s_t, k_t)
                prediction_error = v_t - alpha_t.unsqueeze(-1) * prediction
                delta = torch.einsum("bhd,bhv->bhdv", k_t, prediction_error)

                s_t = (
                    alpha_t.unsqueeze(-1).unsqueeze(-1) * s_t
                    + beta_t.unsqueeze(-1).unsqueeze(-1) * delta
                )

                o_t = torch.einsum("bhd,bhdv->bhv", q_t, s_t)
                outputs.append(o_t)

            out = torch.stack(outputs, dim=1).reshape(batch, seq_len, cfg.model_dim)
            gate = torch.sigmoid(self.output_gate(x_norm))
            y = self.out_proj(out * gate)
            result = self.post_norm(x + y)
            aux = {"alpha": alpha, "beta": beta}
            return result, s_t, aux


    class RegimeRoutedFMoEBlock(nn.Module):
        def __init__(
            self,
            model_dim: int = 768,
            n_experts: int = 4,
            active_experts: int = 2,
            expansion_ratio: int = 2,
        ) -> None:
            super().__init__()
            if active_experts > n_experts:
                raise ValueError("active_experts cannot exceed n_experts")
            self.n_experts = n_experts
            self.active_experts = active_experts
            hidden_dim = model_dim * expansion_ratio

            self.pre_norm = RMSNorm(model_dim)
            self.post_norm = RMSNorm(model_dim)
            self.expert_up = nn.ModuleList(
                [nn.Linear(model_dim, hidden_dim) for _ in range(n_experts)]
            )
            self.expert_gate = nn.ModuleList(
                [nn.Linear(model_dim, hidden_dim) for _ in range(n_experts)]
            )
            self.expert_down = nn.ModuleList(
                [nn.Linear(hidden_dim, model_dim) for _ in range(n_experts)]
            )

        def _expert_forward(self, x: Tensor, idx: int) -> Tensor:
            return self.expert_down[idx](
                self.expert_up[idx](x) * F.silu(self.expert_gate[idx](x))
            )

        def forward(
            self, x: Tensor, regime_posteriors: Tensor
        ) -> tuple[Tensor, dict[str, Tensor]]:
            if regime_posteriors.shape[-1] != self.n_experts:
                raise ValueError(
                    "regime_posteriors last dimension must equal n_experts"
                )
            x_norm = self.pre_norm(x)
            top_values, top_indices = torch.topk(
                regime_posteriors, k=self.active_experts, dim=-1
            )
            routing_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(
                1e-9
            )

            expert_outputs = [self._expert_forward(x_norm, i) for i in range(self.n_experts)]
            combined = torch.zeros_like(x_norm)
            for expert_idx, expert_output in enumerate(expert_outputs):
                expert_weight = torch.zeros_like(regime_posteriors[..., 0])
                for k in range(self.active_experts):
                    expert_weight = expert_weight + routing_weights[..., k] * (
                        top_indices[..., k] == expert_idx
                    ).float()
                combined = combined + expert_weight.unsqueeze(-1) * expert_output

            output = self.post_norm(x + combined)

            token_count = float(top_indices[..., 0].numel())
            usage = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
            for expert_idx in range(self.n_experts):
                usage[expert_idx] = (top_indices == expert_idx).float().sum() / token_count
            posterior_mean = regime_posteriors.mean(dim=(0, 1))
            load_balance_loss = self.n_experts * torch.sum(usage * posterior_mean)

            aux = {
                "expert_usage": usage,
                "load_balance_loss": load_balance_loss,
                "top_indices": top_indices,
                "routing_weights": routing_weights,
            }
            return output, aux


    class MeridianPhase1Backbone(nn.Module):
        def __init__(
            self,
            config: Phase1CoreConfig = Phase1CoreConfig(),
            n_blocks: int = 8,
            moe_positions: tuple[int, ...] = (2, 5, 7),
        ) -> None:
            super().__init__()
            self.config = config
            self.n_blocks = n_blocks
            self.regime_detector = RegimeDetector(
                input_dim=config.regime_input_dim,
                regime_dim=config.regime_dim,
                n_regimes=config.n_regimes,
            )
            self.blocks = nn.ModuleList([RAGDNBlock(config) for _ in range(n_blocks)])
            self.moe_by_position = nn.ModuleDict(
                {
                    str(position): RegimeRoutedFMoEBlock(
                        model_dim=config.model_dim,
                        n_experts=config.n_regimes,
                        active_experts=2,
                    )
                    for position in moe_positions
                }
            )

        def forward(
            self,
            x: Tensor,
            regime_input: Tensor,
            hmm_posteriors: Tensor | None = None,
        ) -> tuple[Tensor, dict[str, Any]]:
            detector_out = self.regime_detector(regime_input, hmm_posteriors=hmm_posteriors)
            out = x
            state_history: list[Tensor] = []
            alpha_ranges: list[tuple[float, float]] = []
            beta_ranges: list[tuple[float, float]] = []
            moe_aux: dict[str, dict[str, Tensor]] = {}

            state = None
            for block_idx, block in enumerate(self.blocks):
                out, state, aux = block(out, detector_out["regime_embedding"], state=state)
                state_history.append(state)
                alpha_ranges.append(
                    (
                        float(aux["alpha"].min().detach().cpu().item()),
                        float(aux["alpha"].max().detach().cpu().item()),
                    )
                )
                beta_ranges.append(
                    (
                        float(aux["beta"].min().detach().cpu().item()),
                        float(aux["beta"].max().detach().cpu().item()),
                    )
                )

                moe = (
                    self.moe_by_position[str(block_idx)]
                    if str(block_idx) in self.moe_by_position
                    else None
                )
                if moe is not None:
                    out, moe_block_aux = moe(out, detector_out["final_posteriors"])
                    moe_aux[str(block_idx)] = moe_block_aux

            diagnostics: dict[str, Any] = {
                "regime": detector_out,
                "state_history": state_history,
                "alpha_ranges": alpha_ranges,
                "beta_ranges": beta_ranges,
                "moe_aux": moe_aux,
            }
            return out, diagnostics

else:
    class RegimeDetector:  # pragma: no cover - import guard path
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ensure_torch()


    class RAGDNBlock:  # pragma: no cover - import guard path
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ensure_torch()


    class RegimeRoutedFMoEBlock:  # pragma: no cover - import guard path
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ensure_torch()


    class MeridianPhase1Backbone:  # pragma: no cover - import guard path
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ensure_torch()


def run_phase1_smoke_check(
    batch_size: int = 2,
    seq_len: int = 16,
    config: Phase1CoreConfig = Phase1CoreConfig(),
    device: str | None = None,
) -> dict[str, object]:
    _ensure_torch()
    assert torch is not None

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = RegimeDetector(
        input_dim=config.regime_input_dim,
        regime_dim=config.regime_dim,
        n_regimes=config.n_regimes,
    ).to(run_device)
    block = RAGDNBlock(config).to(run_device)

    regime_input = torch.randn(batch_size, seq_len, config.regime_input_dim, device=run_device)
    x = torch.randn(batch_size, seq_len, config.model_dim, device=run_device)

    detector_out = detector(regime_input)
    out, state, aux = block(x, detector_out["regime_embedding"])

    summary = {
        "device": run_device,
        "regime_embedding_shape": list(detector_out["regime_embedding"].shape),
        "posteriors_shape": list(detector_out["final_posteriors"].shape),
        "velocity_shape": list(detector_out["regime_change_velocity"].shape),
        "block_output_shape": list(out.shape),
        "state_shape": list(state.shape),
        "alpha_min": float(aux["alpha"].min().detach().cpu().item()),
        "alpha_max": float(aux["alpha"].max().detach().cpu().item()),
        "beta_min": float(aux["beta"].min().detach().cpu().item()),
        "beta_max": float(aux["beta"].max().detach().cpu().item()),
    }
    return summary


def run_phase1_backbone_smoke_check(
    batch_size: int = 2,
    seq_len: int = 16,
    config: Phase1CoreConfig = Phase1CoreConfig(),
    device: str | None = None,
) -> dict[str, object]:
    _ensure_torch()
    assert torch is not None

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MeridianPhase1Backbone(config).to(run_device)

    x = torch.randn(batch_size, seq_len, config.model_dim, device=run_device)
    regime_input = torch.randn(batch_size, seq_len, config.regime_input_dim, device=run_device)
    output, diagnostics = model(x, regime_input)

    moe_usage = {
        key: value["expert_usage"].detach().cpu().tolist()
        for key, value in diagnostics["moe_aux"].items()
    }
    summary = {
        "device": run_device,
        "backbone_output_shape": list(output.shape),
        "num_blocks": model.n_blocks,
        "num_moe_insertions": len(diagnostics["moe_aux"]),
        "moe_usage": moe_usage,
        "alpha_ranges": diagnostics["alpha_ranges"],
        "beta_ranges": diagnostics["beta_ranges"],
        "regime_embedding_shape": list(
            diagnostics["regime"]["regime_embedding"].shape
        ),
        "regime_posterior_shape": list(
            diagnostics["regime"]["final_posteriors"].shape
        ),
    }
    return summary
