"""3-tier continuous learning system — Section 8 of meridian-spec.md.

Tier 1 — Surprise-Gated Delta Dynamics (Section 8.1):
  Operates every forward step (per-minute at inference).
  surprise_t = ||v_t - S_{t-1}^T k_t||^2  — GDN prediction error
  beta_t *= (1 + gamma * surprise_t)       — modulate update strength
  gamma = 0.1 (plasticity coefficient)
  No parameter updates — purely dynamic state modulation.

Tier 2 — Normalization Adaptation (Section 8.2):
  Operates every 60 steps (minutes).
  Online update of RMSNorm running statistics using exponential moving average.
  Only RMSNorm scale parameters are updated (tiny fraction of params).
  Learning rate: 1e-5, EMA decay: 0.99

Tier 3 — Periodic Consolidation (Section 8.3):
  Operates daily (overnight, during market close).
  EWC: L_ewc = lambda * sum_i F_i * (theta_i - theta*_i)^2
  Ternary LoRA: rank=4, applied to RA-GDN weight matrices only.
  Updated on recent 7-day replay buffer.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.regime_aware_backbone import RAGDNBackbone, RMSNorm

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────���─────────────────────────


@dataclass
class AdaptationConfig:
    """Hyperparameters for the 3-tier adaptation system."""

    # Tier 1
    gamma: float = 0.1  # plasticity coefficient

    # Tier 2
    tier2_lr: float = 1e-5
    tier2_ema_decay: float = 0.99
    tier2_interval_steps: int = 60  # steps between Tier 2 updates

    # Tier 3
    ewc_lambda: float = 1000.0
    lora_rank: int = 4
    fisher_samples: int = 256  # mini-batches for Fisher estimation
    consolidation_steps: int = 500  # gradient steps per nightly consolidation
    consolidation_lr: float = 1e-4

    # Replay buffer
    replay_buffer_days: int = 7

    # Feedback loop
    max_consecutive_failures: int = 3


# ── Tier 1 — Surprise-Gated Plasticity ──────────────────────────────────────


class SurpriseTracker:
    """Exponential moving average and spike counter for surprise signal."""

    def __init__(self, ema_decay: float = 0.99) -> None:
        self._ema: float = 0.0
        self._ema_sq: float = 0.0
        self._decay = ema_decay
        self._count: int = 0
        self._spike_count: int = 0

    def update(self, surprise: float) -> None:
        self._ema = self._decay * self._ema + (1 - self._decay) * surprise
        self._ema_sq = self._decay * self._ema_sq + (1 - self._decay) * surprise ** 2
        self._count += 1
        std = max((self._ema_sq - self._ema ** 2) ** 0.5, 1e-8)
        if surprise > self._ema + 3 * std:
            self._spike_count += 1

    @property
    def ema(self) -> float:
        return self._ema

    @property
    def spike_frequency(self) -> float:
        return self._spike_count / max(self._count, 1)

    def reset_spikes(self) -> None:
        self._spike_count = 0
        self._count = 0


class Tier1SurpriseGate:
    """Attaches surprise-gated beta modulation hooks to all GDN blocks."""

    def __init__(self, gamma: float = 0.1) -> None:
        self.gamma = gamma
        self.tracker = SurpriseTracker()

    def _hook(self, surprise: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        surprise_scalar = surprise.mean().item()
        self.tracker.update(surprise_scalar)
        modulation = 1.0 + self.gamma * surprise.view(-1, 1, 1, 1)
        return beta * modulation

    def attach(self, backbone: RAGDNBackbone) -> None:
        for block in backbone.gdn_blocks:
            block.surprise_hook = self._hook

    def detach(self, backbone: RAGDNBackbone) -> None:
        for block in backbone.gdn_blocks:
            block.surprise_hook = None


# ── Tier 2 — Normalization Adaptation ──────────────────────────────────��─────


class Tier2NormAdapter:
    """Online EMA update of RMSNorm parameters.

    Collects all RMSNorm modules in a model, snapshots their weights at
    init (checkpoint), and applies EMA-weighted updates on a schedule.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        ema_decay: float = 0.99,
    ) -> None:
        self.lr = lr
        self.ema_decay = ema_decay
        self._norms: list[tuple[str, RMSNorm]] = []
        self._checkpoints: dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if isinstance(module, RMSNorm):
                self._norms.append((name, module))
                self._checkpoints[name] = module.weight.data.clone()

    def step(self) -> None:
        """Apply one EMA update to all RMSNorm weights."""
        for name, norm in self._norms:
            norm.weight.data.mul_(self.ema_decay).add_(
                self._checkpoints[name], alpha=1 - self.ema_decay
            )

    def drift(self) -> float:
        """L2 distance of current norm params from checkpoint."""
        total = 0.0
        for name, norm in self._norms:
            diff = norm.weight.data - self._checkpoints[name]
            total += diff.pow(2).sum().item()
        return total ** 0.5

    def update_checkpoint(self) -> None:
        """Snapshot current norm params as the new checkpoint."""
        for name, norm in self._norms:
            self._checkpoints[name] = norm.weight.data.clone()


# ── Tier 3 — EWC + Ternary LoRA ─────────────────────────────────────────────


def _ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to {-1, 0, +1} with absmean scaling."""
    scale = w.abs().mean().clamp(min=1e-8)
    w_t = (w / scale).round().clamp(-1, 1)
    return w_t, scale


class TernaryLoRA(nn.Module):
    """Rank-r LoRA adapter with ternary-quantized A and B matrices.

    Wraps a target ``nn.Linear`` by adding a low-rank residual:
        output += (x @ A^T) @ B^T * scale

    A and B are ternary-quantized in forward using STE.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4) -> None:
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.scaling = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ternary LoRA residual for input x."""
        A_t, A_s = _ternary_quantize(self.A)
        B_t, B_s = _ternary_quantize(self.B)
        # STE: use quantized in forward, pass gradients through as identity
        A_q = self.A + (A_t * A_s - self.A).detach()
        B_q = self.B + (B_t * B_s - self.B).detach()
        return (x @ A_q.T) @ B_q.T * self.scaling

    def magnitude(self) -> float:
        """L2 norm of the effective adapter weight."""
        with torch.no_grad():
            A_t, A_s = _ternary_quantize(self.A)
            B_t, B_s = _ternary_quantize(self.B)
            effective = (B_t * B_s) @ (A_t * A_s) * self.scaling
            return effective.norm().item()


# Target projections in each GDN layer for LoRA adaptation
_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "alpha_proj", "beta_proj")


class LoRAManager:
    """Manages LoRA adapters across all GDN layers in the backbone."""

    def __init__(self, backbone: RAGDNBackbone, rank: int = 4) -> None:
        self.adapters: dict[str, TernaryLoRA] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        for block_idx, block in enumerate(backbone.gdn_blocks):
            for attr_name in _LORA_TARGETS:
                linear = getattr(block, attr_name)
                key = f"gdn_{block_idx}.{attr_name}"
                adapter = TernaryLoRA(linear.in_features, linear.out_features, rank)
                adapter = adapter.to(linear.weight.device)
                self.adapters[key] = adapter

    def attach(self, backbone: RAGDNBackbone) -> None:
        """Register forward hooks that add LoRA residuals."""
        self.detach()
        for block_idx, block in enumerate(backbone.gdn_blocks):
            for attr_name in _LORA_TARGETS:
                linear = getattr(block, attr_name)
                key = f"gdn_{block_idx}.{attr_name}"
                adapter = self.adapters[key]

                def _make_hook(a: TernaryLoRA):
                    def hook(_module: nn.Module, _input: tuple, output: torch.Tensor) -> torch.Tensor:
                        return output + a(_input[0])
                    return hook

                h = linear.register_forward_hook(_make_hook(adapter))
                self._hooks.append(h)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def parameters(self):
        for adapter in self.adapters.values():
            yield from adapter.parameters()

    def total_magnitude(self) -> float:
        return sum(a.magnitude() for a in self.adapters.values())

    def state_dict(self) -> dict:
        return {k: v.state_dict() for k, v in self.adapters.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            if k in self.adapters:
                self.adapters[k].load_state_dict(v)


class FisherEstimator:
    """Diagonal Fisher information via squared gradients."""

    @staticmethod
    def estimate(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn,
        max_batches: int = 256,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher for all parameters with requires_grad.

        Returns dict mapping param name -> Fisher diagonal tensor (on CPU).
        """
        device = device or next(model.parameters()).device
        fisher: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param, device="cpu")

        model.eval()
        n = 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            model.zero_grad()
            loss = loss_fn(batch, device)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().cpu().pow(2)
            n += 1

        for name in fisher:
            fisher[name] /= max(n, 1)

        model.train()
        return fisher


class EWCRegularizer:
    """Elastic Weight Consolidation penalty.

    L_ewc = lambda * sum_i F_i * (theta_i - theta*_i)^2
    """

    def __init__(
        self,
        theta_star: dict[str, torch.Tensor],
        fisher_diag: dict[str, torch.Tensor],
        ewc_lambda: float = 1000.0,
    ) -> None:
        self._theta_star = theta_star
        self._fisher = fisher_diag
        self.ewc_lambda = ewc_lambda

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty on the model's current parameters."""
        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self._fisher:
                f = self._fisher[name].to(device)
                t_star = self._theta_star[name].to(device)
                total = total + (f * (param - t_star).pow(2)).sum()
        return self.ewc_lambda * total

    def update(
        self,
        model: nn.Module,
        fisher_diag: dict[str, torch.Tensor],
    ) -> None:
        """Update theta_star and Fisher after a consolidation round."""
        for name, param in model.named_parameters():
            if name in self._fisher:
                self._theta_star[name] = param.data.clone().cpu()
        self._fisher = fisher_diag


class Tier3Consolidator:
    """Orchestrates nightly EWC + LoRA consolidation."""

    def __init__(
        self,
        model: nn.Module,
        backbone: RAGDNBackbone,
        lora: LoRAManager,
        cfg: AdaptationConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.backbone = backbone
        self.lora = lora
        self.cfg = cfg
        self.device = device

        # Snapshot theta_star from current params
        theta_star = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.ewc = EWCRegularizer(theta_star, {}, cfg.ewc_lambda)
        self._last_loss: float | None = None

    def consolidate(
        self,
        replay_loader: DataLoader,
        loss_fn,
    ) -> float:
        """Run one nightly consolidation cycle.

        1. Estimate Fisher diagonal from replay buffer.
        2. Fine-tune LoRA adapters with EWC regularization.
        3. Update theta_star and Fisher.

        Returns the final consolidation loss.
        """
        # Step 1: Fisher estimation
        fisher = FisherEstimator.estimate(
            self.model,
            replay_loader,
            loss_fn,
            max_batches=self.cfg.fisher_samples,
            device=self.device,
        )
        self.ewc.update(self.model, fisher)

        # Step 2: fine-tune LoRA with EWC
        self.lora.attach(self.backbone)
        lora_optimizer = torch.optim.AdamW(
            self.lora.parameters(), lr=self.cfg.consolidation_lr
        )

        total_loss = 0.0
        n = 0
        data_iter = iter(replay_loader)

        for step in range(self.cfg.consolidation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(replay_loader)
                batch = next(data_iter)

            lora_optimizer.zero_grad(set_to_none=True)
            task_loss = loss_fn(batch, self.device)
            ewc_pen = self.ewc.penalty(self.model)
            loss = task_loss + ewc_pen
            loss.backward()
            lora_optimizer.step()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)

        # Step 3: update reference
        self.ewc.update(self.model, fisher)
        self._last_loss = avg_loss

        logger.info(
            "Tier 3 consolidation: loss=%.4f  lora_mag=%.4f",
            avg_loss,
            self.lora.total_magnitude(),
        )
        return avg_loss

    @property
    def last_loss(self) -> float | None:
        return self._last_loss


# ── Adaptation Feedback Loop ───────────────────────���─────────────────────────


class AdaptationFeedbackLoop:
    """Tracks consecutive consolidation failures and adjusts parameters."""

    def __init__(self, max_failures: int = 3) -> None:
        self.max_failures = max_failures
        self._prev_loss: float | None = None
        self._consecutive_failures: int = 0

    def on_consolidation(self, loss: float, cfg: AdaptationConfig) -> list[str]:
        """Report consolidation result.  Returns list of alert messages."""
        alerts: list[str] = []

        if self._prev_loss is not None and loss >= self._prev_loss:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        self._prev_loss = loss

        if self._consecutive_failures >= self.max_failures:
            cfg.replay_buffer_days = min(cfg.replay_buffer_days + 3, 30)
            alerts.append(
                f"Consolidation loss failed to decrease for "
                f"{self._consecutive_failures} consecutive nights. "
                f"Increased replay buffer to {cfg.replay_buffer_days} days."
            )
            if cfg.lora_rank > 1:
                cfg.lora_rank = max(1, cfg.lora_rank - 1)
                alerts.append(f"Reduced LoRA rank to {cfg.lora_rank}.")
            alerts.append(
                "ALERT: market structure may have shifted beyond adaptation capacity."
            )
            self._consecutive_failures = 0

        return alerts

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


# ── Top-Level Facade ─────────────────────────────────────────────────────────


class AdaptationSystem:
    """Unified interface for the 3-tier adaptation system.

    Usage::

        adapt = AdaptationSystem(backbone, model, cfg, device)
        adapt.attach()

        # During inference loop:
        for step in range(...):
            output = model(input)
            adapt.on_step(step)

        # During market close:
        adapt.on_market_close(replay_loader, loss_fn)
    """

    def __init__(
        self,
        backbone: RAGDNBackbone,
        model: nn.Module,
        cfg: AdaptationConfig,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.backbone = backbone

        self.tier1 = Tier1SurpriseGate(gamma=cfg.gamma)
        self.tier2 = Tier2NormAdapter(model, lr=cfg.tier2_lr, ema_decay=cfg.tier2_ema_decay)
        self.lora = LoRAManager(backbone, rank=cfg.lora_rank)
        self.tier3 = Tier3Consolidator(model, backbone, self.lora, cfg, device)
        self.feedback = AdaptationFeedbackLoop(cfg.max_consecutive_failures)

    def attach(self) -> None:
        """Wire all adaptation hooks into the model."""
        self.tier1.attach(self.backbone)
        self.lora.attach(self.backbone)

    def detach(self) -> None:
        """Remove all adaptation hooks."""
        self.tier1.detach(self.backbone)
        self.lora.detach()

    def on_step(self, step: int) -> None:
        """Called after each forward pass.  Tier 1 runs via hook; Tier 2 on schedule."""
        if step > 0 and step % self.cfg.tier2_interval_steps == 0:
            self.tier2.step()

    def on_market_close(self, replay_loader: DataLoader, loss_fn) -> list[str]:
        """Trigger Tier 3 nightly consolidation.  Returns any alert messages."""
        loss = self.tier3.consolidate(replay_loader, loss_fn)
        return self.feedback.on_consolidation(loss, self.cfg)

    def health_metrics(self) -> dict[str, float]:
        """Snapshot of adaptation health for monitoring."""
        return {
            "surprise_ema": self.tier1.tracker.ema,
            "surprise_spike_freq": self.tier1.tracker.spike_frequency,
            "norm_drift": self.tier2.drift(),
            "lora_magnitude": self.lora.total_magnitude(),
            "consolidation_loss": self.tier3.last_loss or 0.0,
            "consecutive_failures": float(self.feedback.consecutive_failures),
        }

    def state_dict(self) -> dict:
        return {
            "lora": self.lora.state_dict(),
            "cfg": {
                "gamma": self.cfg.gamma,
                "lora_rank": self.cfg.lora_rank,
                "replay_buffer_days": self.cfg.replay_buffer_days,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if "lora" in state:
            self.lora.load_state_dict(state["lora"])
        if "cfg" in state:
            for k, v in state["cfg"].items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
