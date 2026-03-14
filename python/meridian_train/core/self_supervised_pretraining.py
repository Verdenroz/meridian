from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .regime_aware_backbone import (
    MeridianPhase1Backbone,
    Phase1CoreConfig,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
else:  # pragma: no cover - import guard path
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "Phase 1 training requires torch. Install with: pip install -e '.[train]'"
        )


@dataclass(frozen=True)
class Phase1TrainingConfig:
    steps: int = 20
    batch_size: int = 4
    seq_len: int = 32
    learning_rate: float = 3e-4
    reconstruction_weight: float = 1.0
    regime_smoothness_weight: float = 0.1
    moe_balance_weight: float = 0.01
    grad_clip_norm: float = 1.0
    core: Phase1CoreConfig = Phase1CoreConfig()

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.batch_size < 1 or self.seq_len < 1:
            raise ValueError("batch_size and seq_len must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")


def build_phase1_training_spec(
    config: Phase1TrainingConfig,
    checkpoint_output_path: Path,
    metrics_output_path: Path,
) -> dict[str, object]:
    config.validate()
    return {
        **asdict(config),
        "checkpoint_output_path": str(checkpoint_output_path),
        "metrics_output_path": str(metrics_output_path),
        "objective": "self_supervised_backbone_scaffold",
    }


def _extract_moe_balance_loss(moe_aux: dict[str, dict[str, Tensor]], device: str) -> Tensor:
    assert torch is not None
    if not moe_aux:
        return torch.tensor(0.0, device=device)
    values = [aux["load_balance_loss"] for aux in moe_aux.values()]
    return torch.stack(values).mean()


def run_phase1_training_smoke(
    config: Phase1TrainingConfig,
    checkpoint_output_path: Path,
    metrics_output_path: Path | None = None,
    device: str | None = None,
) -> dict[str, object]:
    _ensure_torch()
    assert torch is not None
    assert nn is not None
    assert F is not None

    config.validate()
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = MeridianPhase1Backbone(config=config.core).to(run_device)
    reconstruction_head = nn.Linear(config.core.model_dim, config.core.model_dim).to(
        run_device
    )
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(reconstruction_head.parameters()),
        lr=config.learning_rate,
    )

    total_loss_history: list[float] = []
    reconstruction_loss_history: list[float] = []
    regime_smoothness_loss_history: list[float] = []
    moe_balance_loss_history: list[float] = []

    backbone.train()
    reconstruction_head.train()
    for _ in range(config.steps):
        batch_x = torch.randn(
            config.batch_size, config.seq_len, config.core.model_dim, device=run_device
        )
        regime_input = torch.randn(
            config.batch_size,
            config.seq_len,
            config.core.regime_input_dim,
            device=run_device,
        )

        optimizer.zero_grad()
        encoded, diagnostics = backbone(batch_x, regime_input)
        reconstructed = reconstruction_head(encoded)

        reconstruction_loss = F.mse_loss(reconstructed, batch_x)
        posteriors = diagnostics["regime"]["final_posteriors"]
        if config.seq_len > 1:
            regime_smoothness_loss = F.mse_loss(posteriors[:, 1:, :], posteriors[:, :-1, :])
        else:
            regime_smoothness_loss = torch.tensor(0.0, device=run_device)
        moe_balance_loss = _extract_moe_balance_loss(
            diagnostics["moe_aux"], device=run_device
        )

        total_loss = (
            config.reconstruction_weight * reconstruction_loss
            + config.regime_smoothness_weight * regime_smoothness_loss
            + config.moe_balance_weight * moe_balance_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(reconstruction_head.parameters()),
            config.grad_clip_norm,
        )
        optimizer.step()

        total_loss_history.append(float(total_loss.detach().cpu().item()))
        reconstruction_loss_history.append(float(reconstruction_loss.detach().cpu().item()))
        regime_smoothness_loss_history.append(
            float(regime_smoothness_loss.detach().cpu().item())
        )
        moe_balance_loss_history.append(float(moe_balance_loss.detach().cpu().item()))

    checkpoint_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "reconstruction_head_state_dict": reconstruction_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        checkpoint_output_path,
    )

    target_metrics_path = metrics_output_path or checkpoint_output_path.with_suffix(
        ".metrics.json"
    )
    target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "steps": config.steps,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "device": run_device,
        "loss_history": total_loss_history,
        "reconstruction_loss_history": reconstruction_loss_history,
        "regime_smoothness_loss_history": regime_smoothness_loss_history,
        "moe_balance_loss_history": moe_balance_loss_history,
        "avg_total_loss": sum(total_loss_history) / len(total_loss_history),
        "avg_reconstruction_loss": sum(reconstruction_loss_history)
        / len(reconstruction_loss_history),
        "avg_regime_smoothness_loss": sum(regime_smoothness_loss_history)
        / len(regime_smoothness_loss_history),
        "avg_moe_balance_loss": sum(moe_balance_loss_history)
        / len(moe_balance_loss_history),
        "checkpoint_output_path": str(checkpoint_output_path),
    }
    target_metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["metrics_output_path"] = str(target_metrics_path)
    return summary
