from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .regime_aware_backbone import (
    MeridianBackbone,
    BackboneCoreConfig,
    TORCH_AVAILABLE,
)
from train.feature_schema import PER_ASSET_FEATURE_DIM

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
            "Backbone pretraining requires torch. Install with: pip install -e '.[train]'"
        )


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_TRACK = "backbone_pretraining"


@dataclass(frozen=True)
class BackboneTrainingConfig:
    steps: int = 20
    batch_size: int = 4
    seq_len: int = 32
    learning_rate: float = 3e-4
    reconstruction_weight: float = 1.0
    regime_smoothness_weight: float = 0.1
    moe_balance_weight: float = 0.01
    return_prediction_weight: float = 0.5
    volatility_prediction_weight: float = 0.3
    correlation_prediction_weight: float = 0.2
    prediction_ramp_steps: int = 0
    grad_clip_norm: float = 1.0
    core: BackboneCoreConfig = BackboneCoreConfig()

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.batch_size < 1 or self.seq_len < 1:
            raise ValueError("batch_size and seq_len must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")
        if self.reconstruction_weight < 0 or self.regime_smoothness_weight < 0:
            raise ValueError("reconstruction_weight and regime_smoothness_weight must be >= 0")
        if self.moe_balance_weight < 0:
            raise ValueError("moe_balance_weight must be >= 0")
        if (
            self.return_prediction_weight < 0
            or self.volatility_prediction_weight < 0
            or self.correlation_prediction_weight < 0
        ):
            raise ValueError("prediction weights must be >= 0")
        if self.prediction_ramp_steps < 0:
            raise ValueError("prediction_ramp_steps must be >= 0")


def build_backbone_pretraining_spec(
    config: BackboneTrainingConfig,
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


def validate_checkpoint_compatibility(
    checkpoint_payload: dict[str, Any],
    current_config: BackboneTrainingConfig,
    requires_input_projection: bool,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    checkpoint_version = checkpoint_payload.get("checkpoint_format_version")
    if checkpoint_version is None:
        warnings.append("checkpoint missing checkpoint_format_version; assuming legacy format")
    elif not isinstance(checkpoint_version, int):
        errors.append(
            "checkpoint_format_version must be an integer for compatibility validation"
        )
    elif checkpoint_version > CHECKPOINT_FORMAT_VERSION:
        errors.append(
            "checkpoint_format_version "
            f"{checkpoint_version} is newer than supported {CHECKPOINT_FORMAT_VERSION}"
        )
    elif checkpoint_version < CHECKPOINT_FORMAT_VERSION:
        warnings.append(
            "checkpoint_format_version "
            f"{checkpoint_version} is older than current {CHECKPOINT_FORMAT_VERSION}"
        )

    checkpoint_track = checkpoint_payload.get("checkpoint_track")
    if checkpoint_track is None:
        warnings.append("checkpoint missing checkpoint_track; assuming legacy backbone artifact")
    elif checkpoint_track != CHECKPOINT_TRACK:
        errors.append(
            f"checkpoint_track mismatch: checkpoint={checkpoint_track} expected={CHECKPOINT_TRACK}"
        )

    required_keys = (
        "backbone_state_dict",
        "reconstruction_head_state_dict",
        "return_head_state_dict",
        "volatility_head_state_dict",
        "correlation_head_state_dict",
    )
    for key in required_keys:
        if key not in checkpoint_payload:
            errors.append(f"missing checkpoint key: {key}")

    checkpoint_config = checkpoint_payload.get("config")
    checkpoint_core: dict[str, Any] | None = None
    if isinstance(checkpoint_config, dict):
        core_payload = checkpoint_config.get("core")
        if isinstance(core_payload, dict):
            checkpoint_core = core_payload
        else:
            warnings.append("checkpoint config has no core section; skipping core validation")
    else:
        warnings.append("checkpoint has no config section; skipping core validation")

    if checkpoint_core is not None:
        current_core = asdict(current_config.core)
        for field in (
            "model_dim",
            "regime_input_dim",
            "regime_dim",
            "n_regimes",
            "n_heads",
            "d_head",
            "d_v",
            "conv_kernel",
        ):
            checkpoint_value = checkpoint_core.get(field)
            if checkpoint_value != current_core[field]:
                errors.append(
                    f"core.{field} mismatch: checkpoint={checkpoint_value} current={current_core[field]}"
                )

    if requires_input_projection:
        if checkpoint_payload.get("input_projection_state_dict") is None:
            warnings.append("checkpoint missing input projection state; using fresh input projection")
        if checkpoint_payload.get("regime_projection_state_dict") is None:
            warnings.append(
                "checkpoint missing regime projection state; using fresh regime projection"
            )

    return {
        "compatible": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checkpoint_format_version": (
            checkpoint_version if isinstance(checkpoint_version, int) else None
        ),
        "supported_checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_track": checkpoint_track if isinstance(checkpoint_track, str) else None,
        "supported_checkpoint_track": CHECKPOINT_TRACK,
    }


def _extract_moe_balance_loss(moe_aux: dict[str, dict[str, Tensor]], device: str) -> Tensor:
    assert torch is not None
    if not moe_aux:
        return torch.tensor(0.0, device=device)
    values = [aux["load_balance_loss"] for aux in moe_aux.values()]
    return torch.stack(values).mean()


def resolve_objective_weights(
    config: BackboneTrainingConfig, step_index: int
) -> dict[str, float]:
    if step_index < 0:
        raise ValueError("step_index must be >= 0")

    prediction_scale = 1.0
    if config.prediction_ramp_steps > 0:
        prediction_scale = min((step_index + 1) / config.prediction_ramp_steps, 1.0)

    return {
        "reconstruction_weight": config.reconstruction_weight,
        "regime_smoothness_weight": config.regime_smoothness_weight,
        "moe_balance_weight": config.moe_balance_weight,
        "return_prediction_weight": config.return_prediction_weight * prediction_scale,
        "volatility_prediction_weight": (
            config.volatility_prediction_weight * prediction_scale
        ),
        "correlation_prediction_weight": (
            config.correlation_prediction_weight * prediction_scale
        ),
    }


def load_f32_feature_shard_rows(
    shard_path: Path,
    feature_dim: int = PER_ASSET_FEATURE_DIM,
) -> list[list[float]]:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if not shard_path.exists():
        raise FileNotFoundError(f"Feature shard not found: {shard_path}")

    row_struct = struct.Struct(f"<{feature_dim}f")
    payload = shard_path.read_bytes()
    if len(payload) % row_struct.size != 0:
        raise ValueError(
            f"Invalid shard size for feature_dim={feature_dim}: {len(payload)} bytes"
        )

    rows: list[list[float]] = []
    for offset in range(0, len(payload), row_struct.size):
        rows.append(list(row_struct.unpack(payload[offset : offset + row_struct.size])))
    return rows


def build_shard_sequence_windows(
    rows: list[list[float]],
    seq_len: int,
    max_sequences: int | None = None,
) -> list[list[list[float]]]:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("max_sequences must be >= 1 when provided")
    if len(rows) < seq_len:
        raise ValueError(
            f"Not enough rows for seq_len={seq_len}: got {len(rows)} rows"
        )

    windows: list[list[list[float]]] = []
    limit = max_sequences if max_sequences is not None else (len(rows) // seq_len)
    if limit < 1:
        raise ValueError("No complete sequence windows available from shard rows")

    cursor = 0
    while cursor + seq_len <= len(rows) and len(windows) < limit:
        windows.append(rows[cursor : cursor + seq_len])
        cursor += seq_len
    return windows


def build_multi_shard_sequence_windows(
    shard_paths: list[Path],
    seq_len: int,
    max_sequences_per_shard: int | None = None,
) -> list[list[list[float]]]:
    if not shard_paths:
        raise ValueError("shard_paths cannot be empty")
    if max_sequences_per_shard is not None and max_sequences_per_shard < 1:
        raise ValueError("max_sequences_per_shard must be >= 1 when provided")

    windows: list[list[list[float]]] = []
    for shard_path in shard_paths:
        rows = load_f32_feature_shard_rows(shard_path)
        shard_windows = build_shard_sequence_windows(
            rows,
            seq_len=seq_len,
            max_sequences=max_sequences_per_shard,
        )
        windows.extend(shard_windows)

    if not windows:
        raise ValueError("No sequence windows produced from shard_paths")
    return windows


def discover_split_shards(
    shards_root: Path,
    splits: list[str],
    suffix: str = ".f32bin",
) -> dict[str, list[Path]]:
    if not shards_root.exists():
        raise FileNotFoundError(f"Shard root not found: {shards_root}")
    if not splits:
        raise ValueError("splits cannot be empty")

    discovered: dict[str, list[Path]] = {}
    for split in splits:
        split_dir = shards_root / split
        if not split_dir.exists():
            discovered[split] = []
            continue
        discovered[split] = sorted(split_dir.glob(f"*{suffix}"))
    return discovered


def _build_prediction_targets(
    batch_source: Tensor | None,
    batch_size: int,
    seq_len: int,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    assert torch is not None
    if batch_source is None:
        return_target = torch.randn(batch_size, seq_len, 1, device=device)
        volatility_target = torch.abs(torch.randn(batch_size, seq_len, 1, device=device))
        correlation_target = torch.tanh(torch.randn(batch_size, seq_len, 1, device=device))
        return return_target, volatility_target, correlation_target

    return_target = batch_source[..., 5:6]
    volatility_target = torch.abs(batch_source[..., 5:6])
    correlation_target = torch.tanh(batch_source[..., 6:7] - batch_source[..., 5:6])
    return return_target, volatility_target, correlation_target


def run_split_shard_evaluation(
    config: BackboneTrainingConfig,
    shards_root: Path,
    splits: list[str],
    checkpoint_output_path: Path,
    resume_from_checkpoint_path: Path | None = None,
    max_sequences: int | None = None,
    device: str | None = None,
    include_per_shard: bool = False,
    max_shards_per_split: int | None = None,
) -> dict[str, object]:
    if max_shards_per_split is not None and max_shards_per_split < 1:
        raise ValueError("max_shards_per_split must be >= 1 when provided")

    discovered = discover_split_shards(shards_root=shards_root, splits=splits)
    selected_by_split = {
        split: (
            discovered[split][:max_shards_per_split]
            if max_shards_per_split is not None
            else discovered[split]
        )
        for split in splits
    }
    total_shards = sum(len(paths) for paths in selected_by_split.values())
    if total_shards == 0:
        raise ValueError(f"No shards found under {shards_root} for splits {splits}")

    split_metrics: dict[str, dict[str, object]] = {}
    for split in splits:
        shard_paths = selected_by_split[split]
        discovered_count = len(discovered[split])
        if not shard_paths:
            split_metrics[split] = {
                "shard_count": 0,
                "discovered_shard_count": discovered_count,
            }
            continue
        summary = run_backbone_pretraining_smoke(
            config=config,
            checkpoint_output_path=checkpoint_output_path,
            device=device,
            input_shard_paths=shard_paths,
            max_sequences=max_sequences,
            resume_from_checkpoint_path=resume_from_checkpoint_path,
            eval_only=True,
        )
        split_metrics[split] = {
            "shard_count": len(shard_paths),
            "discovered_shard_count": discovered_count,
            "avg_total_loss": summary["avg_total_loss"],
            "avg_reconstruction_loss": summary["avg_reconstruction_loss"],
            "avg_regime_smoothness_loss": summary["avg_regime_smoothness_loss"],
            "avg_moe_balance_loss": summary["avg_moe_balance_loss"],
            "avg_return_prediction_loss": summary["avg_return_prediction_loss"],
            "avg_volatility_prediction_loss": summary["avg_volatility_prediction_loss"],
            "avg_correlation_prediction_loss": summary["avg_correlation_prediction_loss"],
        }
        if include_per_shard:
            per_shard_metrics: list[dict[str, object]] = []
            for shard_path in shard_paths:
                shard_summary = run_backbone_pretraining_smoke(
                    config=config,
                    checkpoint_output_path=checkpoint_output_path,
                    device=device,
                    input_shard_path=shard_path,
                    max_sequences=max_sequences,
                    resume_from_checkpoint_path=resume_from_checkpoint_path,
                    eval_only=True,
                )
                per_shard_metrics.append(
                    {
                        "shard_path": str(shard_path),
                        "avg_total_loss": shard_summary["avg_total_loss"],
                        "avg_reconstruction_loss": shard_summary["avg_reconstruction_loss"],
                        "avg_regime_smoothness_loss": shard_summary[
                            "avg_regime_smoothness_loss"
                        ],
                        "avg_moe_balance_loss": shard_summary["avg_moe_balance_loss"],
                        "avg_return_prediction_loss": shard_summary[
                            "avg_return_prediction_loss"
                        ],
                        "avg_volatility_prediction_loss": shard_summary[
                            "avg_volatility_prediction_loss"
                        ],
                        "avg_correlation_prediction_loss": shard_summary[
                            "avg_correlation_prediction_loss"
                        ],
                    }
                )
            split_metrics[split]["per_shard"] = per_shard_metrics

    available_losses = [
        metrics["avg_total_loss"]
        for metrics in split_metrics.values()
        if "avg_total_loss" in metrics
    ]
    return {
        "shards_root": str(shards_root),
        "splits": split_metrics,
        "include_per_shard": include_per_shard,
        "max_shards_per_split": max_shards_per_split,
        "overall_avg_total_loss": (
            sum(float(loss) for loss in available_losses) / len(available_losses)
            if available_losses
            else None
        ),
    }


def run_backbone_pretraining_smoke(
    config: BackboneTrainingConfig,
    checkpoint_output_path: Path,
    metrics_output_path: Path | None = None,
    device: str | None = None,
    input_shard_path: Path | None = None,
    input_shard_paths: list[Path] | None = None,
    max_sequences: int | None = None,
    resume_from_checkpoint_path: Path | None = None,
    eval_only: bool = False,
) -> dict[str, object]:
    _ensure_torch()
    assert torch is not None
    assert nn is not None
    assert F is not None

    config.validate()
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = MeridianBackbone(config=config.core).to(run_device)
    reconstruction_head = nn.Linear(config.core.model_dim, config.core.model_dim).to(
        run_device
    )
    return_head = nn.Linear(config.core.model_dim, 1).to(run_device)
    volatility_head = nn.Linear(config.core.model_dim, 1).to(run_device)
    correlation_head = nn.Linear(config.core.model_dim, 1).to(run_device)
    input_projection: nn.Module | None = None
    regime_projection: nn.Module | None = None
    source_sequences: Tensor | None = None
    if input_shard_path is not None and input_shard_paths:
        raise ValueError("Provide either input_shard_path or input_shard_paths, not both")
    if input_shard_path is not None:
        shard_rows = load_f32_feature_shard_rows(input_shard_path)
        windows = build_shard_sequence_windows(
            shard_rows,
            seq_len=config.seq_len,
            max_sequences=max_sequences,
        )
        source_sequences = torch.tensor(windows, dtype=torch.float32, device=run_device)
        input_projection = nn.Linear(PER_ASSET_FEATURE_DIM, config.core.model_dim).to(
            run_device
        )
        regime_projection = nn.Linear(
            PER_ASSET_FEATURE_DIM, config.core.regime_input_dim
        ).to(run_device)
    elif input_shard_paths:
        windows = build_multi_shard_sequence_windows(
            shard_paths=input_shard_paths,
            seq_len=config.seq_len,
            max_sequences_per_shard=max_sequences,
        )
        source_sequences = torch.tensor(windows, dtype=torch.float32, device=run_device)
        input_projection = nn.Linear(PER_ASSET_FEATURE_DIM, config.core.model_dim).to(
            run_device
        )
        regime_projection = nn.Linear(
            PER_ASSET_FEATURE_DIM, config.core.regime_input_dim
        ).to(run_device)

    train_modules: list[nn.Module] = [
        backbone,
        reconstruction_head,
        return_head,
        volatility_head,
        correlation_head,
    ]
    if input_projection is not None and regime_projection is not None:
        train_modules.extend([input_projection, regime_projection])
    train_parameters = [param for module in train_modules for param in module.parameters()]
    optimizer = torch.optim.AdamW(train_parameters, lr=config.learning_rate)
    compatibility_report: dict[str, object] | None = None

    if resume_from_checkpoint_path is not None:
        if not resume_from_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_from_checkpoint_path}"
            )
        checkpoint_payload = torch.load(
            resume_from_checkpoint_path,
            map_location=run_device,
        )
        compatibility_report = validate_checkpoint_compatibility(
            checkpoint_payload=checkpoint_payload,
            current_config=config,
            requires_input_projection=(
                input_projection is not None and regime_projection is not None
            ),
        )
        if not bool(compatibility_report["compatible"]):
            errors = compatibility_report["errors"]
            raise ValueError(
                "Incompatible resume checkpoint: " + "; ".join(str(error) for error in errors)
            )
        backbone.load_state_dict(checkpoint_payload["backbone_state_dict"])
        reconstruction_head.load_state_dict(
            checkpoint_payload["reconstruction_head_state_dict"]
        )
        if "return_head_state_dict" in checkpoint_payload:
            return_head.load_state_dict(checkpoint_payload["return_head_state_dict"])
        if "volatility_head_state_dict" in checkpoint_payload:
            volatility_head.load_state_dict(
                checkpoint_payload["volatility_head_state_dict"]
            )
        if "correlation_head_state_dict" in checkpoint_payload:
            correlation_head.load_state_dict(
                checkpoint_payload["correlation_head_state_dict"]
            )
        if input_projection is not None and checkpoint_payload.get(
            "input_projection_state_dict"
        ) is not None:
            input_projection.load_state_dict(checkpoint_payload["input_projection_state_dict"])
        if regime_projection is not None and checkpoint_payload.get(
            "regime_projection_state_dict"
        ) is not None:
            regime_projection.load_state_dict(
                checkpoint_payload["regime_projection_state_dict"]
            )
        if not eval_only and "optimizer_state_dict" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])

    total_loss_history: list[float] = []
    reconstruction_loss_history: list[float] = []
    regime_smoothness_loss_history: list[float] = []
    moe_balance_loss_history: list[float] = []
    return_prediction_loss_history: list[float] = []
    volatility_prediction_loss_history: list[float] = []
    correlation_prediction_loss_history: list[float] = []
    effective_return_weight_history: list[float] = []
    effective_volatility_weight_history: list[float] = []
    effective_correlation_weight_history: list[float] = []

    backbone.train()
    reconstruction_head.train()
    if eval_only:
        backbone.eval()
        reconstruction_head.eval()
        return_head.eval()
        volatility_head.eval()
        correlation_head.eval()
        if input_projection is not None:
            input_projection.eval()
        if regime_projection is not None:
            regime_projection.eval()

    for step in range(config.steps):
        if source_sequences is not None:
            assert input_projection is not None
            assert regime_projection is not None
            sequence_count = int(source_sequences.shape[0])
            start_idx = (step * config.batch_size) % sequence_count
            end_idx = start_idx + config.batch_size
            if end_idx <= sequence_count:
                batch_source = source_sequences[start_idx:end_idx]
            else:
                batch_source = torch.cat(
                    [
                        source_sequences[start_idx:],
                        source_sequences[: end_idx - sequence_count],
                    ],
                    dim=0,
                )
            batch_x = input_projection(batch_source)
            regime_input = regime_projection(batch_source)
        else:
            batch_x = torch.randn(
                config.batch_size, config.seq_len, config.core.model_dim, device=run_device
            )
            regime_input = torch.randn(
                config.batch_size,
                config.seq_len,
                config.core.regime_input_dim,
                device=run_device,
            )

        if eval_only:
            with torch.no_grad():
                encoded, diagnostics = backbone(batch_x, regime_input)
                reconstructed = reconstruction_head(encoded)
                return_prediction = return_head(encoded)
                volatility_prediction = F.softplus(volatility_head(encoded))
                correlation_prediction = torch.tanh(correlation_head(encoded))
        else:
            optimizer.zero_grad()
            encoded, diagnostics = backbone(batch_x, regime_input)
            reconstructed = reconstruction_head(encoded)
            return_prediction = return_head(encoded)
            volatility_prediction = F.softplus(volatility_head(encoded))
            correlation_prediction = torch.tanh(correlation_head(encoded))

        reconstruction_loss = F.mse_loss(reconstructed, batch_x)
        (
            return_target,
            volatility_target,
            correlation_target,
        ) = _build_prediction_targets(
            batch_source=batch_source if source_sequences is not None else None,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=run_device,
        )
        return_prediction_loss = F.mse_loss(return_prediction, return_target)
        volatility_prediction_loss = F.mse_loss(
            volatility_prediction, volatility_target
        )
        correlation_prediction_loss = F.mse_loss(
            correlation_prediction, correlation_target
        )
        posteriors = diagnostics["regime"]["final_posteriors"]
        if config.seq_len > 1:
            regime_smoothness_loss = F.mse_loss(posteriors[:, 1:, :], posteriors[:, :-1, :])
        else:
            regime_smoothness_loss = torch.tensor(0.0, device=run_device)
        moe_balance_loss = _extract_moe_balance_loss(
            diagnostics["moe_aux"], device=run_device
        )
        weights = resolve_objective_weights(config=config, step_index=step)

        total_loss = (
            weights["reconstruction_weight"] * reconstruction_loss
            + weights["regime_smoothness_weight"] * regime_smoothness_loss
            + weights["moe_balance_weight"] * moe_balance_loss
            + weights["return_prediction_weight"] * return_prediction_loss
            + weights["volatility_prediction_weight"] * volatility_prediction_loss
            + weights["correlation_prediction_weight"] * correlation_prediction_loss
        )
        if not eval_only:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, config.grad_clip_norm)
            optimizer.step()

        total_loss_history.append(float(total_loss.detach().cpu().item()))
        reconstruction_loss_history.append(float(reconstruction_loss.detach().cpu().item()))
        regime_smoothness_loss_history.append(
            float(regime_smoothness_loss.detach().cpu().item())
        )
        moe_balance_loss_history.append(float(moe_balance_loss.detach().cpu().item()))
        return_prediction_loss_history.append(
            float(return_prediction_loss.detach().cpu().item())
        )
        volatility_prediction_loss_history.append(
            float(volatility_prediction_loss.detach().cpu().item())
        )
        correlation_prediction_loss_history.append(
            float(correlation_prediction_loss.detach().cpu().item())
        )
        effective_return_weight_history.append(weights["return_prediction_weight"])
        effective_volatility_weight_history.append(
            weights["volatility_prediction_weight"]
        )
        effective_correlation_weight_history.append(
            weights["correlation_prediction_weight"]
        )

    if not eval_only:
        checkpoint_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone_state_dict": backbone.state_dict(),
                "reconstruction_head_state_dict": reconstruction_head.state_dict(),
                "return_head_state_dict": return_head.state_dict(),
                "volatility_head_state_dict": volatility_head.state_dict(),
                "correlation_head_state_dict": correlation_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "checkpoint_track": CHECKPOINT_TRACK,
                "checkpoint_created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_projection_state_dict": input_projection.state_dict()
                if input_projection is not None
                else None,
                "regime_projection_state_dict": regime_projection.state_dict()
                if regime_projection is not None
                else None,
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
        "return_prediction_loss_history": return_prediction_loss_history,
        "volatility_prediction_loss_history": volatility_prediction_loss_history,
        "correlation_prediction_loss_history": correlation_prediction_loss_history,
        "effective_return_prediction_weight_history": effective_return_weight_history,
        "effective_volatility_prediction_weight_history": effective_volatility_weight_history,
        "effective_correlation_prediction_weight_history": effective_correlation_weight_history,
        "avg_total_loss": sum(total_loss_history) / len(total_loss_history),
        "avg_reconstruction_loss": sum(reconstruction_loss_history)
        / len(reconstruction_loss_history),
        "avg_regime_smoothness_loss": sum(regime_smoothness_loss_history)
        / len(regime_smoothness_loss_history),
        "avg_moe_balance_loss": sum(moe_balance_loss_history)
        / len(moe_balance_loss_history),
        "avg_return_prediction_loss": sum(return_prediction_loss_history)
        / len(return_prediction_loss_history),
        "avg_volatility_prediction_loss": sum(volatility_prediction_loss_history)
        / len(volatility_prediction_loss_history),
        "avg_correlation_prediction_loss": sum(correlation_prediction_loss_history)
        / len(correlation_prediction_loss_history),
        "checkpoint_output_path": str(checkpoint_output_path) if not eval_only else None,
        "input_mode": (
            "multi_shard"
            if input_shard_paths
            else ("shard" if input_shard_path is not None else "synthetic")
        ),
        "input_shard_path": str(input_shard_path) if input_shard_path is not None else None,
        "input_shard_paths": [str(path) for path in input_shard_paths]
        if input_shard_paths
        else None,
        "resume_from_checkpoint_path": str(resume_from_checkpoint_path)
        if resume_from_checkpoint_path is not None
        else None,
        "checkpoint_compatibility": compatibility_report,
        "eval_only": eval_only,
        "prediction_ramp_steps": config.prediction_ramp_steps,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_track": CHECKPOINT_TRACK,
    }
    target_metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["metrics_output_path"] = str(target_metrics_path)
    return summary
