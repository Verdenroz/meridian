from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    model_dim: int = 768
    ra_gdn_blocks: int = 8
    fmoe_blocks: int = 3
    gdn_heads: int = 4
    gdn_head_dim: int = 64
    gdn_value_dim: int = 192
    gdn_conv_kernel: int = 4
    experts: int = 4
    active_experts: int = 2
    regime_dim: int = 64
    regimes: int = 4
    hourly_dim: int = 512
    daily_dim: int = 512
    weekly_dim: int = 384
    cross_asset_attention_heads: int = 8
    cross_asset_attention_layers: int = 2
    episodic_slots: int = 50_000
    episodic_dim: int = 128
    retrieval_count: int = 8
    market_reader_dim: int = 256
    market_reader_layers: int = 4
    market_reader_vocab: int = 4096
    market_reader_output_dim: int = 64
    max_assets: int = 100

    def validate(self) -> None:
        if self.active_experts > self.experts:
            raise ValueError("active_experts cannot exceed experts")
        if self.model_dim <= 0 or self.max_assets <= 0:
            raise ValueError("model_dim and max_assets must be positive")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    batch_size: int
    sequence_length: int
    steps: int
    gradient_clip: float
    precision: str

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


DEFAULT_ARCHITECTURE = ArchitectureSpec()

TRAINING_PHASES: dict[str, TrainingPhase] = {
    "phase1": TrainingPhase(
        name="self_supervised_pretraining",
        optimizer="AdamW",
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        batch_size=32,
        sequence_length=1024,
        steps=100_000,
        gradient_clip=1.0,
        precision="BF16",
    ),
    "phase2": TrainingPhase(
        name="portfolio_optimization",
        optimizer="AdamW",
        learning_rate=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        batch_size=16,
        sequence_length=2048,
        steps=80_000,
        gradient_clip=1.0,
        precision="BF16",
    ),
    "phase3": TrainingPhase(
        name="meta_learning_plasticity",
        optimizer="Adam",
        learning_rate=1e-5,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        batch_size=8,
        sequence_length=1000,
        steps=10_000,
        gradient_clip=0.5,
        precision="BF16",
    ),
    "phase4": TrainingPhase(
        name="quantization_aware_training",
        optimizer="AdamW",
        learning_rate=1e-5,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        batch_size=32,
        sequence_length=1024,
        steps=20_000,
        gradient_clip=1.0,
        precision="QAT",
    ),
}

