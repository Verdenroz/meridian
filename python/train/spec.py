"""Architecture hyperparameter contracts for Meridian (Appendix C)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeridianSpec:
    # ── Versioning / data contract (v4 hard requirement) ─────────────────────
    architecture_version: str = "v4.0"
    data_interval: str = "1d"
    pretraining_objective: str = "tcp"

    # ── Model dimensions ────────────────────────────────────────────────────
    d_model: int = 768
    n_assets: int = 100
    n_regimes: int = 4
    regime_emb_dim: int = 64

    # ── Hybrid RA-GDN + SWA backbone ───────────────────────────────────────
    n_gdn_blocks: int = 8
    n_fmoe_blocks: int = 3
    fmoe_after_gdn: tuple[int, ...] = field(default_factory=lambda: (2, 5, 8))
    n_swa_layers: int = 2
    # SWA blocks inserted after GDN blocks 3 and 7 (1-indexed)
    swa_after_gdn: tuple[int, ...] = field(default_factory=lambda: (3, 7))
    swa_window: int = 64
    swa_heads: int = 8
    n_experts: int = 4
    top_k_experts: int = 2
    n_heads: int = 4
    d_head: int = 64
    d_v: int = 192
    # Expert FFN hidden dim — tuned so 3 × 4 experts ≈ 42.5M params
    expert_ffn_dim: int = 2304  # 768 × 3

    # ── Temporal cascade ────────────────────────────────────────────────────
    cascade_levels: int = 3
    cascade_dims: tuple[int, ...] = field(default_factory=lambda: (512, 512, 384))
    # How many minute-steps between updates at each cascade level
    cascade_strides: tuple[int, ...] = field(default_factory=lambda: (60, 390, 1950))

    # ── Cross-asset fusion ──────────────────────────────────────────────────
    n_fusion_layers: int = 2
    n_fusion_heads: int = 8
    n_episodic_retrievals: int = 8

    # ── Episodic memory ─────────────────────────────────────────────────────
    episodic_capacity: int = 50_000
    episodic_surprise_threshold: float = 2.0  # σ units

    # ── Event encoder ───────────────────────────────────────────────────────
    n_event_types: int = 18
    event_emb_dim: int = 64
    event_decay_lambda: float = 0.01  # per-minute exponential decay rate

    # ── Input dimensions ────────────────────────────────────────────────────
    per_asset_continuous_dim: int = 120
    per_asset_text_dim: int = 64
    market_feature_dim: int = 96
    portfolio_state_dim: int = 201
    calendar_dim: int = 20

    # ── Regime detector ─────────────────────────────────────────────────────
    # 96 (market) + 3 (portfolio summary) + 64 (rolling regime EMA)
    regime_detector_input_dim: int = 163

    # ── MarketReader ────────────────────────────────────────────────────────
    mr_dim: int = 256
    mr_n_layers: int = 4
    mr_n_heads: int = 4
    mr_d_head: int = 32
    mr_d_v: int = 64
    mr_vocab_size: int = 4_096
    mr_max_seq_len: int = 512
    headline_out_dim: int = 4
    summary_out_dim: int = 32
    deep_out_dim: int = 64

    # ── Training ────────────────────────────────────────────────────────────
    pretrain_steps: int = 20_000
    portfolio_steps_return: int = 50_000
    portfolio_steps_sharpe: int = 30_000
    metalearning_steps: int = 100_000
    qat_steps: int = 50_000
    train_batch_size: int = 128
    train_seq_len: int = 64
    learning_rate: float = 3e-4
    pretrain_min_lr: float = 1e-5
    tcp_temperature: float = 0.1
    tcp_lambda_contrastive: float = 0.7
    tcp_lambda_prediction: float = 0.3
    pretrain_fda_loss_weight: float = 0.5
    pretrain_positive_offset: int = 4
    dropout: float = 0.1

    @property
    def per_asset_feature_dim(self) -> int:
        return self.per_asset_continuous_dim + self.per_asset_text_dim

    def validate(self) -> None:
        assert self.architecture_version == "v4.0", "spec is pinned to v4.0"
        assert self.data_interval == "1d", "v4 requires daily-aligned training data"
        assert self.pretraining_objective == "tcp", (
            "v4 requires temporal contrastive pretraining"
        )
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_regimes == 4, "spec requires exactly 4 regime classes"
        assert self.n_gdn_blocks == 8, "spec requires 8 RA-GDN blocks"
        assert self.n_fmoe_blocks == 3, "spec requires 3 FMoE blocks"
        assert len(self.fmoe_after_gdn) == self.n_fmoe_blocks
        assert all(1 <= p <= self.n_gdn_blocks for p in self.fmoe_after_gdn)
        assert self.n_swa_layers == 2, "v4 requires 2 SWA layers"
        assert len(self.swa_after_gdn) == self.n_swa_layers
        assert all(1 <= p <= self.n_gdn_blocks for p in self.swa_after_gdn)
        assert self.swa_window == 64, "v4 fixes SWA window at 64"
        assert self.swa_heads > 0 and self.d_model % self.swa_heads == 0
        assert self.top_k_experts <= self.n_experts
        assert self.top_k_experts <= self.n_regimes, (
            "top_k ≤ n_regimes for regime routing"
        )
        assert self.per_asset_continuous_dim == 120
        assert self.per_asset_text_dim == 64
        assert self.per_asset_feature_dim == 184
        assert self.market_feature_dim == 96
        assert self.portfolio_state_dim == 201
        assert self.cascade_levels == 3
        assert len(self.cascade_dims) == self.cascade_levels
        assert len(self.cascade_strides) == self.cascade_levels
        assert self.n_event_types == 18
        assert self.regime_emb_dim == 64
        assert self.mr_vocab_size == 4_096
        assert self.regime_detector_input_dim == (
            self.market_feature_dim + 3 + self.regime_emb_dim
        )
        assert self.pretrain_steps == 20_000
        assert self.train_batch_size == 128
        assert self.learning_rate == 3e-4
        assert self.pretrain_min_lr == 1e-5
        assert self.tcp_temperature == 0.1
        assert self.tcp_lambda_contrastive == 0.7
        assert self.tcp_lambda_prediction == 0.3
        assert self.pretrain_fda_loss_weight == 0.5
        assert self.pretrain_positive_offset == 4


DEFAULT_SPEC = MeridianSpec()
