"""Meridian training project setup package."""

from .train.feature_schema import (
    CALENDAR_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    PER_ASSET_FEATURE_DIM,
    PORTFOLIO_STATE_DIM,
    validate_feature_schema,
)
from .train.spec import DEFAULT_ARCHITECTURE, TRAINING_PHASES
from .data.dataset_ingestion import ingest_feature_jsonl, ingest_feature_rows
from .data.environment import build_environment_report, validate_phase0_environment
from .data.feature_assembly import AssetMinuteInput, FeatureAssembler
from .data.finance_query_collector import (
    FinanceQueryCollectorConfig,
    build_collect_command,
    collect_feature_rows,
    default_rust_collector_command_template,
)
from .data.marketreader import (
    MarketReaderConfig,
    build_marketreader_training_plan,
    load_corpus_texts,
    run_marketreader_training,
    train_bpe_tokenizer,
)
from .data.pipeline import PipelineConfig, build_pipeline_plan, write_pipeline_manifest
from .data.regime_hmm import (
    RegimeHMMConfig,
    build_hmm_training_spec,
    load_regime_feature_rows,
    run_hmm_training_pipeline,
)
from .core.regime_aware_backbone import (
    MeridianPhase1Backbone,
    Phase1CoreConfig,
    RAGDNBlock,
    RegimeRoutedFMoEBlock,
    RegimeDetector,
    run_phase1_backbone_smoke_check,
    run_phase1_smoke_check,
)
from .core.self_supervised_pretraining import (
    Phase1TrainingConfig,
    build_phase1_training_spec,
    run_phase1_training_smoke,
)

__all__ = [
    "AssetMinuteInput",
    "FeatureAssembler",
    "FinanceQueryCollectorConfig",
    "MarketReaderConfig",
    "PipelineConfig",
    "Phase1CoreConfig",
    "Phase1TrainingConfig",
    "MeridianPhase1Backbone",
    "RAGDNBlock",
    "RegimeHMMConfig",
    "RegimeDetector",
    "RegimeRoutedFMoEBlock",
    "CALENDAR_FEATURE_DIM",
    "DEFAULT_ARCHITECTURE",
    "MARKET_FEATURE_DIM",
    "PER_ASSET_FEATURE_DIM",
    "PORTFOLIO_STATE_DIM",
    "TRAINING_PHASES",
    "build_environment_report",
    "build_collect_command",
    "build_hmm_training_spec",
    "build_marketreader_training_plan",
    "build_phase1_training_spec",
    "build_pipeline_plan",
    "ingest_feature_jsonl",
    "ingest_feature_rows",
    "collect_feature_rows",
    "default_rust_collector_command_template",
    "load_corpus_texts",
    "load_regime_feature_rows",
    "run_marketreader_training",
    "run_hmm_training_pipeline",
    "run_phase1_backbone_smoke_check",
    "run_phase1_smoke_check",
    "run_phase1_training_smoke",
    "train_bpe_tokenizer",
    "validate_phase0_environment",
    "validate_feature_schema",
    "write_pipeline_manifest",
]
