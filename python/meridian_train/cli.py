from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from datetime import date as date_type
from pathlib import Path

from .train.feature_schema import (
    CALENDAR_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    PER_ASSET_BLOCKS,
    PER_ASSET_FEATURE_DIM,
    PORTFOLIO_STATE_DIM,
    validate_feature_schema,
)
from .train.spec import DEFAULT_ARCHITECTURE, TRAINING_PHASES
from .train.splits import resolve_split
from .data.dataset_ingestion import ingest_feature_jsonl, ingest_feature_rows
from .data.environment import build_environment_report, validate_phase0_environment
from .data.feature_assembly import AssetMinuteInput, FeatureAssembler
from .data.finance_query_collector import (
    FinanceQueryCollectorConfig,
    collect_feature_rows,
)
from .data.marketreader import (
    MarketReaderConfig,
    build_marketreader_training_plan,
    run_marketreader_training,
)
from .data.pipeline import PipelineConfig, write_pipeline_manifest
from .data.regime_hmm import (
    RegimeHMMConfig,
    build_hmm_training_spec,
    run_hmm_training_pipeline,
)
from .data.workspace import initialize_workspace
from .core.regime_aware_backbone import run_phase1_backbone_smoke_check, run_phase1_smoke_check
from .core.self_supervised_pretraining import Phase1TrainingConfig, run_phase1_training_smoke


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meridian")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("spec-check", help="Validate Meridian setup contracts")

    phase_plan = subparsers.add_parser(
        "phase-plan", help="Print training phase defaults as JSON"
    )
    phase_plan.add_argument(
        "--phase", choices=sorted(TRAINING_PHASES.keys()), required=True
    )

    split_check = subparsers.add_parser(
        "split-check", help="Resolve a timestamp into train/validation/test split"
    )
    split_check.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp in ISO-8601 format (e.g. 2024-08-15T13:30:00)",
    )

    init_workspace = subparsers.add_parser(
        "init-workspace", help="Create the standard Meridian training workspace layout"
    )
    init_workspace.add_argument(
        "--root",
        default=".",
        help="Root directory for workspace creation (default: current directory)",
    )

    env_check = subparsers.add_parser(
        "env-check", help="Validate Phase 0 H100/PyTorch/FLA environment readiness"
    )
    env_check.add_argument(
        "--allow-non-h100",
        action="store_true",
        help="Do not fail if GPU is not H100 (useful for local dry-runs)",
    )

    data_plan = subparsers.add_parser(
        "data-plan", help="Write historical data pipeline manifest from spec defaults"
    )
    data_plan.add_argument(
        "--assets",
        required=True,
        help="Comma-separated assets, e.g. AAPL,SPY,BTC-USD",
    )
    data_plan.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    data_plan.add_argument(
        "--end",
        default="2025-06-30",
        help="End date (YYYY-MM-DD)",
    )
    data_plan.add_argument(
        "--output",
        default="reports/data_pipeline_manifest.json",
        help="Output path for generated manifest",
    )

    marketreader_plan = subparsers.add_parser(
        "marketreader-plan", help="Print MarketReader training plan from spec defaults"
    )
    marketreader_plan.add_argument(
        "--corpus",
        default="data/raw/news_corpus.jsonl",
        help="Path to text corpus",
    )
    marketreader_plan.add_argument(
        "--tokenizer-output",
        default="artifacts/marketreader_tokenizer.json",
        help="Tokenizer output path",
    )
    marketreader_plan.add_argument(
        "--model-output",
        default="checkpoints/marketreader.pt",
        help="Model checkpoint output path",
    )

    marketreader_train = subparsers.add_parser(
        "marketreader-train", help="Run MarketReader tokenizer/model training"
    )
    marketreader_train.add_argument(
        "--corpus",
        default="data/raw/news_corpus.jsonl",
        help="Path to JSONL text corpus",
    )
    marketreader_train.add_argument(
        "--tokenizer-output",
        default="artifacts/marketreader_tokenizer.json",
        help="Tokenizer output artifact path",
    )
    marketreader_train.add_argument(
        "--model-output",
        default="checkpoints/marketreader.pt",
        help="Model checkpoint output path",
    )
    marketreader_train.add_argument(
        "--metrics-output",
        default=None,
        help="Optional metrics JSON output path (defaults to model_output with .metrics.json)",
    )
    marketreader_train.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on corpus documents for faster runs",
    )
    marketreader_train.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs",
    )
    marketreader_train.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    marketreader_train.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Training learning rate",
    )
    marketreader_train.add_argument(
        "--device",
        default=None,
        help="Optional explicit torch device (e.g. cuda, cpu)",
    )

    hmm_spec = subparsers.add_parser(
        "hmm-spec", help="Print 4-state regime HMM training spec"
    )
    hmm_spec.add_argument(
        "--features",
        required=True,
        help="Comma-separated feature names for HMM training",
    )

    hmm_train = subparsers.add_parser(
        "hmm-train", help="Run 4-state regime HMM training pipeline"
    )
    hmm_train.add_argument(
        "--input",
        required=True,
        help="Path to JSONL feature rows for HMM fitting",
    )
    hmm_train.add_argument(
        "--features",
        required=True,
        help="Comma-separated feature names expected in input rows",
    )
    hmm_train.add_argument(
        "--model-output",
        default="checkpoints/regime_hmm.pkl",
        help="Output path for pickled HMM model",
    )
    hmm_train.add_argument(
        "--meta-output",
        default=None,
        help="Optional metadata JSON output path",
    )
    hmm_train.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of training rows",
    )
    hmm_train.add_argument(
        "--covariance-type",
        choices=["diag", "full"],
        default="diag",
        help="GaussianHMM covariance type",
    )
    hmm_train.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Maximum EM iterations for HMM fitting",
    )
    hmm_train.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for HMM initialization",
    )

    feature_vector = subparsers.add_parser(
        "feature-vector-check",
        help="Assemble a synthetic per-asset feature vector and validate dimension",
    )
    feature_vector.add_argument(
        "--asset-class",
        choices=["equity", "etf", "crypto"],
        default="equity",
        help="Asset class used for missing options defaults",
    )

    ingest_jsonl = subparsers.add_parser(
        "ingest-jsonl",
        help="Ingest feature JSONL and materialize mmap-compatible split shards",
    )
    ingest_jsonl.add_argument(
        "--input",
        required=True,
        help="Path to JSONL file with minute feature rows",
    )
    ingest_jsonl.add_argument(
        "--asset",
        required=True,
        help="Asset symbol for the shard name (e.g. AAPL)",
    )
    ingest_jsonl.add_argument(
        "--output-root",
        default=".",
        help="Output root directory for data/mmaps and reports/ingestion",
    )

    collect_ingest = subparsers.add_parser(
        "collect-ingest",
        help="Collect features via finance-query and ingest directly to split shards",
    )
    collect_ingest.add_argument(
        "--asset",
        required=True,
        help="Asset symbol (e.g. AAPL)",
    )
    collect_ingest.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    collect_ingest.add_argument(
        "--end",
        default="2025-06-30",
        help="End date (YYYY-MM-DD)",
    )
    collect_ingest.add_argument(
        "--output-root",
        default=".",
        help="Output root directory for data/mmaps and reports/ingestion",
    )
    collect_ingest.add_argument(
        "--command-template",
        default=None,
        help=(
            "Command template string. Placeholders: {symbol}, {start}, {end}. "
            "By default this uses the local Rust collector binary "
            "(src/main.rs) built on the finance-query Rust library. "
            "Override only if needed."
        ),
    )

    phase1_smoke = subparsers.add_parser(
        "phase1-smoke", help="Run synthetic forward-pass check for Phase 1 core modules"
    )
    phase1_smoke.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size",
    )
    phase1_smoke.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Synthetic sequence length",
    )
    phase1_smoke.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (e.g. cuda, cpu)",
    )

    phase1_backbone = subparsers.add_parser(
        "phase1-backbone-smoke",
        help="Run synthetic forward-pass check for stacked Phase 1 backbone (RA-GDN + FMoE)",
    )
    phase1_backbone.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size",
    )
    phase1_backbone.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Synthetic sequence length",
    )
    phase1_backbone.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (e.g. cuda, cpu)",
    )

    phase1_train_smoke = subparsers.add_parser(
        "phase1-train-smoke",
        help="Run short synthetic Phase 1 self-supervised training scaffold and save checkpoint",
    )
    phase1_train_smoke.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of synthetic optimization steps",
    )
    phase1_train_smoke.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size",
    )
    phase1_train_smoke.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Synthetic sequence length",
    )
    phase1_train_smoke.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate",
    )
    phase1_train_smoke.add_argument(
        "--checkpoint-output",
        default="checkpoints/phase1_backbone.pt",
        help="Output checkpoint path",
    )
    phase1_train_smoke.add_argument(
        "--metrics-output",
        default=None,
        help="Optional metrics JSON output path",
    )
    phase1_train_smoke.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (e.g. cuda, cpu)",
    )

    return parser


def _run_spec_check() -> int:
    validate_feature_schema()
    DEFAULT_ARCHITECTURE.validate()

    output = {
        "per_asset_feature_dim": PER_ASSET_FEATURE_DIM,
        "market_feature_dim": MARKET_FEATURE_DIM,
        "portfolio_state_dim": PORTFOLIO_STATE_DIM,
        "calendar_feature_dim": CALENDAR_FEATURE_DIM,
        "feature_blocks": {block.name: block.dim for block in PER_ASSET_BLOCKS},
        "architecture": DEFAULT_ARCHITECTURE.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _run_phase_plan(phase_key: str) -> int:
    phase = TRAINING_PHASES[phase_key]
    print(json.dumps(phase.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_split_check(timestamp: str) -> int:
    parsed_timestamp = datetime.fromisoformat(timestamp)
    split = resolve_split(parsed_timestamp)
    print(split)
    return 0


def _run_init_workspace(root: str) -> int:
    created_paths = initialize_workspace(Path(root).resolve())
    print(json.dumps({"created": [str(path) for path in created_paths]}, indent=2))
    return 0


def _run_env_check(allow_non_h100: bool) -> int:
    report = build_environment_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    validate_phase0_environment(require_h100=not allow_non_h100)
    return 0


def _run_data_plan(assets: str, start: str, end: str, output: str) -> int:
    asset_list = [asset.strip() for asset in assets.split(",") if asset.strip()]
    config = PipelineConfig(
        start_date=date_type.fromisoformat(start),
        end_date=date_type.fromisoformat(end),
        assets=asset_list,
        output_root=Path("."),
    )
    output_path = write_pipeline_manifest(config, Path(output))
    print(json.dumps({"manifest": str(output_path.resolve())}, indent=2))
    return 0


def _run_marketreader_plan(corpus: str, tokenizer_output: str, model_output: str) -> int:
    plan = build_marketreader_training_plan(
        config=MarketReaderConfig(),
        corpus_path=Path(corpus),
        tokenizer_output_path=Path(tokenizer_output),
        model_output_path=Path(model_output),
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def _run_marketreader_train(
    corpus: str,
    tokenizer_output: str,
    model_output: str,
    metrics_output: str | None,
    max_documents: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str | None,
) -> int:
    config = MarketReaderConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )
    metrics = run_marketreader_training(
        config=config,
        corpus_path=Path(corpus),
        tokenizer_output_path=Path(tokenizer_output),
        model_output_path=Path(model_output),
        metrics_output_path=Path(metrics_output) if metrics_output else None,
        max_documents=max_documents,
        device=device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _run_hmm_spec(features: str) -> int:
    feature_names = [feature.strip() for feature in features.split(",") if feature.strip()]
    spec = build_hmm_training_spec(RegimeHMMConfig(), feature_names=feature_names)
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


def _run_hmm_train(
    input_path: str,
    features: str,
    model_output: str,
    meta_output: str | None,
    max_rows: int | None,
    covariance_type: str,
    n_iter: int,
    random_state: int,
) -> int:
    feature_names = [feature.strip() for feature in features.split(",") if feature.strip()]
    config = RegimeHMMConfig(
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    summary = run_hmm_training_pipeline(
        config=config,
        input_path=Path(input_path),
        feature_names=feature_names,
        model_output_path=Path(model_output),
        meta_output_path=Path(meta_output) if meta_output else None,
        max_rows=max_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_feature_vector_check(asset_class: str) -> int:
    assembler = FeatureAssembler()
    sample = AssetMinuteInput(
        price_volume=[0.0] * 10,
        technical=[0.0] * 20,
        options=None,
        fundamental=[0.0] * 10,
        text=[0.0] * 64,
        asset_calendar=[0.0] * 3,
        asset_class=asset_class,
    )
    vector = assembler.assemble_asset(sample)
    print(json.dumps({"asset_class": asset_class, "feature_dim": len(vector)}, indent=2))
    return 0


def _run_ingest_jsonl(input_path: str, asset: str, output_root: str) -> int:
    manifest = ingest_feature_jsonl(
        input_path=Path(input_path),
        asset_symbol=asset,
        output_root=Path(output_root),
    )
    print(json.dumps({"manifest": manifest["manifest_path"], "split_counts": manifest["split_counts"]}, indent=2))
    return 0


def _run_collect_ingest(
    asset: str,
    start: str,
    end: str,
    output_root: str,
    command_template: str | None,
) -> int:
    config = FinanceQueryCollectorConfig(
        command_template=tuple(shlex.split(command_template))
        if command_template is not None
        else FinanceQueryCollectorConfig().command_template
    )
    rows = collect_feature_rows(
        symbol=asset,
        start_date=date_type.fromisoformat(start),
        end_date=date_type.fromisoformat(end),
        config=config,
    )
    manifest = ingest_feature_rows(
        rows=rows,
        asset_symbol=asset,
        output_root=Path(output_root),
        input_descriptor=f"finance-query:{asset}:{start}:{end}",
    )
    print(json.dumps({"manifest": manifest["manifest_path"], "split_counts": manifest["split_counts"]}, indent=2))
    return 0


def _run_phase1_smoke(batch_size: int, seq_len: int, device: str | None) -> int:
    summary = run_phase1_smoke_check(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_phase1_backbone_smoke(batch_size: int, seq_len: int, device: str | None) -> int:
    summary = run_phase1_backbone_smoke_check(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_phase1_train_smoke(
    steps: int,
    batch_size: int,
    seq_len: int,
    learning_rate: float,
    checkpoint_output: str,
    metrics_output: str | None,
    device: str | None,
) -> int:
    config = Phase1TrainingConfig(
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        learning_rate=learning_rate,
    )
    summary = run_phase1_training_smoke(
        config=config,
        checkpoint_output_path=Path(checkpoint_output),
        metrics_output_path=Path(metrics_output) if metrics_output else None,
        device=device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parse_target = argv[1:] if argv is not None else None
    args = parser.parse_args(parse_target)

    if args.command == "spec-check":
        return _run_spec_check()
    if args.command == "phase-plan":
        return _run_phase_plan(args.phase)
    if args.command == "split-check":
        return _run_split_check(args.timestamp)
    if args.command == "init-workspace":
        return _run_init_workspace(args.root)
    if args.command == "env-check":
        try:
            return _run_env_check(args.allow_non_h100)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
    if args.command == "data-plan":
        return _run_data_plan(args.assets, args.start, args.end, args.output)
    if args.command == "marketreader-plan":
        return _run_marketreader_plan(
            args.corpus,
            args.tokenizer_output,
            args.model_output,
        )
    if args.command == "marketreader-train":
        try:
            return _run_marketreader_train(
                args.corpus,
                args.tokenizer_output,
                args.model_output,
                args.metrics_output,
                args.max_documents,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.device,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            print(str(error), file=sys.stderr)
            return 1
    if args.command == "hmm-spec":
        return _run_hmm_spec(args.features)
    if args.command == "hmm-train":
        try:
            return _run_hmm_train(
                args.input,
                args.features,
                args.model_output,
                args.meta_output,
                args.max_rows,
                args.covariance_type,
                args.n_iter,
                args.random_state,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            print(str(error), file=sys.stderr)
            return 1
    if args.command == "feature-vector-check":
        return _run_feature_vector_check(args.asset_class)
    if args.command == "ingest-jsonl":
        return _run_ingest_jsonl(args.input, args.asset, args.output_root)
    if args.command == "collect-ingest":
        return _run_collect_ingest(
            args.asset,
            args.start,
            args.end,
            args.output_root,
            args.command_template,
        )
    if args.command == "phase1-smoke":
        try:
            return _run_phase1_smoke(args.batch_size, args.seq_len, args.device)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
    if args.command == "phase1-backbone-smoke":
        try:
            return _run_phase1_backbone_smoke(
                args.batch_size, args.seq_len, args.device
            )
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
    if args.command == "phase1-train-smoke":
        try:
            return _run_phase1_train_smoke(
                args.steps,
                args.batch_size,
                args.seq_len,
                args.learning_rate,
                args.checkpoint_output,
                args.metrics_output,
                args.device,
            )
        except (RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
