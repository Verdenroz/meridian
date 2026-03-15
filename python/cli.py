"""Meridian CLI  —  meridian <group> <subcommand> [options]

Groups:
  check   Validate setup contracts, environment, and feature dimensions
  data    Workspace init, data planning, ingestion
  train   Component training (MarketReader, HMM) and training-schedule plans
  smoke   Forward-pass and training smoke tests for model modules
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from datetime import date as date_type
from pathlib import Path

from train.feature_schema import (
    CALENDAR_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    PER_ASSET_BLOCKS,
    PER_ASSET_FEATURE_DIM,
    PORTFOLIO_STATE_DIM,
    validate_feature_schema,
)
from train.spec import DEFAULT_ARCHITECTURE, TRAINING_STAGES
from train.splits import resolve_split
from market_data.dataset_ingestion import ingest_feature_jsonl, ingest_feature_rows
from market_data.environment import build_environment_report, validate_training_environment
from market_data.feature_assembly import AssetMinuteInput, FeatureAssembler
from market_data.finance_query_collector import FinanceQueryCollectorConfig, collect_feature_rows
from market_data.marketreader import MarketReaderConfig, build_marketreader_training_plan, run_marketreader_training
from market_data.pipeline import PipelineConfig, write_pipeline_manifest
from market_data.regime_hmm import RegimeHMMConfig, build_hmm_training_spec, run_hmm_training_pipeline
from market_data.workspace import initialize_workspace
from core.regime_aware_backbone import run_core_block_smoke_check, run_stacked_backbone_smoke_check
from core.temporal_cascade import run_cascade_smoke_check
from core.cross_asset_fusion import run_fusion_smoke_check
from core.self_supervised_pretraining import BackboneTrainingConfig, run_backbone_pretraining_smoke, run_split_shard_evaluation


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meridian",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    groups = parser.add_subparsers(dest="group", required=True)

    _add_check_group(groups)
    _add_data_group(groups)
    _add_train_group(groups)
    _add_smoke_group(groups)

    return parser


def _add_check_group(groups: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = groups.add_parser("check", help="Validate setup, environment, and feature contracts")
    sub = p.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("spec", help="Validate feature schema and architecture contracts")

    env = sub.add_parser("env", help="Validate H100/PyTorch/FLA environment readiness")
    env.add_argument("--allow-non-h100", action="store_true",
                     help="Don't fail if GPU is not H100 (useful for local dry-runs)")

    feat = sub.add_parser("features", help="Assemble a synthetic per-asset feature vector and check its dimension")
    feat.add_argument("--asset-class", choices=["equity", "etf", "crypto"], default="equity")

    split = sub.add_parser("split", help="Resolve a timestamp to its train/validation/test split")
    split.add_argument("--timestamp", required=True, help="ISO-8601 timestamp (e.g. 2024-08-15T13:30:00)")


def _add_data_group(groups: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = groups.add_parser("data", help="Workspace init, pipeline planning, and feature ingestion")
    sub = p.add_subparsers(dest="subcommand", required=True)

    init = sub.add_parser("init", help="Create the standard Meridian training workspace layout")
    init.add_argument("--root", default=".", help="Root directory (default: current directory)")

    plan = sub.add_parser("plan", help="Write historical data pipeline manifest")
    plan.add_argument("--assets", required=True, help="Comma-separated assets, e.g. AAPL,SPY,BTC-USD")
    plan.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    plan.add_argument("--end",   default="2025-06-30", help="End date (YYYY-MM-DD)")
    plan.add_argument("--output", default="reports/data_pipeline_manifest.json")

    ingest = sub.add_parser("ingest", help="Ingest feature JSONL into memory-mapped split shards")
    ingest.add_argument("--input",       required=True, help="Path to JSONL feature rows")
    ingest.add_argument("--asset",       required=True, help="Asset symbol (e.g. AAPL)")
    ingest.add_argument("--output-root", default=".", help="Output root for data/mmaps and reports/ingestion")

    collect = sub.add_parser("collect", help="Collect features via finance-query and ingest to split shards")
    collect.add_argument("--asset",       required=True, help="Asset symbol (e.g. AAPL)")
    collect.add_argument("--start",       default="2020-01-01", help="Start date (YYYY-MM-DD)")
    collect.add_argument("--end",         default="2025-06-30", help="End date (YYYY-MM-DD)")
    collect.add_argument("--output-root", default=".")
    collect.add_argument(
        "--command-template", default=None,
        help="Shell command template with {symbol}/{start}/{end} placeholders. "
             "Defaults to the local Rust finance-query collector (src/main.rs).",
    )


def _add_train_group(groups: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = groups.add_parser("train", help="Train components (MarketReader, HMM) or print training-schedule plans")
    sub = p.add_subparsers(dest="subcommand", required=True)

    plan = sub.add_parser("plan", help="Print training-stage hyperparameter defaults as JSON")
    plan.add_argument("--stage", choices=sorted(TRAINING_STAGES.keys()), required=True)

    mr = sub.add_parser("marketreader", help="Train (or preview) the MarketReader text encoder")
    mr.add_argument("--corpus",           default="data/raw/news_corpus.jsonl")
    mr.add_argument("--tokenizer-output", default="artifacts/marketreader_tokenizer.json")
    mr.add_argument("--model-output",     default="checkpoints/marketreader.pt")
    mr.add_argument("--metrics-output",   default=None)
    mr.add_argument("--max-documents",    type=int, default=None)
    mr.add_argument("--epochs",           type=int, default=1)
    mr.add_argument("--batch-size",       type=int, default=16)
    mr.add_argument("--learning-rate",    type=float, default=3e-4)
    mr.add_argument("--device",           default=None)
    mr.add_argument("--plan", action="store_true",
                    help="Print training plan JSON and exit without training")

    hmm = sub.add_parser("hmm", help="Train (or preview) the 4-state regime HMM")
    hmm.add_argument("--input",           required=False, default=None,
                     help="Path to JSONL feature rows (required unless --plan)")
    hmm.add_argument("--features",        required=True,
                     help="Comma-separated feature names")
    hmm.add_argument("--model-output",    default="checkpoints/regime_hmm.pkl")
    hmm.add_argument("--meta-output",     default=None)
    hmm.add_argument("--max-rows",        type=int, default=None)
    hmm.add_argument("--covariance-type", choices=["diag", "full"], default="diag")
    hmm.add_argument("--n-iter",          type=int, default=200)
    hmm.add_argument("--random-state",    type=int, default=42)
    hmm.add_argument("--plan", action="store_true",
                     help="Print training spec JSON and exit without training")


def _add_smoke_group(groups: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = groups.add_parser("smoke", help="Forward-pass and training smoke tests for model modules")
    sub = p.add_subparsers(dest="subcommand", required=True)

    def _batch_seq_device(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--batch-size", type=int, default=2)
        sp.add_argument("--seq-len",    type=int, default=16)
        sp.add_argument("--device",     default=None)

    _batch_seq_device(sub.add_parser("core",     help="RA-GDN block + regime detector forward pass"))
    _batch_seq_device(sub.add_parser("backbone", help="Stacked backbone (RA-GDN × 8 + FMoE × 3) forward pass"))
    _batch_seq_device(sub.add_parser("cascade",  help="Temporal cascade forward pass"))

    fusion = sub.add_parser("fusion", help="Cross-asset fusion forward pass")
    fusion.add_argument("--n-assets",   type=int, default=10)
    fusion.add_argument("--n-ctx",      type=int, default=3, help="Context tokens (market+portfolio+episodic)")
    fusion.add_argument("--batch-size", type=int, default=2)
    fusion.add_argument("--device",     default=None)

    train_smoke = sub.add_parser("train", help="Short backbone self-supervised training loop with checkpoint save")
    train_smoke.add_argument("--steps",                type=int,   default=5)
    train_smoke.add_argument("--batch-size",           type=int,   default=2)
    train_smoke.add_argument("--seq-len",              type=int,   default=16)
    train_smoke.add_argument("--learning-rate",        type=float, default=3e-4)
    train_smoke.add_argument("--prediction-ramp-steps",type=int,   default=0)
    train_smoke.add_argument("--checkpoint-output",    default="checkpoints/backbone_pretrain.pt")
    train_smoke.add_argument("--metrics-output",       default=None)
    train_smoke.add_argument("--device",               default=None)
    train_smoke.add_argument("--input-shard",          default=None, help=".f32bin shard for real-data batches")
    train_smoke.add_argument("--input-shards",         default=None, help="Comma-separated .f32bin shard paths")
    train_smoke.add_argument("--max-sequences",        type=int, default=None)
    train_smoke.add_argument("--resume-checkpoint",    default=None)
    train_smoke.add_argument("--eval-only", action="store_true",
                             help="Evaluate only — no optimizer updates, no checkpoint write")

    eval_smoke = sub.add_parser("eval", help="Split-aware backbone evaluation over shard groups")
    eval_smoke.add_argument("--shards-root",          default="data/mmaps")
    eval_smoke.add_argument("--splits",               default="train,validation,test")
    eval_smoke.add_argument("--steps",                type=int,   default=5)
    eval_smoke.add_argument("--batch-size",           type=int,   default=2)
    eval_smoke.add_argument("--seq-len",              type=int,   default=16)
    eval_smoke.add_argument("--learning-rate",        type=float, default=3e-4)
    eval_smoke.add_argument("--prediction-ramp-steps",type=int,   default=0)
    eval_smoke.add_argument("--checkpoint-output",    default="checkpoints/backbone_eval_placeholder.pt")
    eval_smoke.add_argument("--resume-checkpoint",    default=None)
    eval_smoke.add_argument("--max-sequences",        type=int, default=None)
    eval_smoke.add_argument("--max-shards-per-split", type=int, default=None)
    eval_smoke.add_argument("--device",               default=None)
    eval_smoke.add_argument("--include-per-shard",    action="store_true")
    eval_smoke.add_argument("--report-output",        default=None)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _run_check(args: argparse.Namespace) -> int:
    match args.subcommand:
        case "spec":
            validate_feature_schema()
            DEFAULT_ARCHITECTURE.validate()
            print(json.dumps({
                "per_asset_feature_dim": PER_ASSET_FEATURE_DIM,
                "market_feature_dim": MARKET_FEATURE_DIM,
                "portfolio_state_dim": PORTFOLIO_STATE_DIM,
                "calendar_feature_dim": CALENDAR_FEATURE_DIM,
                "feature_blocks": {b.name: b.dim for b in PER_ASSET_BLOCKS},
                "architecture": DEFAULT_ARCHITECTURE.to_dict(),
            }, indent=2, sort_keys=True))
        case "env":
            try:
                report = build_environment_report()
                print(json.dumps(report, indent=2, sort_keys=True))
                validate_training_environment(require_h100=not args.allow_non_h100)
            except RuntimeError as e:
                print(str(e), file=sys.stderr)
                return 1
        case "features":
            assembler = FeatureAssembler()
            sample = AssetMinuteInput(
                price_volume=[0.0] * 10, technical=[0.0] * 20, options=None,
                fundamental=[0.0] * 10, text=[0.0] * 64, asset_calendar=[0.0] * 3,
                asset_class=args.asset_class,
            )
            vector = assembler.assemble_asset(sample)
            print(json.dumps({"asset_class": args.asset_class, "feature_dim": len(vector)}, indent=2))
        case "split":
            print(resolve_split(datetime.fromisoformat(args.timestamp)))
    return 0


def _run_data(args: argparse.Namespace) -> int:
    match args.subcommand:
        case "init":
            created = initialize_workspace(Path(args.root).resolve())
            print(json.dumps({"created": [str(p) for p in created]}, indent=2))
        case "plan":
            asset_list = [a.strip() for a in args.assets.split(",") if a.strip()]
            config = PipelineConfig(
                start_date=date_type.fromisoformat(args.start),
                end_date=date_type.fromisoformat(args.end),
                assets=asset_list,
                output_root=Path("."),
            )
            out = write_pipeline_manifest(config, Path(args.output))
            print(json.dumps({"manifest": str(out.resolve())}, indent=2))
        case "ingest":
            manifest = ingest_feature_jsonl(
                input_path=Path(args.input),
                asset_symbol=args.asset,
                output_root=Path(args.output_root),
            )
            print(json.dumps({"manifest": manifest["manifest_path"], "split_counts": manifest["split_counts"]}, indent=2))
        case "collect":
            config = FinanceQueryCollectorConfig(
                command_template=tuple(shlex.split(args.command_template))
                if args.command_template is not None
                else FinanceQueryCollectorConfig().command_template
            )
            rows = collect_feature_rows(
                symbol=args.asset,
                start_date=date_type.fromisoformat(args.start),
                end_date=date_type.fromisoformat(args.end),
                config=config,
            )
            manifest = ingest_feature_rows(
                rows=rows,
                asset_symbol=args.asset,
                output_root=Path(args.output_root),
                input_descriptor=f"finance-query:{args.asset}:{args.start}:{args.end}",
            )
            print(json.dumps({"manifest": manifest["manifest_path"], "split_counts": manifest["split_counts"]}, indent=2))
    return 0


def _run_train(args: argparse.Namespace) -> int:
    match args.subcommand:
        case "plan":
            print(json.dumps(TRAINING_STAGES[args.stage].to_dict(), indent=2, sort_keys=True))
        case "marketreader":
            config = MarketReaderConfig(
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            if args.plan:
                plan = build_marketreader_training_plan(
                    config=config,
                    corpus_path=Path(args.corpus),
                    tokenizer_output_path=Path(args.tokenizer_output),
                    model_output_path=Path(args.model_output),
                )
                print(json.dumps(plan, indent=2, sort_keys=True))
            else:
                try:
                    metrics = run_marketreader_training(
                        config=config,
                        corpus_path=Path(args.corpus),
                        tokenizer_output_path=Path(args.tokenizer_output),
                        model_output_path=Path(args.model_output),
                        metrics_output_path=Path(args.metrics_output) if args.metrics_output else None,
                        max_documents=args.max_documents,
                        device=args.device,
                    )
                    print(json.dumps(metrics, indent=2, sort_keys=True))
                except (RuntimeError, ValueError, FileNotFoundError) as e:
                    print(str(e), file=sys.stderr)
                    return 1
        case "hmm":
            feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
            config = RegimeHMMConfig(
                covariance_type=args.covariance_type,
                n_iter=args.n_iter,
                random_state=args.random_state,
            )
            if args.plan:
                print(json.dumps(build_hmm_training_spec(config, feature_names=feature_names), indent=2, sort_keys=True))
            else:
                if not args.input:
                    print("error: --input is required for hmm training (use --plan to preview spec)", file=sys.stderr)
                    return 1
                try:
                    summary = run_hmm_training_pipeline(
                        config=config,
                        input_path=Path(args.input),
                        feature_names=feature_names,
                        model_output_path=Path(args.model_output),
                        meta_output_path=Path(args.meta_output) if args.meta_output else None,
                        max_rows=args.max_rows,
                    )
                    print(json.dumps(summary, indent=2, sort_keys=True))
                except (RuntimeError, ValueError, FileNotFoundError) as e:
                    print(str(e), file=sys.stderr)
                    return 1
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    try:
        match args.subcommand:
            case "core":
                result = run_core_block_smoke_check(args.batch_size, args.seq_len, device=args.device)
            case "backbone":
                result = run_stacked_backbone_smoke_check(args.batch_size, args.seq_len, device=args.device)
            case "cascade":
                result = run_cascade_smoke_check(args.batch_size, args.seq_len, device=args.device)
            case "fusion":
                result = run_fusion_smoke_check(args.n_assets, args.n_ctx, args.batch_size, device=args.device)
            case "train":
                if args.input_shard is not None and args.input_shards is not None:
                    print("error: use --input-shard or --input-shards, not both", file=sys.stderr)
                    return 1
                shard_paths = (
                    [Path(s.strip()) for s in args.input_shards.split(",") if s.strip()]
                    if args.input_shards else None
                )
                config = BackboneTrainingConfig(
                    steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
                    learning_rate=args.learning_rate, prediction_ramp_steps=args.prediction_ramp_steps,
                )
                result = run_backbone_pretraining_smoke(
                    config=config,
                    checkpoint_output_path=Path(args.checkpoint_output),
                    metrics_output_path=Path(args.metrics_output) if args.metrics_output else None,
                    device=args.device,
                    input_shard_path=Path(args.input_shard) if args.input_shard else None,
                    input_shard_paths=shard_paths,
                    max_sequences=args.max_sequences,
                    resume_from_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
                    eval_only=args.eval_only,
                )
            case "eval":
                split_values = [s.strip() for s in args.splits.split(",") if s.strip()]
                if not split_values:
                    print("error: at least one split required", file=sys.stderr)
                    return 1
                config = BackboneTrainingConfig(
                    steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
                    learning_rate=args.learning_rate, prediction_ramp_steps=args.prediction_ramp_steps,
                )
                result = run_split_shard_evaluation(
                    config=config,
                    shards_root=Path(args.shards_root),
                    splits=split_values,
                    checkpoint_output_path=Path(args.checkpoint_output),
                    resume_from_checkpoint_path=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
                    max_sequences=args.max_sequences,
                    max_shards_per_split=args.max_shards_per_split,
                    device=args.device,
                    include_per_shard=args.include_per_shard,
                )
                if args.report_output is not None:
                    out = Path(args.report_output)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                    result["report_output_path"] = str(out)
            case _:
                return 2
        print(json.dumps(result, indent=2, sort_keys=True))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv[1:] if argv is not None else None)

    match args.group:
        case "check": return _run_check(args)
        case "data":  return _run_data(args)
        case "train": return _run_train(args)
        case "smoke": return _run_smoke(args)
        case _:
            parser.error(f"Unknown group: {args.group}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
