"""Self-supervised pre-training entry point.

Trains the hybrid RA-GDN + SWA backbone + cross-asset fusion using the
temporal contrastive pretraining (TCP) objective on split-aware f32bin
shards produced by the data pipeline.

Usage (production — H100):

    python -m train.pretrain \\
        --data-dir  data/mmaps_aug \\
        --steps     20000 \\
        --batch-size 128 \\
        --seq-len   64 \\
        --lr        3e-4 \\
        --workers   4 \\
        --checkpoint-dir checkpoints/pretrain

Usage (smoke test — CPU, synthetic data):

    python -m train.pretrain --smoke --steps 5 --batch-size 2 --seq-len 8

Resume from checkpoint:

    python -m train.pretrain \\
        --data-dir  data/mmaps_aug \\
        --resume-from checkpoints/pretrain/backbone_pretrain.pt \\
        --steps     20000
"""

from __future__ import annotations

import argparse
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as _torch_mp

# DataLoader workers ship batches to the main process via shared-memory
# tensors. Default sharing strategy on Linux is 'file_descriptor', which
# allocates one FD per tensor per inflight sample — with 8 tensors per
# sample, batch=32, prefetch=4, workers=4, that's ~4096 FDs in flight,
# easily blowing past the default ulimit -n of 1024. Symptom: training
# crashes mid-run with "OSError: [Errno 24] Too many open files" inside
# the worker collate. 'file_system' uses tmpfs entries instead of FDs and
# scales without a per-tensor FD cost.
_torch_mp.set_sharing_strategy("file_system")
from core.cross_asset_fusion import CrossAssetFusion
from core.regime_aware_backbone import InputProjection, RAGDNBackbone, RegimeDetector
from core.self_supervised_pretraining import (
    SelfSupervisedPretrainer,
    TrainConfig,
    build_dataloader,
    build_split_loaders,
)

from train.spec import DEFAULT_SPEC, MeridianSpec

_SECTOR_HEADER_RE = re.compile(r"([A-Za-z][A-Za-z\s&/\-]+)\s*\(\d+\)")
_GICS_SECTOR_NAMES = (
    "Information Technology",
    "Financials",
    "Health Care",
    "Industrials",
    "Consumer Discretionary",
    "Communication Services",
    "Consumer Staples",
    "Energy",
    "Materials",
    "Real Estate",
    "Utilities",
)
_SECTOR_NAME_TO_ID = {name: idx for idx, name in enumerate(_GICS_SECTOR_NAMES)}

# ── Model factory ─────────────────────────────────────────────────────────────


def build_model(
    spec: MeridianSpec,
    device: torch.device,
) -> tuple[RAGDNBackbone, CrossAssetFusion, RegimeDetector, InputProjection]:
    """Instantiate all model components from spec."""
    backbone = RAGDNBackbone(
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        d_head=spec.d_head,
        d_v=spec.d_v,
        n_gdn_blocks=spec.n_gdn_blocks,
        n_experts=spec.n_experts,
        top_k_experts=spec.top_k_experts,
        expert_ffn_dim=spec.expert_ffn_dim,
        fmoe_after_gdn=spec.fmoe_after_gdn,
        swa_after_gdn=spec.swa_after_gdn,
        swa_window=spec.swa_window,
        swa_heads=spec.swa_heads,
        regime_emb_dim=spec.regime_emb_dim,
    ).to(device)

    fusion = CrossAssetFusion(
        d_model=spec.d_model,
        n_layers=spec.n_fusion_layers,
        n_heads=spec.n_fusion_heads,
    ).to(device)

    regime_detector = RegimeDetector(
        input_dim=spec.regime_detector_input_dim,
        regime_emb_dim=spec.regime_emb_dim,
        n_regimes=spec.n_regimes,
    ).to(device)

    input_projection = InputProjection(
        d_model=spec.d_model,
        per_asset_dim=spec.per_asset_feature_dim,
        market_dim=spec.market_feature_dim,
        portfolio_dim=spec.portfolio_state_dim,
        calendar_dim=spec.calendar_dim,
        regime_emb_dim=spec.regime_emb_dim,
    ).to(device)

    return backbone, fusion, regime_detector, input_projection


def count_parameters(modules: list[torch.nn.Module]) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def load_symbol_sector_map(symbols_file: Path) -> dict[str, int]:
    """Parse symbols.txt section headers into symbol→sector-id mapping."""
    if not symbols_file.exists():
        raise FileNotFoundError(f"symbols file not found: {symbols_file}")

    symbol_to_sector: dict[str, int] = {}
    current_sector: int | None = None
    for raw_line in symbols_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _SECTOR_HEADER_RE.search(line)
            if m is not None:
                section_name = " ".join(m.group(1).split())
                current_sector = _SECTOR_NAME_TO_ID.get(section_name)
            continue

        symbol = line.split("#", 1)[0].strip()
        if symbol and current_sector is not None:
            symbol_to_sector[symbol] = current_sector

    return symbol_to_sector


# ── Synthetic shard helpers ───────────────────────────────────────────────────


def _write_synthetic_shard(path: Path, n_rows: int, feature_dim: int = 280) -> None:
    """Write a shard of random float32 rows for smoke testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row_struct = struct.Struct(f"<{feature_dim}f")
    rng = np.random.default_rng(seed=42)
    with open(path, "wb") as f:
        for _ in range(n_rows):
            row = rng.standard_normal(feature_dim).astype(np.float32)
            f.write(row_struct.pack(*row))


def _make_smoke_shards(tmp_dir: Path, seq_len: int) -> tuple[Path, Path]:
    """Create minimal synthetic shards (train + val) for smoke testing."""
    n_rows = seq_len + 10  # enough rows for a few sequences
    for split in ("train", "validation"):
        shard_path = tmp_dir / split / "SMOKE.f32bin"
        _write_synthetic_shard(shard_path, n_rows)
    return tmp_dir / "train", tmp_dir / "validation"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-supervised backbone pre-training"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps_aug"))
    parser.add_argument("--symbols-file", type=Path, default=Path("symbols.txt"))
    parser.add_argument("--steps", type=int, default=DEFAULT_SPEC.pretrain_steps)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SPEC.train_batch_size)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SPEC.train_seq_len)
    parser.add_argument("--lr", type=float, default=DEFAULT_SPEC.learning_rate)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_SPEC.pretrain_min_lr)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--eval-max-batches", type=int, default=25)
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_SPEC.tcp_temperature
    )
    parser.add_argument(
        "--lambda-contrastive", type=float, default=DEFAULT_SPEC.tcp_lambda_contrastive
    )
    parser.add_argument(
        "--lambda-prediction", type=float, default=DEFAULT_SPEC.tcp_lambda_prediction
    )
    parser.add_argument(
        "--fda-loss-weight", type=float, default=DEFAULT_SPEC.pretrain_fda_loss_weight
    )
    parser.add_argument(
        "--positive-offset", type=int, default=DEFAULT_SPEC.pretrain_positive_offset
    )
    parser.add_argument("--multi-asset-corr-threshold", type=float, default=0.6)
    parser.add_argument("--negative-min-separation", type=int, default=50)
    parser.add_argument("--early-stop-patience-evals", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/pretrain")
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the backbone for faster steady-state steps",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable activation checkpointing inside each backbone call (~33%% "
        "extra compute, reduces intra-step VRAM)",
    )
    parser.add_argument(
        "--bptt-horizon",
        type=int,
        default=0,
        help="Truncate BPTT to this many steps (0 = full sequence). "
        "Reduces inter-step state-chain VRAM from O(T) to O(K). "
        "32 fits comfortably in 24 GB with T=64 B=128.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: use synthetic data, 2-step run on CPU",
    )
    args = parser.parse_args()

    # ── Override for smoke test ──
    if args.smoke:
        args.steps = args.steps if args.steps < DEFAULT_SPEC.pretrain_steps else 3
        args.batch_size = 2
        args.seq_len = max(args.seq_len, 8)
        args.log_every = 1
        args.eval_every = 2
        args.save_every = 10_000
        args.warmup_steps = 0
        args.workers = 0  # no multiprocessing in smoke mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # TF32 for the fp32 matmul paths outside bf16 autocast (norms, residuals,
        # some attention math). Free ~10–30% on Ampere/Hopper, no quality impact.
        torch.set_float32_matmul_precision("high")
    spec = DEFAULT_SPEC
    spec.validate()

    print(f"Device:      {device}")
    print(f"Steps:       {args.steps}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Seq len:     {args.seq_len}")

    # ── Build model ──
    backbone, fusion, regime_detector, input_projection = build_model(spec, device)
    n_params = count_parameters([backbone, fusion, regime_detector, input_projection])
    print(f"Parameters:  {n_params:,}  ({n_params / 1e6:.1f}M)")

    if args.gradient_checkpointing:
        print("Gradient checkpointing: enabled")
    if args.bptt_horizon > 0:
        print(f"BPTT horizon:          {args.bptt_horizon} steps")

    pretrainer = SelfSupervisedPretrainer(
        backbone=backbone,
        fusion=fusion,
        regime_detector=regime_detector,
        input_projection=input_projection,
        d_model=spec.d_model,
        n_assets=spec.n_assets,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
        bptt_horizon=args.bptt_horizon,
    )

    # ── Build data loaders ──
    if args.smoke:
        with tempfile.TemporaryDirectory() as td:
            _run_with_smoke_data(pretrainer, args, Path(td))
    else:
        symbol_to_sector = load_symbol_sector_map(args.symbols_file)
        print(
            f"Loaded sector IDs for {len(symbol_to_sector)} equity symbols "
            f"(ETFs get sector_id=-1 and are excluded from cross-asset positive pairs)"
        )
        train_loader, val_loader = build_split_loaders(
            args.data_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.workers,
            positive_offset=args.positive_offset,
            synthetic_loss_weight=args.fda_loss_weight,
            symbol_to_sector=symbol_to_sector,
        )
        cfg = TrainConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_every=args.save_every,
            eval_max_batches=args.eval_max_batches,
            contrastive_temperature=args.temperature,
            contrastive_weight=args.lambda_contrastive,
            prediction_weight=args.lambda_prediction,
            positive_pair_offset=args.positive_offset,
            fda_loss_weight=args.fda_loss_weight,
            multi_asset_corr_threshold=args.multi_asset_corr_threshold,
            negative_min_separation=args.negative_min_separation,
            early_stop_patience_evals=args.early_stop_patience_evals,
            early_stop_min_delta=args.early_stop_min_delta,
            checkpoint_dir=args.checkpoint_dir,
            compile=args.compile,
        )
        pretrainer.train(
            train_loader=train_loader,
            cfg=cfg,
            val_loader=val_loader,
            resume_from=args.resume_from,
        )


def _run_with_smoke_data(
    pretrainer: SelfSupervisedPretrainer,
    args: argparse.Namespace,
    tmp_dir: Path,
) -> None:
    """Run smoke training inside a temporary directory."""
    _make_smoke_shards(tmp_dir, args.seq_len)

    from market_data.dataset_ingestion import enumerate_shards

    train_paths = enumerate_shards(tmp_dir, "train")
    val_paths = enumerate_shards(tmp_dir, "validation")

    all_stems = sorted({p.stem for p in train_paths + val_paths})
    symbol_to_id = {stem: idx for idx, stem in enumerate(all_stems)}

    train_loader = build_dataloader(
        train_paths,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        positive_offset=args.positive_offset,
        synthetic_loss_weight=args.fda_loss_weight,
        symbol_to_id=symbol_to_id,
    )
    val_loader = build_dataloader(
        val_paths,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        positive_offset=args.positive_offset,
        synthetic_loss_weight=args.fda_loss_weight,
        symbol_to_id=symbol_to_id,
    )

    cfg = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        eval_max_batches=args.eval_max_batches,
        contrastive_temperature=args.temperature,
        contrastive_weight=args.lambda_contrastive,
        prediction_weight=args.lambda_prediction,
        positive_pair_offset=args.positive_offset,
        fda_loss_weight=args.fda_loss_weight,
        multi_asset_corr_threshold=args.multi_asset_corr_threshold,
        negative_min_separation=args.negative_min_separation,
        early_stop_patience_evals=args.early_stop_patience_evals,
        early_stop_min_delta=args.early_stop_min_delta,
        checkpoint_dir=tmp_dir / "checkpoints",
        compile=args.compile,
    )

    pretrainer.train(
        train_loader=train_loader,
        cfg=cfg,
        val_loader=val_loader,
        resume_from=args.resume_from,
    )
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
