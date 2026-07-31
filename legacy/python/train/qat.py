"""Quantization-Aware Training (QAT) — Section 10.5 of meridian-spec.md.

Strategy: BitNet b1.58 — absmean ternary weights {-1, 0, +1} with FP16 scaling.

  Ternary quantization (per weight matrix W):
    scale = mean(|W|)
    W_ternary = round(W / scale).clamp(-1, 1)

  Straight-through estimator (STE):
    Forward:  use W_ternary for compute
    Backward: pass gradients through as if W_ternary == W (identity STE)

  Precision highway (kept in full precision):
    Output heads, RMSNorm params, regime detector,
    MarketReader, episodic memory projections.

  Schedule:
    - Start from the meta-learning checkpoint
    - ~50K steps on combined pre-training + portfolio data
    - Validate: quantized Sharpe within 5% of full-precision Sharpe on test set

Usage:

    python -m train.qat \\
        --resume-meta checkpoints/meta_learn/meta_learn_best.pt \\
        --data-dir data/mmaps \\
        --steps 50000 \\
        --checkpoint-dir checkpoints/qat

    # Smoke test
    python -m train.qat --smoke --steps 3
"""

from __future__ import annotations

import argparse
import math
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.cross_asset_fusion import CrossAssetFusion
from core.regime_aware_backbone import (
    InputProjection,
    RAGDNBackbone,
    RegimeDetector,
)
from core.self_supervised_pretraining import Checkpoint, load_checkpoint, save_checkpoint
from optimization.losses import return_head_loss
from optimization.portfolio_heads import AuxHeadSurprise, MeridianR, MeridianS
from train.portfolio_train import (
    MultiAssetShardDataset,
    build_portfolio_loader,
)
from train.spec import DEFAULT_SPEC, MeridianSpec


# ── TernaryLinear ────────────────────────────────────────────────────────────


class TernaryLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weight quantization.

    Forward pass uses ternary weights {-1, 0, +1} scaled by mean(|W|).
    Backward pass uses the straight-through estimator (STE): gradients flow
    through the quantization as if it were the identity function.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @staticmethod
    def _quantize_ste(w: torch.Tensor) -> torch.Tensor:
        """Ternary quantization with STE.

        Forward: w_q = round(w / scale).clamp(-1, 1) * scale
        Backward: dL/dw passes through as if w_q == w
        """
        scale = w.abs().mean().clamp(min=1e-8)
        w_t = (w / scale).round().clamp(-1, 1)
        return w + (w_t * scale - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._quantize_ste(self.weight)
        return F.linear(x, w_q, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> TernaryLinear:
        """Create a TernaryLinear by copying weights from an nn.Linear."""
        has_bias = linear.bias is not None
        ternary = cls(linear.in_features, linear.out_features, bias=has_bias)
        ternary.weight.data.copy_(linear.weight.data)
        if has_bias:
            ternary.bias.data.copy_(linear.bias.data)
        return ternary.to(linear.weight.device)


# ── Precision highway ────────────────────────────────────────────────────────

# Module name patterns that stay in full precision (not quantized).
PRECISION_HIGHWAY = (
    "head_r",
    "head_s",
    "aux_head",
    "regime_detector",
    "norm",           # RMSNorm layers — Tier 2 adaptation targets
    "episodic",
    "market_reader",
)


def _should_skip(full_name: str) -> bool:
    return any(p in full_name for p in PRECISION_HIGHWAY)


def replace_linear_with_ternary(
    model: nn.Module,
    skip_patterns: tuple[str, ...] = PRECISION_HIGHWAY,
) -> int:
    """Walk the module tree and replace eligible nn.Linear with TernaryLinear.

    Returns the number of modules replaced.
    """
    replacements: list[tuple[nn.Module, str, TernaryLinear]] = []

    for name, module in model.named_modules():
        if _should_skip(name):
            continue
        for attr_name, child in list(module.named_children()):
            full = f"{name}.{attr_name}" if name else attr_name
            if isinstance(child, nn.Linear) and not _should_skip(full):
                replacements.append((module, attr_name, TernaryLinear.from_linear(child)))

    for parent, attr_name, ternary_mod in replacements:
        setattr(parent, attr_name, ternary_mod)

    return len(replacements)


# ── Training config ──────────────────────────────────────────────────────────


@dataclass
class QATConfig:
    total_steps: int = 50_000
    batch_size: int = 4
    seq_len: int = 64
    lr: float = 1e-5             # very low LR for QAT fine-tuning
    warmup_steps: int = 500
    grad_clip: float = 0.5
    lambda_turn: float = 0.1
    lambda_vol: float = 0.05
    dsr_eta: float = 0.01
    sharpe_threshold: float = 0.05  # max allowed relative degradation
    log_every: int = 100
    eval_every: int = 1_000
    save_every: int = 5_000
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/qat"))

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "qat_best.pt"

    @property
    def step_path_fn(self):
        return lambda step: self.checkpoint_dir / f"qat_step{step}.pt"


# ── QAT Trainer ──────────────────────────────────────────────────────────────


class QATTrainer:
    """Fine-tunes the model with ternary weight quantization."""

    def __init__(
        self,
        backbone: RAGDNBackbone,
        fusion: CrossAssetFusion,
        regime_detector: RegimeDetector,
        input_projection: InputProjection,
        head_r: MeridianR,
        head_s: MeridianS,
        aux_head: AuxHeadSurprise,
        spec: MeridianSpec,
        device: torch.device,
    ) -> None:
        self.device = device
        self.spec = spec
        self.backbone = backbone.to(device)
        self.fusion = fusion.to(device)
        self.regime_detector = regime_detector.to(device)
        self.input_projection = input_projection.to(device)
        self.head_r = head_r.to(device)
        self.head_s = head_s.to(device)
        self.aux_head = aux_head.to(device)

    def quantize(self) -> int:
        """Replace eligible Linear modules with TernaryLinear. Returns count."""
        count = 0
        count += replace_linear_with_ternary(self.backbone)
        count += replace_linear_with_ternary(self.fusion)
        count += replace_linear_with_ternary(self.input_projection)
        return count

    def _cosine_lr(self, step: int, cfg: QATConfig) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _forward(
        self,
        features: torch.Tensor,  # (B, N, T, F=280)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — identical to PortfolioTrainer._forward for return head."""
        features = features.to(self.device)
        B, N, T, Fd = features.shape

        asset_feats = features[..., :184]
        mkt_mean = features[..., 184:].mean(dim=1)

        regime_emb = torch.zeros(B, self.spec.regime_emb_dim, device=self.device)
        posteriors = torch.full((B, self.spec.n_regimes), 0.25, device=self.device)
        portfolio_s = torch.zeros(B, self.spec.portfolio_state_dim, device=self.device)
        calendar = torch.zeros(B, self.spec.calendar_dim, device=self.device)
        sector_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        episodic = torch.zeros(
            B, self.spec.n_episodic_retrievals, self.spec.d_model, device=self.device
        )

        states = self.backbone.init_state(B, N, self.device)
        all_weights: list[torch.Tensor] = []

        for t in range(T):
            af_t = asset_feats[:, :, t, :]
            mf_t = mkt_mean[:, t, :]
            asset_tok, mkt_tok, port_tok, _ = self.input_projection(
                af_t, mf_t, portfolio_s, calendar, regime_emb
            )
            out, states = self.backbone(asset_tok, regime_emb, posteriors, states)
            fused, mkt_out = self.fusion(out, mkt_tok, port_tok, episodic, sector_ids)
            w_t = self.head_r(fused)
            all_weights.append(w_t)

            n_full = self.spec.n_assets
            w_padded = F.pad(w_t.detach(), (0, n_full - N))
            portfolio_s = torch.cat([
                w_padded,
                torch.zeros(B, n_full, device=self.device),
                torch.zeros(B, 1, device=self.device),
            ], dim=-1)

        weights = torch.stack(all_weights, dim=1)
        asset_returns = asset_feats[:, :, :, 3].permute(0, 2, 1)
        return weights, asset_returns

    def _all_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.aux_head):
            params.extend(p for p in m.parameters() if p.requires_grad)
        return params

    def evaluate(
        self,
        val_loader: DataLoader,
        cfg: QATConfig,
        max_batches: int = 20,
    ) -> float:
        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.head_s):
            m.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for i, (features, _) in enumerate(val_loader):
                if i >= max_batches:
                    break
                features = features.permute(0, 2, 1, 3)
                weights, returns = self._forward(features)
                loss_dict = return_head_loss(weights, returns,
                                              lambda_turn=cfg.lambda_turn,
                                              lambda_vol=cfg.lambda_vol,
                                              dsr_eta=cfg.dsr_eta)
                total += loss_dict["total"].item()
                n += 1
        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.head_s):
            m.train()
        return total / max(n, 1)

    def validate_quality(
        self,
        val_loader: DataLoader,
        fp_reference_loss: float,
        cfg: QATConfig,
    ) -> dict[str, float]:
        """Compare quantized model quality against full-precision reference."""
        qt_loss = self.evaluate(val_loader, cfg)
        degradation = (qt_loss - fp_reference_loss) / max(abs(fp_reference_loss), 1e-8)
        passed = abs(degradation) <= cfg.sharpe_threshold
        return {
            "fp_loss": fp_reference_loss,
            "qt_loss": qt_loss,
            "relative_degradation": degradation,
            "within_threshold": float(passed),
        }

    def train(
        self,
        train_loader: DataLoader,
        cfg: QATConfig,
        val_loader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> None:
        if resume_from is not None and resume_from.exists():
            ckpt = load_checkpoint(resume_from)
            state = ckpt.model_state
            if "backbone" in state:
                self.backbone.load_state_dict(state["backbone"])
                self.fusion.load_state_dict(state["fusion"])
                self.regime_detector.load_state_dict(state["regime_detector"])
                self.input_projection.load_state_dict(state["input_projection"])
            if "head_return" in state:
                self.head_r.load_state_dict(state["head_return"])
            if "head_sharpe" in state:
                self.head_s.load_state_dict(state["head_sharpe"])
            if "aux_head" in state:
                self.aux_head.load_state_dict(state["aux_head"])
            print(f"Loaded checkpoint from {resume_from}")

        n_replaced = self.quantize()
        print(f"Replaced {n_replaced} Linear modules with TernaryLinear")

        params = self._all_params()
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: self._cosine_lr(s, cfg)
        )

        best_val_loss = float("inf")

        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.head_s):
            m.train()

        step = 0
        data_iter = iter(train_loader)

        while step < cfg.total_steps:
            try:
                features, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                features, _ = next(data_iter)

            optimizer.zero_grad(set_to_none=True)
            weights, returns = self._forward(features)
            loss_dict = return_head_loss(
                weights, returns,
                lambda_turn=cfg.lambda_turn,
                lambda_vol=cfg.lambda_vol,
                dsr_eta=cfg.dsr_eta,
            )
            loss = loss_dict["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                parts = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
                print(f"step={step:>7d}  {parts}  lr={lr_now:.2e}")

            if val_loader is not None and step % cfg.eval_every == 0:
                val_loss = self.evaluate(val_loader, cfg)
                print(f"  val loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save(cfg.best_path, step, optimizer, scheduler, best_val_loss)

            if step % cfg.save_every == 0:
                self._save(cfg.step_path_fn(step), step, optimizer, scheduler, best_val_loss)

        print(f"QAT complete at step {step}. Best val: {best_val_loss:.4f}")

    def _model_state(self) -> dict:
        return {
            "backbone": self.backbone.state_dict(),
            "fusion": self.fusion.state_dict(),
            "regime_detector": self.regime_detector.state_dict(),
            "input_projection": self.input_projection.state_dict(),
            "head_return": self.head_r.state_dict(),
            "head_sharpe": self.head_s.state_dict(),
            "aux_head": self.aux_head.state_dict(),
            "quantized": True,
        }

    def _save(self, path: Path, step: int, optimizer, scheduler, best: float) -> None:
        save_checkpoint(
            Checkpoint(
                step=step,
                model_state=self._model_state(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                best_val_loss=best,
            ),
            path,
        )


# ── Smoke data helpers ───────────────────────────────────────────────────────


def _write_smoke_shard(path: Path, n_rows: int, feature_dim: int = 280) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_struct = struct.Struct(f"<{feature_dim}f")
    rng = np.random.default_rng(seed=99)
    with open(path, "wb") as f:
        for _ in range(n_rows):
            row = rng.standard_normal(feature_dim).astype(np.float32)
            f.write(row_struct.pack(*row))


def _make_smoke_shards(tmp_dir: Path, n_assets: int, seq_len: int) -> list[Path]:
    paths = []
    for i in range(n_assets):
        for split in ("train", "validation"):
            p = tmp_dir / split / f"ASSET{i:02d}.f32bin"
            _write_smoke_shard(p, seq_len + 10)
            if split == "train":
                paths.append(p)
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization-Aware Training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument("--resume-meta", type=Path, default=None,
                        help="Meta-learning checkpoint to start from")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/qat"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.steps = args.steps if args.steps < 50_000 else 3
        args.batch_size = 2
        args.seq_len = max(args.seq_len, 8)
        args.workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DEFAULT_SPEC
    spec.validate()

    from train.portfolio_train import _build_model
    backbone, fusion, regime_detector, input_projection, head_r, head_s, aux_head = (
        _build_model(spec, device)
    )

    trainer = QATTrainer(
        backbone, fusion, regime_detector, input_projection,
        head_r, head_s, aux_head, spec, device,
    )

    cfg = QATConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.smoke:
        cfg.warmup_steps = 0
        cfg.log_every = 1
        cfg.eval_every = 2

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n_smoke = 5
            shard_paths = _make_smoke_shards(td, n_smoke, args.seq_len)
            val_paths = [td / "validation" / f"ASSET{i:02d}.f32bin" for i in range(n_smoke)]
            train_loader = build_portfolio_loader(
                shard_paths, args.seq_len, args.batch_size, num_workers=0
            )
            val_loader = build_portfolio_loader(
                val_paths, args.seq_len, args.batch_size, shuffle=False, num_workers=0
            )
            cfg.checkpoint_dir = td / "checkpoints"
            trainer.train(train_loader, cfg, val_loader, resume_from=args.resume_meta)
            print("Smoke test passed.")
    else:
        from market_data.dataset_ingestion import enumerate_shards
        train_paths = enumerate_shards(args.data_dir, "train")
        val_paths = enumerate_shards(args.data_dir, "validation")
        if not train_paths:
            raise RuntimeError(f"No train shards at {args.data_dir}/train")
        train_loader = build_portfolio_loader(
            train_paths, args.seq_len, args.batch_size, num_workers=args.workers
        )
        val_loader = build_portfolio_loader(
            val_paths, args.seq_len, args.batch_size, shuffle=False, num_workers=args.workers
        )
        trainer.train(train_loader, cfg, val_loader, resume_from=args.resume_meta)


if __name__ == "__main__":
    main()
