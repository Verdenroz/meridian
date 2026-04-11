"""Portfolio optimization training — Meridian-R and Meridian-S heads.

Trains the two portfolio output heads on top of the pre-trained backbone.

Meridian-R (return head):
  Backbone unfrozen. Loss = -DSR + λ_turn·Turnover + λ_vol·VolAux.
  ~50K steps.

Meridian-S (Sharpe head):
  Backbone frozen. Loss = -rolling Sharpe ratio.
  ~30K steps.

Usage:

    # Train return head (loads pre-trained backbone checkpoint)
    python -m train.portfolio_train --head return \\
        --data-dir data/mmaps \\
        --resume-backbone checkpoints/pretrain/backbone_pretrain.pt \\
        --steps 50000 \\
        --checkpoint-dir checkpoints/portfolio

    # Then train Sharpe head (frozen backbone, loads return head checkpoint)
    python -m train.portfolio_train --head sharpe \\
        --data-dir data/mmaps \\
        --resume-backbone checkpoints/portfolio/return_head_best.pt \\
        --steps 30000 \\
        --checkpoint-dir checkpoints/portfolio

    # Smoke test
    python -m train.portfolio_train --head return --smoke --steps 3
"""

from __future__ import annotations

import argparse
import math
import struct
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from core.cross_asset_fusion import CrossAssetFusion
from core.regime_aware_backbone import InputProjection, RAGDNBackbone, RegimeDetector
from core.self_supervised_pretraining import Checkpoint, load_checkpoint, save_checkpoint
from optimization.losses import return_head_loss, rolling_sharpe_loss
from optimization.portfolio_heads import AuxHeadSurprise, MeridianR, MeridianS, PortfolioVolHead
from train.spec import DEFAULT_SPEC, MeridianSpec


# ── Multi-asset dataset ───────────────────────────────────────────────────────

class MultiAssetShardDataset(IterableDataset):
    """Yields synchronized (features, returns) windows across N assets.

    Loads one f32bin shard per symbol, aligns them by truncating to the
    shortest shard length, then samples T+1-step windows at the same
    position offset across all assets.

    Each sample:
        features: (N, T, F)   — F=280 (184 per-asset + 96 market)
        returns:  (N, T)      — close log-return proxy (feature index 3)
    """

    FEATURE_DIM = 280
    RETURN_IDX  = 3   # index of close log-return within price_volume block

    def __init__(
        self,
        shard_paths: list[Path],
        seq_len: int = 64,
        shuffle: bool = True,
        stride: int = 1,
        augment: bool = False,
    ) -> None:
        self.shard_paths = shard_paths
        self.seq_len     = seq_len
        self.shuffle     = shuffle
        self.stride      = stride
        self.augment     = augment

    def _load_shards(self) -> np.ndarray | None:
        """Load and align all shards to shape (N, rows, F)."""
        arrays = []
        for path in self.shard_paths:
            raw  = np.memmap(path, dtype="float32", mode="r")
            n    = len(raw) // self.FEATURE_DIM
            data = raw[: n * self.FEATURE_DIM].reshape(n, self.FEATURE_DIM)
            arrays.append(data)
        if not arrays:
            return None
        min_rows = min(a.shape[0] for a in arrays)
        # Tail-align: keep the most recent min_rows rows (shared calendar end)
        aligned = np.stack([a[-min_rows:] for a in arrays], axis=0)  # (N, T, F)
        return aligned

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        data = self._load_shards()  # (N, rows, F)
        if data is None:
            return

        N, rows, F = data.shape
        n_seqs = rows - self.seq_len
        if n_seqs <= 0:
            return

        indices = np.arange(0, n_seqs, self.stride)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in indices:
            window  = data[:, start : start + self.seq_len + 1, :]  # (N, T+1, F)
            feats   = torch.from_numpy(window[:, :-1, :].copy())    # (N, T, F)
            if self.augment:
                feats = feats + torch.randn_like(feats) * 0.01
            returns = torch.from_numpy(
                window[:, 1:, self.RETURN_IDX].copy()               # (N, T) next-step returns
            )
            yield feats, returns


def build_portfolio_loader(
    shard_paths: list[Path],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    stride: int = 1,
    augment: bool = False,
) -> DataLoader:
    ds = MultiAssetShardDataset(shard_paths, seq_len=seq_len, shuffle=shuffle,
                                stride=stride, augment=augment)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class PortfolioTrainConfig:
    head: str = "return"           # "return" or "sharpe"
    total_steps: int = 50_000
    batch_size: int = 4
    seq_len: int = 64
    lr: float = 3e-5               # lower than pre-training — fine-tuning
    warmup_steps: int = 500
    grad_clip: float = 0.5
    lambda_turn: float = 0.1
    lambda_vol: float = 0.05
    dsr_eta: float = 0.01
    log_every: int = 10
    eval_every: int = 1_000
    save_every: int = 5_000
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/portfolio"))
    bptt_chunk: int = 16           # truncated BPTT chunk length (0 = full seq)
    gradient_checkpointing: bool = False
    compile: bool = False          # torch.compile the backbone (2-4x faster, ~60s warmup)

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / f"{self.head}_head_best.pt"

    @property
    def step_path_fn(self):
        return lambda step: self.checkpoint_dir / f"{self.head}_head_step{step}.pt"


# ── Trainer ───────────────────────────────────────────────────────────────────

class PortfolioTrainer:
    """Trains MeridianR or MeridianS on top of the pre-trained backbone."""

    def __init__(
        self,
        backbone: RAGDNBackbone,
        fusion: CrossAssetFusion,
        regime_detector: RegimeDetector,
        input_projection: InputProjection,
        head_r: MeridianR,
        head_s: MeridianS,
        aux_head: AuxHeadSurprise,
        vol_head: PortfolioVolHead,
        spec: MeridianSpec,
        device: torch.device,
    ) -> None:
        self.device   = device
        self.spec     = spec
        self.backbone          = backbone.to(device)
        self.fusion            = fusion.to(device)
        self.regime_detector   = regime_detector.to(device)
        self.input_projection  = input_projection.to(device)
        self.head_r            = head_r.to(device)
        self.head_s            = head_s.to(device)
        self.aux_head          = aux_head.to(device)
        self.vol_head          = vol_head.to(device)

    def _cosine_lr(self, step: int, cfg: PortfolioTrainConfig) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _forward(
        self,
        features: torch.Tensor,  # (B, N, T, F=280)
        head: str,
        cfg: "PortfolioTrainConfig | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full model forward pass over a T-step sequence.

        Uses bf16 autocast to halve activation memory and truncated BPTT
        (cfg.bptt_chunk > 0) to cap the gradient graph length.

        Returns:
            weights:  (B, T, N) — allocation weights at each step
            asset_returns: (B, T, N) — next-step return proxy from features
        """
        features = features.to(self.device)
        B, N, T, Fd = features.shape
        bptt_chunk = cfg.bptt_chunk if cfg is not None else 0

        # Split per-asset (184) and market (96) blocks
        asset_feats  = features[..., :184]  # (B, N, T, 184)
        market_feats = features[..., 184:]  # (B, N, T, 96) — take mean across assets

        # Market features: average across assets (they share the same market context)
        mkt_mean = market_feats.mean(dim=1)  # (B, T, 96)

        # Bootstrap regime/portfolio state
        regime_emb  = torch.zeros(B, self.spec.regime_emb_dim, device=self.device)
        posteriors  = torch.full((B, self.spec.n_regimes), 0.25, device=self.device)
        portfolio_s = torch.zeros(B, self.spec.portfolio_state_dim, device=self.device)
        calendar    = torch.zeros(B, self.spec.calendar_dim, device=self.device)
        sector_ids  = torch.zeros(N, dtype=torch.long, device=self.device)
        episodic    = torch.zeros(B, self.spec.n_episodic_retrievals, self.spec.d_model, device=self.device)

        states = self.backbone.init_state(B, N, self.device)

        all_weights: list[torch.Tensor] = []
        all_vol_preds: list[torch.Tensor] = []

        use_amp = self.device.type == "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)

        # Pre-allocate reusable zero tensors (avoids 128 CUDA allocs per batch)
        n_full = self.spec.n_assets  # 100
        _zeros_n = torch.zeros(B, n_full, device=self.device)
        _zeros_1 = torch.zeros(B, 1, device=self.device)

        # Pre-compute sector bias (constant across all timesteps)
        with torch.no_grad():
            n_episodic = episodic.shape[1]
            n_total = N + 1 + 1 + n_episodic  # assets + market + portfolio + episodic
            n_extras = n_total - N
            extra_ids = torch.full((n_extras,), -1, dtype=torch.long, device=self.device)
            full_ids = torch.cat([sector_ids, extra_ids], dim=0)
            precomputed_bias = self.fusion.sector_bias.bias_matrix(full_ids, n_total)

        with autocast_ctx:
            for t in range(T):
                af_t  = asset_feats[:, :, t, :]   # (B, N, 184)
                mf_t  = mkt_mean[:, t, :]          # (B, 96)

                asset_tok, mkt_tok, port_tok, _ = self.input_projection(
                    af_t, mf_t, portfolio_s, calendar, regime_emb
                )
                out, states = self.backbone(asset_tok, regime_emb, posteriors, states)
                fused, mkt_out = self.fusion(
                    out, mkt_tok, port_tok, episodic, sector_ids,
                    precomputed_bias=precomputed_bias,
                )

                # Portfolio head
                if head == "return":
                    w_t = self.head_r(fused)   # (B, N)
                else:
                    w_t = self.head_s(fused)   # (B, N)

                all_weights.append(w_t)

                # Volatility auxiliary: predict portfolio vol from market token
                if head == "return":
                    all_vol_preds.append(self.vol_head(mkt_out))  # (B, horizon)

                # Update portfolio state: [weights (100), unrealized P&L (100), value (1)] = 201
                w_padded = F.pad(w_t.detach(), (0, n_full - N))  # (B, 100)
                portfolio_s = torch.cat([
                    w_padded,    # weights (100)
                    _zeros_n,    # unrealised P&L (100)
                    _zeros_1,    # portfolio value (1)
                ], dim=-1)  # (B, 201)

                # Truncated BPTT: detach GDN state every bptt_chunk steps so the
                # backward graph covers at most bptt_chunk timesteps at a time.
                if bptt_chunk > 0 and (t + 1) % bptt_chunk == 0:
                    states = [s.detach_() for s in states]

        weights = torch.stack(all_weights, dim=1)  # (B, T, N)
        pred_vol = torch.stack(all_vol_preds, dim=1) if all_vol_preds else None  # (B, T, horizon)

        # Extract return proxy from features: price_volume block index 3 = close log-return
        # Shift by 1: returns at t are from step t+1 of the input window
        # Here we use the per-asset features at each step as the realized return proxy
        asset_returns = asset_feats[:, :, :, 3].permute(0, 2, 1)  # (B, T, N)

        return weights, asset_returns, pred_vol

    def _all_params(self, head: str) -> list[nn.Parameter]:
        head_params = list(self.head_r.parameters() if head == "return"
                           else self.head_s.parameters())
        if head == "return":
            # Backbone unfrozen for return head training
            return (
                head_params
                + list(self.backbone.parameters())
                + list(self.fusion.parameters())
                + list(self.input_projection.parameters())
                + list(self.aux_head.parameters())
                + list(self.vol_head.parameters())
            )
        else:
            # Backbone frozen for Sharpe head training
            return head_params

    def evaluate(
        self,
        val_loader: DataLoader,
        cfg: PortfolioTrainConfig,
        max_batches: int = 20,
    ) -> float:
        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.head_s, self.vol_head):
            m.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for i, (features, _) in enumerate(val_loader):
                if i >= max_batches:
                    break
                # features: (B, N, T, F) after collation
                features = features.permute(0, 2, 1, 3)  # (B, N, T, F)
                weights, returns, pred_vol = self._forward(features, cfg.head, cfg)
                if cfg.head == "return":
                    loss_dict = return_head_loss(weights, returns,
                                                 pred_vol=pred_vol,
                                                 lambda_turn=cfg.lambda_turn,
                                                 lambda_vol=cfg.lambda_vol,
                                                 dsr_eta=cfg.dsr_eta)
                    loss = loss_dict["total"]
                else:
                    loss = rolling_sharpe_loss(weights, returns)
                total += loss.item()
                n += 1
        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.head_s, self.vol_head):
            m.train()
        return total / max(n, 1)

    def train(
        self,
        train_loader: DataLoader,
        cfg: PortfolioTrainConfig,
        val_loader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> None:
        if cfg.gradient_checkpointing:
            self.backbone.enable_gradient_checkpointing()

        params = self._all_params(cfg.head)
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: self._cosine_lr(s, cfg)
        )

        start_step    = 0
        best_val_loss = float("inf")

        if resume_from is not None and resume_from.exists():
            ckpt = load_checkpoint(resume_from)
            # Load backbone + fusion from checkpoint (pre-training or earlier portfolio ckpt)
            state = ckpt.model_state
            if "backbone" in state:
                self.backbone.load_state_dict(state["backbone"])
                self.fusion.load_state_dict(state["fusion"])
                self.regime_detector.load_state_dict(state["regime_detector"])
                self.input_projection.load_state_dict(state["input_projection"])
            if "vol_head" in state:
                try:
                    self.vol_head.load_state_dict(state["vol_head"])
                except RuntimeError:
                    print("vol_head checkpoint incompatible — initialising fresh.")
            if f"head_{cfg.head}" in state:
                (self.head_r if cfg.head == "return" else self.head_s).load_state_dict(
                    state[f"head_{cfg.head}"]
                )
                optimizer.load_state_dict(ckpt.optimizer_state)
                scheduler.load_state_dict(ckpt.scheduler_state)
                start_step    = ckpt.step
                best_val_loss = ckpt.best_val_loss
            print(f"Resumed from step {start_step}  best_val={best_val_loss:.4f}")

        # Compile after checkpoint loading so state dict keys match (uncompiled backbone
        # has plain keys; torch.compile wraps them under _orig_mod.* prefix)
        if cfg.compile:
            print("Compiling backbone (one-time, ~60s)...")
            self.backbone = torch.compile(self.backbone)
            print("Compiled.")

        if cfg.head == "sharpe":
            for p in (list(self.backbone.parameters())
                      + list(self.fusion.parameters())
                      + list(self.input_projection.parameters())):
                p.requires_grad_(False)

        for m in (self.backbone, self.fusion, self.input_projection,
                  self.head_r, self.head_s, self.vol_head):
            m.train()

        step      = start_step
        data_iter = iter(train_loader)

        while step < cfg.total_steps:
            try:
                features, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                features, _ = next(data_iter)

            # DataLoader collates (N, T, F) items → (B, N, T, F); ready for forward
            optimizer.zero_grad(set_to_none=True)

            weights, returns, pred_vol = self._forward(features, cfg.head, cfg)

            if cfg.head == "return":
                loss_dict = return_head_loss(
                    weights, returns,
                    pred_vol=pred_vol,
                    lambda_turn=cfg.lambda_turn,
                    lambda_vol=cfg.lambda_vol,
                    dsr_eta=cfg.dsr_eta,
                )
                loss = loss_dict["total"]
            else:
                loss = rolling_sharpe_loss(weights, returns)
                loss_dict = {"total": loss}

            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                parts  = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
                print(f"step={step:>7d}  {parts}  lr={lr_now:.2e}")

            if val_loader is not None and step % cfg.eval_every == 0:
                val_loss = self.evaluate(val_loader, cfg)
                print(f"  val loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save(cfg.best_path, step, optimizer, scheduler, best_val_loss)

            if step % cfg.save_every == 0:
                self._save(cfg.step_path_fn(step), step, optimizer, scheduler, best_val_loss)

        print(f"Training complete at step {step}. Best val: {best_val_loss:.4f}")

    def _model_state(self) -> dict:
        # torch.compile wraps the backbone under _orig_mod; unwrap so checkpoints
        # use plain key names and can be loaded by an uncompiled model.
        backbone_mod = getattr(self.backbone, "_orig_mod", self.backbone)
        return {
            "backbone":         backbone_mod.state_dict(),
            "fusion":           self.fusion.state_dict(),
            "regime_detector":  self.regime_detector.state_dict(),
            "input_projection": self.input_projection.state_dict(),
            "head_return":      self.head_r.state_dict(),
            "head_sharpe":      self.head_s.state_dict(),
            "aux_head":         self.aux_head.state_dict(),
            "vol_head":         self.vol_head.state_dict(),
        }

    def _save(self, path: Path, step: int, optimizer, scheduler, best: float) -> None:
        save_checkpoint(
            Checkpoint(
                step             = step,
                model_state      = self._model_state(),
                optimizer_state  = optimizer.state_dict(),
                scheduler_state  = scheduler.state_dict(),
                best_val_loss    = best,
            ),
            path,
        )


# ── Smoke data helpers ────────────────────────────────────────────────────────

def _write_smoke_shard(path: Path, n_rows: int, feature_dim: int = 280) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_struct = struct.Struct(f"<{feature_dim}f")
    rng = np.random.default_rng(seed=0)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_model(spec: MeridianSpec, device: torch.device):
    from train.pretrain import build_model
    backbone, fusion, regime_detector, input_projection = build_model(spec, device)
    head_r   = MeridianR(spec.d_model).to(device)
    head_s   = MeridianS(spec.d_model).to(device)
    aux_head = AuxHeadSurprise(spec.d_model).to(device)
    vol_head = PortfolioVolHead(spec.d_model).to(device)
    return backbone, fusion, regime_detector, input_projection, head_r, head_s, aux_head, vol_head


def main() -> None:
    # Suppress noisy torch.compile / inductor warnings that obscure training logs
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._inductor")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo")

    parser = argparse.ArgumentParser(description="Portfolio head training")
    parser.add_argument("--head",             choices=["return", "sharpe"], default="return")
    parser.add_argument("--data-dir",         type=Path, default=Path("data/mmaps"))
    parser.add_argument("--steps",            type=int,  default=50_000)
    parser.add_argument("--batch-size",       type=int,  default=4)
    parser.add_argument("--seq-len",          type=int,  default=64)
    parser.add_argument("--lr",               type=float, default=3e-5)
    parser.add_argument("--checkpoint-dir",   type=Path, default=Path("checkpoints/portfolio"))
    parser.add_argument("--resume-backbone",  type=Path, default=None,
                        help="Pre-trained backbone checkpoint to load")
    parser.add_argument("--workers",          type=int,  default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute activations during backward to save VRAM. "
                             "Required on 24GB GPUs.")
    parser.add_argument("--compile",          action="store_true",
                        help="torch.compile the backbone for faster training (~60s warmup, "
                             "then 2-4x faster per step). Recommended on A100/H100.")
    parser.add_argument("--smoke",            action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.steps      = args.steps if args.steps < 50_000 else 3
        args.batch_size = 2
        args.seq_len    = max(args.seq_len, 8)
        args.workers    = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec   = DEFAULT_SPEC
    spec.validate()

    backbone, fusion, regime_detector, input_projection, head_r, head_s, aux_head, vol_head = (
        _build_model(spec, device)
    )
    trainer = PortfolioTrainer(
        backbone, fusion, regime_detector, input_projection,
        head_r, head_s, aux_head, vol_head, spec, device,
    )

    n_symbols = spec.n_assets  # 100

    if args.smoke:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n_smoke = 5  # fewer assets for smoke speed
            shard_paths = _make_smoke_shards(td, n_smoke, args.seq_len)
            val_paths   = [td / "validation" / f"ASSET{i:02d}.f32bin" for i in range(n_smoke)]
            train_loader = build_portfolio_loader(
                shard_paths, args.seq_len, args.batch_size, num_workers=0
            )
            val_loader = build_portfolio_loader(
                val_paths, args.seq_len, args.batch_size, shuffle=False, num_workers=0
            )
            cfg = PortfolioTrainConfig(
                head=args.head, total_steps=args.steps,
                batch_size=args.batch_size, seq_len=args.seq_len,
                lr=args.lr, warmup_steps=0, log_every=1, eval_every=2,
                checkpoint_dir=td / "checkpoints",
            )
            trainer.train(train_loader, cfg, val_loader,
                          resume_from=args.resume_backbone)
            print("Smoke test passed.")
    else:
        from market_data.dataset_ingestion import enumerate_shards
        train_paths = enumerate_shards(args.data_dir, "train")
        val_paths   = enumerate_shards(args.data_dir, "validation")
        if not train_paths:
            raise RuntimeError(f"No train shards at {args.data_dir}/train")
        train_loader = build_portfolio_loader(
            train_paths, args.seq_len, args.batch_size, num_workers=args.workers,
            stride=16, augment=True,
        )
        val_loader = build_portfolio_loader(
            val_paths, args.seq_len, args.batch_size, shuffle=False, num_workers=args.workers,
        )
        cfg = PortfolioTrainConfig(
            head=args.head, total_steps=args.steps,
            batch_size=args.batch_size, seq_len=args.seq_len,
            lr=args.lr, checkpoint_dir=args.checkpoint_dir,
            gradient_checkpointing=args.gradient_checkpointing,
            compile=args.compile,
        )
        trainer.train(train_loader, cfg, val_loader,
                      resume_from=args.resume_backbone)


if __name__ == "__main__":
    main()
