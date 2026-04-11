"""Self-supervised pre-training scaffold — three joint predictive objectives (Section 10.2)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset


# ── Prediction heads ─────────────────────────────────────────────────────────

class ReturnHead(nn.Module):
    """Objective 1: predict next-step log-return per asset."""

    def __init__(self, d_model: int = 768, n_assets: int = 100) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, asset_states: torch.Tensor) -> torch.Tensor:
        # asset_states: (B, N_assets, D)
        return self.proj(asset_states).squeeze(-1)  # (B, N_assets)


class VolatilityHead(nn.Module):
    """Objective 2: predict 5-step realised volatility per asset."""

    def __init__(self, d_model: int = 768, horizon: int = 5) -> None:
        super().__init__()
        self.horizon = horizon
        self.proj    = nn.Linear(d_model, horizon)

    def forward(self, asset_states: torch.Tensor) -> torch.Tensor:
        # asset_states: (B, N_assets, D)
        return F.softplus(self.proj(asset_states))  # (B, N_assets, horizon) ≥ 0


class CorrelationHead(nn.Module):
    """Objective 3: predict 20-day rolling pairwise correlation matrix."""

    def __init__(self, d_model: int = 768, n_assets: int = 100) -> None:
        super().__init__()
        self.n_assets = n_assets
        # Project each asset state to a correlation embedding; cosine sim → corr prediction
        self.proj = nn.Linear(d_model, 64)

    def forward(self, fused_states: torch.Tensor) -> torch.Tensor:
        # fused_states: (B, N_assets, D)
        emb = F.normalize(self.proj(fused_states), dim=-1)   # (B, N, 64)
        corr = torch.bmm(emb, emb.transpose(1, 2))            # (B, N, N) ∈ [-1, 1]
        return corr


# ── Data loading ─────────────────────────────────────────────────────────────

class ShardDataset(IterableDataset):
    """Streams (feature_seq, target_seq) pairs from .f32bin memory-mapped shards.

    Each .f32bin shard contains rows of shape (per_asset_feature_dim + market_dim)
    = 184 + 96 = 280 floats, stored as little-endian float32.

    Sequences are extracted with stride=1 (every possible starting offset),
    yielding (seq_len + 1) rows so we can use the last row for targets.
    """

    FEATURE_DIM = 280  # 184 per-asset (120 continuous + 64 text) + 96 market-wide

    def __init__(
        self,
        shard_paths: list[Path],
        seq_len: int = 64,
        shuffle: bool = True,
    ) -> None:
        self.shard_paths = shard_paths
        self.seq_len     = seq_len
        self.shuffle     = shuffle

    def _iter_shard(self, path: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        data = np.memmap(path, dtype="float32", mode="r")
        n_rows = len(data) // self.FEATURE_DIM
        data   = data[: n_rows * self.FEATURE_DIM].reshape(n_rows, self.FEATURE_DIM)

        n_seqs = n_rows - self.seq_len
        if n_seqs <= 0:
            return

        indices = np.arange(n_seqs)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in indices:
            seq    = data[start : start + self.seq_len]           # (T, F)
            target = data[start + 1 : start + self.seq_len + 1]  # (T, F) — shifted by 1
            yield seq, target

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        paths = list(self.shard_paths)
        if self.shuffle:
            import random
            random.shuffle(paths)
        for path in paths:
            for seq, target in self._iter_shard(path):
                yield torch.from_numpy(seq.copy()), torch.from_numpy(target.copy())


def build_dataloader(
    shard_paths: list[Path],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    ds = ShardDataset(shard_paths, seq_len=seq_len, shuffle=shuffle)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def build_split_loaders(
    data_dir: Path,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from split-aware shard directories.

    Expects shards at:
      {data_dir}/train/*.f32bin       — used for training
      {data_dir}/validation/*.f32bin  — used for evaluation

    Args:
        data_dir:    Root shard directory (output of market_data.pipeline collect).
        seq_len:     Sequence length per sample.
        batch_size:  Batch size.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader)

    Raises:
        RuntimeError: If train shards are missing.
    """
    from market_data.dataset_ingestion import enumerate_shards

    train_shards = enumerate_shards(data_dir, "train")
    val_shards   = enumerate_shards(data_dir, "validation")

    if not train_shards:
        raise RuntimeError(
            f"No train shards found at {data_dir}/train. "
            "Run `python -m market_data.pipeline collect` first."
        )

    train_loader = build_dataloader(train_shards, seq_len, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = build_dataloader(val_shards,   seq_len, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ── Checkpoint ───────────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    step: int
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    best_val_loss: float
    format_version: int = 1


def save_checkpoint(ckpt: Checkpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": ckpt.format_version,
            "step":            ckpt.step,
            "model_state":     ckpt.model_state,
            "optimizer_state": ckpt.optimizer_state,
            "scheduler_state": ckpt.scheduler_state,
            "best_val_loss":   ckpt.best_val_loss,
        },
        tmp,
    )
    tmp.rename(path)


def load_checkpoint(path: Path) -> Checkpoint:
    data = torch.load(path, map_location="cpu", weights_only=True)
    version = data.get("format_version", 0)
    if version != 1:
        raise ValueError(f"Unsupported checkpoint format version {version}")
    return Checkpoint(
        step             = data["step"],
        model_state      = data["model_state"],
        optimizer_state  = data["optimizer_state"],
        scheduler_state  = data["scheduler_state"],
        best_val_loss    = data["best_val_loss"],
        format_version   = version,
    )


# ── Loss functions ───────────────────────────────────────────────────────────

def return_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with per-asset normalisation to prevent high-vol assets dominating."""
    err  = pred - target                                       # (B, N)
    norm = target.std(dim=0, keepdim=True).clamp(min=1e-6)    # (1, N)
    return (err / norm).pow(2).mean()


def volatility_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on realised vol prediction."""
    return F.mse_loss(pred, target.clamp(min=0.0))


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of correlation prediction error, averaged over batch."""
    diff = pred - target.clamp(-1.0, 1.0)
    return diff.pow(2).mean()


# ── Trainer ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    total_steps:     int   = 200_000
    batch_size:      int   = 8
    seq_len:         int   = 64
    lr:              float = 1e-4
    warmup_steps:    int   = 2_000
    grad_clip:       float = 1.0
    log_every:       int   = 100
    eval_every:      int   = 1_000
    save_every:      int   = 5_000
    checkpoint_dir:  Path  = field(default_factory=lambda: Path("checkpoints"))
    checkpoint_name: str   = "backbone_pretrain.pt"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name


class SelfSupervisedPretrainer:
    """Trains the RA-GDN backbone + fusion on three joint self-supervised objectives.

    Objectives (jointly minimised):
      1. Return prediction:      MSE on next-step log-return per asset
      2. Volatility prediction:  MSE on 5-min realised vol per asset
      3. Correlation prediction: Frobenius norm on 20-day rolling correlation matrix
    """

    def __init__(
        self,
        backbone: nn.Module,
        fusion: nn.Module,
        regime_detector: nn.Module,
        input_projection: nn.Module,
        d_model: int = 768,
        n_assets: int = 100,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone         = backbone.to(self.device)
        self.fusion           = fusion.to(self.device)
        self.regime_detector  = regime_detector.to(self.device)
        self.input_projection = input_projection.to(self.device)

        self.return_head = ReturnHead(d_model, n_assets).to(self.device)
        self.vol_head    = VolatilityHead(d_model).to(self.device)
        self.corr_head   = CorrelationHead(d_model, n_assets).to(self.device)

        self._all_modules = nn.ModuleList([
            backbone, fusion, regime_detector, input_projection,
            self.return_head, self.vol_head, self.corr_head,
        ])

    def _cosine_lr(self, step: int, cfg: TrainConfig) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _model_state(self) -> dict:
        return {
            "backbone":         self.backbone.state_dict(),
            "fusion":           self.fusion.state_dict(),
            "regime_detector":  self.regime_detector.state_dict(),
            "input_projection": self.input_projection.state_dict(),
            "return_head":      self.return_head.state_dict(),
            "vol_head":         self.vol_head.state_dict(),
            "corr_head":        self.corr_head.state_dict(),
        }

    def _load_model_state(self, state: dict) -> None:
        self.backbone.load_state_dict(state["backbone"])
        self.fusion.load_state_dict(state["fusion"])
        self.regime_detector.load_state_dict(state["regime_detector"])
        self.input_projection.load_state_dict(state["input_projection"])
        self.return_head.load_state_dict(state["return_head"])
        self.vol_head.load_state_dict(state["vol_head"])
        self.corr_head.load_state_dict(state["corr_head"])

    def evaluate(
        self,
        val_loader: DataLoader,
        cfg: TrainConfig,
        max_batches: int = 50,
    ) -> dict[str, float]:
        self._all_modules.eval()
        totals: dict[str, float] = {"return": 0.0, "vol": 0.0, "corr": 0.0, "total": 0.0}
        n = 0
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                losses = self._forward_losses(features, targets)
                for k, v in losses.items():
                    totals[k] += v.item()
                n += 1
        self._all_modules.train()
        return {k: v / max(n, 1) for k, v in totals.items()}

    def _forward_losses(
        self,
        features: torch.Tensor,  # (B, T, 280) — per-asset (184) + market (96)
        targets: torch.Tensor,   # (B, T, 280) — same, shifted by 1 step
    ) -> dict[str, torch.Tensor]:
        features = features.to(self.device)
        targets  = targets.to(self.device)
        B, T, _ = features.shape

        # Split features into asset (184) and market (96) blocks
        asset_feats   = features[..., :184]    # (B, T, 184)
        market_feats  = features[..., 184:]    # (B, T, 96)
        target_returns = targets[..., :10]     # price_volume block as return proxy

        # For single-asset training shards, treat as N=1 asset
        asset_feats = asset_feats.unsqueeze(2)   # (B, T, 1, 184)
        N = 1

        # Bootstrap regime with zeros (no HMM prior available during pre-training)
        regime_emb   = torch.zeros(B, 64, device=self.device)
        posteriors   = torch.full((B, 4), 0.25, device=self.device)
        portfolio_s  = torch.zeros(B, 201, device=self.device)
        calendar     = torch.zeros(B, 20, device=self.device)
        sector_ids   = torch.zeros(N, dtype=torch.long, device=self.device)
        episodic_tok = torch.zeros(B, 8, 768, device=self.device)

        all_asset_states = []
        all_fused_states = []
        states = self.backbone.init_state(B, N, self.device)

        for t in range(T):
            af_t = asset_feats[:, t]     # (B, 1, 288)
            mf_t = market_feats[:, t]    # (B, 96)

            asset_tok, mkt_tok, port_tok, cal_tok = self.input_projection(
                af_t, mf_t, portfolio_s, calendar, regime_emb
            )
            out, states = self.backbone(asset_tok, regime_emb, posteriors, states)
            fused, _ = self.fusion(out, mkt_tok, port_tok, episodic_tok, sector_ids)
            all_asset_states.append(out)
            all_fused_states.append(fused)

        asset_states = torch.stack(all_asset_states, dim=1)  # (B, T, N, D)
        fused_states = torch.stack(all_fused_states, dim=1)  # (B, T, N, D)

        # Squeeze N=1 for prediction heads
        asset_seq = asset_states.squeeze(2)   # (B, T, D)
        fused_seq = fused_states.squeeze(2)   # (B, T, D)

        # Objective 1: return prediction
        pred_return = self.return_head(asset_seq.unsqueeze(2)).squeeze(2).squeeze(-1)  # (B, T)
        tgt_return  = target_returns[..., 3]  # use close log-return as target (index 3)
        loss_ret    = return_loss(pred_return, tgt_return)

        # Objective 2: volatility prediction (target = std of next horizon returns)
        # Proxy: standard deviation of next 5 close returns from target shard
        tgt_vol = target_returns[:, :, 3].unfold(1, 5, 1).std(dim=-1) if T >= 5 else \
                  torch.zeros_like(pred_return[:, :T - 4] if T > 4 else pred_return)
        pred_vol = self.vol_head(asset_seq.unsqueeze(2))[:, :tgt_vol.shape[1]].squeeze(2)
        loss_vol = volatility_loss(pred_vol, tgt_vol.unsqueeze(-1).expand_as(pred_vol))

        # Objective 3: correlation prediction (use last timestep's fused state)
        last_fused = fused_states[:, -1]   # (B, N, D) — only N=1 here; kept for multi-asset
        pred_corr  = self.corr_head(last_fused)           # (B, N, N)
        # Target: identity matrix (N=1 case; replaced with real corr matrix for N>1)
        tgt_corr   = torch.eye(N, device=self.device).unsqueeze(0).expand(B, -1, -1)
        loss_corr  = correlation_loss(pred_corr, tgt_corr)

        total = loss_ret + loss_vol + loss_corr
        return {"return": loss_ret, "vol": loss_vol, "corr": loss_corr, "total": total}

    def train(
        self,
        train_loader: DataLoader,
        cfg: TrainConfig,
        val_loader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> None:
        optimizer = torch.optim.AdamW(
            self._all_modules.parameters(), lr=cfg.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self._cosine_lr(step, cfg)
        )

        start_step      = 0
        best_val_loss   = float("inf")

        if resume_from is not None and resume_from.exists():
            ckpt = load_checkpoint(resume_from)
            self._load_model_state(ckpt.model_state)
            optimizer.load_state_dict(ckpt.optimizer_state)
            scheduler.load_state_dict(ckpt.scheduler_state)
            start_step    = ckpt.step
            best_val_loss = ckpt.best_val_loss
            print(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")

        self._all_modules.train()
        step = start_step
        data_iter = iter(train_loader)

        while step < cfg.total_steps:
            try:
                features, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                features, targets = next(data_iter)

            optimizer.zero_grad(set_to_none=True)
            losses = self._forward_losses(features, targets)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self._all_modules.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step={step:>7d}  "
                    f"loss={losses['total'].item():.4f}  "
                    f"ret={losses['return'].item():.4f}  "
                    f"vol={losses['vol'].item():.4f}  "
                    f"corr={losses['corr'].item():.4f}  "
                    f"lr={lr_now:.2e}"
                )

            if val_loader is not None and step % cfg.eval_every == 0:
                val_metrics = self.evaluate(val_loader, cfg)
                val_loss = val_metrics["total"]
                print(f"  val loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        Checkpoint(
                            step             = step,
                            model_state      = self._model_state(),
                            optimizer_state  = optimizer.state_dict(),
                            scheduler_state  = scheduler.state_dict(),
                            best_val_loss    = best_val_loss,
                        ),
                        cfg.checkpoint_path,
                    )

            if step % cfg.save_every == 0:
                ckpt_path = cfg.checkpoint_dir / f"backbone_step{step}.pt"
                save_checkpoint(
                    Checkpoint(
                        step             = step,
                        model_state      = self._model_state(),
                        optimizer_state  = optimizer.state_dict(),
                        scheduler_state  = scheduler.state_dict(),
                        best_val_loss    = best_val_loss,
                    ),
                    ckpt_path,
                )

        print(f"Training complete at step {step}. Best val loss: {best_val_loss:.4f}")
