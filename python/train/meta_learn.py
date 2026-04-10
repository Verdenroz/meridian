"""Meta-learning for plasticity — Phase 3 of training pipeline.

Trains the model to adapt quickly to regime transitions using Reptile-style
meta-learning.  Each episode finds a regime transition in the data, splits
into support (pre-transition) and query (post-transition), runs adaptation
on the support, then optimises query loss to learn fast-adapting parameters.

Usage:

    python -m train.meta_learn \\
        --data-dir data/mmaps \\
        --resume-portfolio checkpoints/portfolio/return_head_best.pt \\
        --steps 100000 \\
        --checkpoint-dir checkpoints/meta_learn

    # Smoke test
    python -m train.meta_learn --smoke --steps 3 --inner-steps 2
"""

from __future__ import annotations

import argparse
import copy
import math
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adaptation.adaptation import AdaptationConfig, AdaptationSystem
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


# ── Episode representation ───────────────────────────────────────────────────


class Episode(NamedTuple):
    """Support / query split around a regime transition."""

    support_features: torch.Tensor  # (N, T_support, F)
    support_returns: torch.Tensor   # (N, T_support)
    query_features: torch.Tensor    # (N, T_query, F)
    query_returns: torch.Tensor     # (N, T_query)


# ── Regime transition sampler ────────────────────────────────────────────────


class RegimeTransitionSampler:
    """Samples episodes centered on regime transitions from multi-asset data.

    Scans candidate windows with the RegimeDetector to find points where
    the dominant regime changes.  Splits at the transition into support
    (pre-transition) and query (post-transition) segments.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        regime_detector: RegimeDetector,
        spec: MeridianSpec,
        support_len: int = 32,
        query_len: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._loader = dataloader
        self._detector = regime_detector
        self._spec = spec
        self._support_len = support_len
        self._query_len = query_len
        self._device = device
        self._iter = iter(dataloader)

    def _next_batch(self) -> torch.Tensor:
        """Get next (features, _) batch, cycling the dataloader."""
        try:
            features, _ = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            features, _ = next(self._iter)
        return features  # (B, N, T, F) after permute in caller or (N, T, F) per sample

    def _find_transition(self, features: torch.Tensor) -> int | None:
        """Find a regime transition index in a (N, T, F) sequence.

        Returns the timestep index where the dominant regime changes,
        or None if no transition is found.
        """
        N, T, F = features.shape
        if T < self._support_len + self._query_len:
            return None

        market_feats = features[:, :, 184:].mean(dim=0)  # (T, 96)

        prev_emb = None
        regime_ema = torch.zeros(1, self._spec.regime_emb_dim, device=self._device)
        portfolio_summary = torch.zeros(1, 3, device=self._device)
        prev_regime = None

        with torch.no_grad():
            for t in range(T):
                mf = market_feats[t].unsqueeze(0).to(self._device)  # (1, 96)
                emb, post, _ = self._detector(mf, portfolio_summary, regime_ema, prev_emb)
                regime_ema = 0.9 * regime_ema + 0.1 * emb
                prev_emb = emb
                curr_regime = post.argmax(dim=-1).item()

                if prev_regime is not None and curr_regime != prev_regime:
                    # Check if we have enough room for support + query
                    if t >= self._support_len and (T - t) >= self._query_len:
                        return t
                prev_regime = curr_regime

        return None

    def sample(self) -> Episode:
        """Sample a single episode centered on a regime transition.

        Falls back to a random midpoint split if no transition is detected
        within a reasonable number of attempts.
        """
        for _ in range(10):
            batch = self._next_batch()
            # batch may be (B, N, T, F) or (N, T, F) — handle both
            if batch.dim() == 4:
                features = batch[0]  # take first in batch: (N, T, F)
            else:
                features = batch

            transition = self._find_transition(features)
            if transition is not None:
                break
        else:
            # Fallback: split at midpoint
            T = features.shape[1]
            transition = T // 2

        N, T, F = features.shape
        s_start = max(0, transition - self._support_len)
        q_end = min(T, transition + self._query_len)

        support = features[:, s_start:transition, :]
        query = features[:, transition:q_end, :]

        return Episode(
            support_features=support,
            support_returns=support[:, :, 3],  # close log-return proxy
            query_features=query,
            query_returns=query[:, :, 3],
        )


# ── Training config ──────────────────────────────────────────────────────────


@dataclass
class MetaLearnConfig:
    total_steps: int = 100_000
    outer_lr: float = 1e-4
    inner_lr: float = 3e-5
    inner_steps: int = 5
    episode_support_len: int = 32
    episode_query_len: int = 32
    batch_size: int = 4
    seq_len: int = 128          # long enough for support + query + margin
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 2_000
    save_every: int = 5_000
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/meta_learn"))
    checkpoint_name: str = "meta_learn_best.pt"

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_name

    @property
    def step_path_fn(self):
        return lambda step: self.checkpoint_dir / f"meta_learn_step{step}.pt"


# ── Meta-learner ─────────────────────────────────────────────────────────────


class MetaLearner:
    """Reptile-style meta-learner for fast adaptation to regime transitions."""

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

    def _all_modules(self) -> list[nn.Module]:
        return [
            self.backbone, self.fusion, self.regime_detector,
            self.input_projection, self.head_r, self.head_s, self.aux_head,
        ]

    def _all_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for m in self._all_modules():
            params.extend(m.parameters())
        return params

    def _forward_episode(
        self,
        features: torch.Tensor,  # (N, T, F=280)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward on a single episode segment.

        Returns weights (1, T, N) and asset_returns (1, T, N).
        """
        N, T, Fd = features.shape
        features = features.unsqueeze(0).to(self.device)  # (1, N, T, Fd)

        asset_feats = features[..., :184]   # (1, N, T, 184)
        mkt_mean = features[..., 184:].mean(dim=1)  # (1, T, 96)

        regime_emb = torch.zeros(1, self.spec.regime_emb_dim, device=self.device)
        posteriors = torch.full((1, self.spec.n_regimes), 0.25, device=self.device)
        portfolio_s = torch.zeros(1, self.spec.portfolio_state_dim, device=self.device)
        calendar = torch.zeros(1, self.spec.calendar_dim, device=self.device)
        sector_ids = torch.zeros(N, dtype=torch.long, device=self.device)
        episodic = torch.zeros(1, self.spec.n_episodic_retrievals, self.spec.d_model, device=self.device)

        states = self.backbone.init_state(1, self.device)
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
                torch.zeros(1, n_full, device=self.device),
                torch.zeros(1, 1, device=self.device),
            ], dim=-1)

        weights = torch.stack(all_weights, dim=1)  # (1, T, N)
        asset_returns = asset_feats[:, :, :, 3].permute(0, 2, 1)  # (1, T, N)
        return weights, asset_returns

    def _cosine_lr(self, step: int, cfg: MetaLearnConfig) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(
        self,
        sampler: RegimeTransitionSampler,
        cfg: MetaLearnConfig,
        resume_from: Path | None = None,
    ) -> None:
        """Reptile meta-learning loop.

        For each outer step:
          1. Snapshot current parameters (theta)
          2. Inner loop: run ``inner_steps`` of gradient descent on support
          3. Compute query loss with adapted parameters
          4. Outer update: theta += outer_lr * (theta_adapted - theta)
        """
        outer_params = self._all_params()
        outer_optimizer = torch.optim.AdamW(outer_params, lr=cfg.outer_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            outer_optimizer, lr_lambda=lambda s: self._cosine_lr(s, cfg)
        )

        start_step = 0
        best_val_loss = float("inf")

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
            if "meta_step" in state:
                start_step = state["meta_step"]
                best_val_loss = ckpt.best_val_loss
                outer_optimizer.load_state_dict(ckpt.optimizer_state)
                scheduler.load_state_dict(ckpt.scheduler_state)
            print(f"Resumed from step {start_step}")

        for m in self._all_modules():
            m.train()

        for step in range(start_step, cfg.total_steps):
            episode = sampler.sample()

            # 1. Snapshot parameters
            theta_snapshot = {
                name: p.data.clone()
                for name, p in self._named_params()
            }

            # 2. Inner loop on support set
            inner_optimizer = torch.optim.SGD(outer_params, lr=cfg.inner_lr)
            for _ in range(cfg.inner_steps):
                inner_optimizer.zero_grad(set_to_none=True)
                w, r = self._forward_episode(episode.support_features)
                loss_dict = return_head_loss(w, r)
                loss_dict["total"].backward()
                nn.utils.clip_grad_norm_(outer_params, cfg.grad_clip)
                inner_optimizer.step()

            # 3. Query loss with adapted parameters
            outer_optimizer.zero_grad(set_to_none=True)
            w_q, r_q = self._forward_episode(episode.query_features)
            query_loss_dict = return_head_loss(w_q, r_q)
            query_loss = query_loss_dict["total"]

            # 4. Reptile outer update: theta += outer_lr * (theta_adapted - theta)
            # We achieve this by setting .grad to -(theta_adapted - theta) and stepping
            query_loss.backward()

            with torch.no_grad():
                for name, p in self._named_params():
                    if p.grad is None:
                        p.grad = theta_snapshot[name] - p.data
                    else:
                        # Blend query gradient with Reptile direction
                        reptile_dir = theta_snapshot[name] - p.data
                        p.grad.add_(reptile_dir, alpha=0.5)

            nn.utils.clip_grad_norm_(outer_params, cfg.grad_clip)
            outer_optimizer.step()
            scheduler.step()

            # Restore to updated theta (the optimizer already stepped)
            # but we need to undo the inner loop changes that went beyond
            # what the outer optimizer captured — the optimizer step on .grad
            # handles the update from the snapshot base.

            if (step + 1) % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                parts = "  ".join(f"{k}={v.item():.4f}" for k, v in query_loss_dict.items())
                print(f"step={step + 1:>7d}  query: {parts}  lr={lr_now:.2e}")

            if (step + 1) % cfg.save_every == 0:
                self._save(cfg.step_path_fn(step + 1), step + 1,
                           outer_optimizer, scheduler, best_val_loss)

        self._save(cfg.best_path, cfg.total_steps, outer_optimizer, scheduler, best_val_loss)
        print(f"Meta-learning complete at step {cfg.total_steps}.")

    def _named_params(self):
        seen: set[int] = set()
        for module_name, m in zip(
            ["backbone", "fusion", "regime_detector", "input_projection",
             "head_r", "head_s", "aux_head"],
            self._all_modules(),
        ):
            for pname, p in m.named_parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    yield f"{module_name}.{pname}", p

    def _model_state(self) -> dict:
        return {
            "backbone": self.backbone.state_dict(),
            "fusion": self.fusion.state_dict(),
            "regime_detector": self.regime_detector.state_dict(),
            "input_projection": self.input_projection.state_dict(),
            "head_return": self.head_r.state_dict(),
            "head_sharpe": self.head_s.state_dict(),
            "aux_head": self.aux_head.state_dict(),
        }

    def _save(self, path: Path, step: int, optimizer, scheduler, best: float) -> None:
        state = self._model_state()
        state["meta_step"] = step
        save_checkpoint(
            Checkpoint(
                step=step,
                model_state=state,
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
    rng = np.random.default_rng(seed=42)
    with open(path, "wb") as f:
        for _ in range(n_rows):
            row = rng.standard_normal(feature_dim).astype(np.float32)
            f.write(row_struct.pack(*row))


def _make_smoke_shards(tmp_dir: Path, n_assets: int, seq_len: int) -> list[Path]:
    paths = []
    for i in range(n_assets):
        p = tmp_dir / "train" / f"ASSET{i:02d}.f32bin"
        _write_smoke_shard(p, seq_len + 20)
        paths.append(p)
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-learning for plasticity")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument("--resume-portfolio", type=Path, default=None,
                        help="Portfolio training checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=3e-5)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/meta_learn"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.steps = args.steps if args.steps < 100_000 else 3
        args.batch_size = 2
        args.seq_len = max(args.seq_len, 64)
        args.inner_steps = min(args.inner_steps, 2)
        args.workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DEFAULT_SPEC
    spec.validate()

    from train.portfolio_train import _build_model
    backbone, fusion, regime_detector, input_projection, head_r, head_s, aux_head = (
        _build_model(spec, device)
    )

    learner = MetaLearner(
        backbone, fusion, regime_detector, input_projection,
        head_r, head_s, aux_head, spec, device,
    )

    cfg = MetaLearnConfig(
        total_steps=args.steps,
        outer_lr=args.outer_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.smoke:
        cfg.warmup_steps = 0
        cfg.log_every = 1
        cfg.eval_every = 2
        cfg.save_every = args.steps + 1  # don't save during smoke

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n_smoke = 5
            shard_paths = _make_smoke_shards(td, n_smoke, args.seq_len)
            train_loader = build_portfolio_loader(
                shard_paths, args.seq_len, args.batch_size, num_workers=0
            )
            sampler = RegimeTransitionSampler(
                train_loader, regime_detector, spec,
                support_len=cfg.episode_support_len,
                query_len=cfg.episode_query_len,
                device=device,
            )
            learner.train(sampler, cfg, resume_from=args.resume_portfolio)
            print("Smoke test passed.")
    else:
        from market_data.dataset_ingestion import enumerate_shards
        train_paths = enumerate_shards(args.data_dir, "train")
        if not train_paths:
            raise RuntimeError(f"No train shards at {args.data_dir}/train")
        train_loader = build_portfolio_loader(
            train_paths, args.seq_len, args.batch_size, num_workers=args.workers
        )
        sampler = RegimeTransitionSampler(
            train_loader, regime_detector, spec,
            support_len=cfg.episode_support_len,
            query_len=cfg.episode_query_len,
            device=device,
        )
        learner.train(sampler, cfg, resume_from=args.resume_portfolio)


if __name__ == "__main__":
    main()
