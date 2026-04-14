"""Financial Diffusion Augmentation (FDA) offline pipeline

This module provides:
1. A lightweight regime-conditioned diffusion trainer/generator
2. A stylized-fact validation gate
3. Synthetic shard writing compatible with f32bin training readers
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.self_supervised_pretraining import infer_regime_label


@dataclass
class FDAConfig:
    window_len: int = 64
    feature_dim: int = 280
    diffusion_steps: int = 100
    hidden_dim: int = 256
    lr: float = 3e-4
    epochs: int = 5
    batch_size: int = 32
    device: str = "cpu"
    generation_batch_size: int = 64
    target_valid_windows: int = 1_000
    gate_min_pass_ratio: float = 0.85
    max_generation_rounds: int = 1_000


@dataclass
class ReferenceStats:
    mean_corr_proxy: float
    tail_comovement: float


@dataclass
class StylizedFactReport:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]


def _safe_std(x: np.ndarray) -> float:
    return float(max(np.std(x), 1e-8))


def _autocorr_lag1(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    den = np.linalg.norm(x0) * np.linalg.norm(x1)
    if den <= 1e-8:
        return 0.0
    return float(np.dot(x0, x1) / den)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError("correlation inputs must have same length")
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = np.linalg.norm(a0) * np.linalg.norm(b0)
    if den <= 1e-8:
        return 0.0
    return float(np.dot(a0, b0) / den)


def _compute_rsi_like(close_returns: np.ndarray, period: int = 14) -> np.ndarray:
    if close_returns.size < period + 1:
        return np.full_like(close_returns, 50.0, dtype=np.float64)
    gains = np.maximum(close_returns, 0.0)
    losses = -np.minimum(close_returns, 0.0)
    rsi = np.full(close_returns.shape[0], 50.0, dtype=np.float64)
    for i in range(period, close_returns.shape[0]):
        avg_gain = gains[i - period + 1 : i + 1].mean()
        avg_loss = losses[i - period + 1 : i + 1].mean()
        rs = avg_gain / max(avg_loss, 1e-8)
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_reference_stats(real_windows: np.ndarray) -> ReferenceStats:
    # Returns proxy: close return at index 3.
    r = real_windows[:, :, 3].reshape(-1)
    # Market proxy: first market feature at index 184.
    m = real_windows[:, :, 184].reshape(-1)

    mean_corr_proxy = _corr(r, m)
    r_sigma = _safe_std(r)
    m_sigma = _safe_std(m)
    tail = (
        (np.abs(r) > 2 * r_sigma)
        & (np.abs(m) > 2 * m_sigma)
        & (np.sign(r) == np.sign(m))
    ).mean()
    return ReferenceStats(
        mean_corr_proxy=float(mean_corr_proxy), tail_comovement=float(tail)
    )


def validate_stylized_facts(
    window: np.ndarray, reference: ReferenceStats
) -> StylizedFactReport:
    r = window[:, 3]
    vol = np.abs(window[:, 4]) + 1e-6
    mkt = window[:, 184] if window.shape[1] > 184 else np.zeros_like(r)

    mu = float(r.mean())
    std = _safe_std(r)
    z = (r - mu) / std
    excess_kurtosis = float(np.mean(z**4) - 3.0)
    skewness = float(np.mean(z**3))
    vol_cluster = _autocorr_lag1(r**2)
    return_ac = _autocorr_lag1(r)
    leverage = _corr(r[:-1], np.abs(r[1:])) if r.shape[0] > 2 else 0.0
    vol_corr = _corr(np.log(vol), np.abs(r))
    corr_proxy = _corr(r, mkt)
    r_sigma = _safe_std(r)
    m_sigma = _safe_std(mkt)
    tail_move = float(
        (
            (np.abs(r) > 2 * r_sigma)
            & (np.abs(mkt) > 2 * m_sigma)
            & (np.sign(r) == np.sign(mkt))
        ).mean()
    )

    rsi_comp = _compute_rsi_like(r)
    # Technical block starts at index 10; use a bounded proxy for indicator consistency.
    if window.shape[1] > 10:
        rsi_feat = np.clip(window[:, 10] * 50.0 + 50.0, 0.0, 100.0)
    else:
        rsi_feat = np.full_like(rsi_comp, 50.0)
    indicator_mae = float(np.mean(np.abs(rsi_comp - rsi_feat)) / 100.0)

    checks = {
        "fat_tails": excess_kurtosis > 1.0,
        "volatility_clustering": vol_cluster > 0.10,
        "negative_skewness": skewness < -0.05,
        "no_return_autocorr": abs(return_ac) < 0.10,
        "leverage_effect": leverage < -0.05,
        "volume_vol_correlation": vol_corr > 0.15,
        "cross_asset_correlation_proxy": abs(corr_proxy - reference.mean_corr_proxy)
        <= 0.10,
        "tail_comovement_proxy": abs(tail_move - reference.tail_comovement) <= 0.05,
        "indicator_consistency": indicator_mae < 0.10,
    }
    metrics = {
        "excess_kurtosis": excess_kurtosis,
        "vol_cluster": vol_cluster,
        "skewness": skewness,
        "return_autocorr": return_ac,
        "leverage": leverage,
        "volume_vol_corr": vol_corr,
        "corr_proxy": corr_proxy,
        "tail_move": tail_move,
        "indicator_mae": indicator_mae,
    }
    score = float(sum(1 for ok in checks.values() if ok) / len(checks))
    metrics["score"] = score
    passed = score >= (2.0 / 3.0)
    return StylizedFactReport(passed=passed, checks=checks, metrics=metrics)


class FDADiffusionModel(nn.Module):
    """Lightweight regime-conditioned diffusion denoiser over (T, F) windows."""

    def __init__(
        self,
        feature_dim: int = 280,
        hidden_dim: int = 256,
        diffusion_steps: int = 100,
        n_regimes: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.diffusion_steps = diffusion_steps
        self.time_emb = nn.Embedding(diffusion_steps + 1, hidden_dim)
        self.regime_emb = nn.Embedding(n_regimes, hidden_dim)
        self.in_proj = nn.Linear(feature_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, diffusion_steps + 1))
        alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, regime: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, T, F), t: (B,), regime: (B,)
        time_cond = self.time_emb(t)[:, None, :]
        regime_cond = self.regime_emb(regime)[:, None, :]
        h = self.in_proj(x) + time_cond + regime_cond
        return self.ff(h)


def load_windows_from_shards(
    shard_paths: list[Path],
    window_len: int = 64,
    stride: int = 4,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    regimes: list[int] = []
    malformed = 0
    too_short = 0
    for path in shard_paths:
        raw = np.memmap(path, dtype="float32", mode="r")
        if len(raw) % 280 != 0:
            malformed += 1
            continue
        rows = raw.reshape(-1, 280)

        n = rows.shape[0]
        if n <= window_len:
            too_short += 1
            continue
        for start in range(0, n - window_len, stride):
            w = rows[start : start + window_len]
            windows.append(w.astype(np.float32, copy=True))
            regimes.append(infer_regime_label(w))
            if max_windows is not None and len(windows) >= max_windows:
                break
        if max_windows is not None and len(windows) >= max_windows:
            break
    if not windows:
        raise RuntimeError(
            "no windows could be loaded from shard paths "
            f"(checked={len(shard_paths)}, malformed={malformed}, too_short={too_short}); "
            "re-run collection to refresh shards"
        )
    return np.stack(windows, axis=0), np.asarray(regimes, dtype=np.int64)


def train_diffusion_model(
    windows: np.ndarray,
    regimes: np.ndarray,
    cfg: FDAConfig,
) -> FDADiffusionModel:
    device = torch.device(cfg.device)
    x = torch.from_numpy(windows).to(device)
    r = torch.from_numpy(regimes).to(device)
    model = FDADiffusionModel(
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        diffusion_steps=cfg.diffusion_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    n = x.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            x0 = x[idx]
            regime = r[idx]
            t = torch.randint(1, cfg.diffusion_steps + 1, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            ab = model.alpha_bar[t][:, None, None]
            noisy = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
            pred = model(noisy, t, regime)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def sample_diffusion_windows(
    model: FDADiffusionModel,
    n_windows: int,
    window_len: int,
    feature_dim: int,
    regimes: np.ndarray | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    dev = device or next(model.parameters()).device
    x_t = torch.randn(n_windows, window_len, feature_dim, device=dev)
    if regimes is None:
        regime = torch.randint(0, 4, (n_windows,), device=dev)
    else:
        regime = torch.from_numpy(regimes.astype(np.int64)).to(dev)

    model.eval()
    with torch.no_grad():
        for step in range(model.diffusion_steps, 0, -1):
            t = torch.full((n_windows,), step, dtype=torch.long, device=dev)
            eps_theta = model(x_t, t, regime)
            alpha_t = model.alphas[t][:, None, None]
            alpha_bar_t = model.alpha_bar[t][:, None, None]
            beta_t = model.betas[t][:, None, None]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            )
            if step > 1:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean
    return x_t.cpu().numpy().astype(np.float32)


def generate_valid_synthetic_windows(
    model: FDADiffusionModel,
    reference: ReferenceStats,
    cfg: FDAConfig,
    window_len: int = 64,
    feature_dim: int = 280,
) -> tuple[np.ndarray, list[StylizedFactReport]]:
    accepted: list[np.ndarray] = []
    reports: list[StylizedFactReport] = []
    rounds = 0

    while len(accepted) < cfg.target_valid_windows:
        rounds += 1
        if rounds > cfg.max_generation_rounds:
            raise RuntimeError(
                "FDA generation exceeded max rounds before reaching target valid windows"
            )
        candidates = sample_diffusion_windows(
            model,
            n_windows=cfg.generation_batch_size,
            window_len=window_len,
            feature_dim=feature_dim,
        )
        pass_count = 0
        for w in candidates:
            rep = validate_stylized_facts(w, reference)
            reports.append(rep)
            if rep.passed:
                accepted.append(w)
                pass_count += 1
                if len(accepted) >= cfg.target_valid_windows:
                    break
        if pass_count / max(len(candidates), 1) < cfg.gate_min_pass_ratio:
            # Keep generating; this mirrors gate rejection behavior rather than failing hard.
            continue
    return np.stack(accepted, axis=0), reports


def write_synthetic_shards(
    windows: np.ndarray,
    output_dir: Path,
    split: str = "train",
    prefix: str = "FDA_SYNTH",
) -> list[Path]:
    if windows.ndim != 3:
        raise ValueError("windows must have shape (N, T, F)")
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for i, w in enumerate(windows):
        path = split_dir / f"{prefix}_{i:05d}.f32bin"
        w.astype(np.float32).tofile(path)
        out_paths.append(path)
    return out_paths


def write_fda_manifest(
    output_dir: Path,
    cfg: FDAConfig,
    reference: ReferenceStats,
    reports: list[StylizedFactReport],
    shard_paths: list[Path],
) -> Path:
    pass_total = sum(1 for r in reports if r.passed)
    manifest = {
        "config": asdict(cfg),
        "reference": asdict(reference),
        "candidate_windows": len(reports),
        "accepted_windows": pass_total,
        "acceptance_rate": float(pass_total / max(len(reports), 1)),
        "shards_written": [str(p) for p in shard_paths],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fda_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDA offline generation pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("data/mmaps/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mmaps_synth"))
    parser.add_argument("--max-real-windows", type=int, default=4096)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-valid-windows", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = FDAConfig(
        window_len=args.window_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_valid_windows=args.target_valid_windows,
        device=args.device,
    )
    shard_paths = sorted(args.input_dir.glob("*.f32bin"))
    if not shard_paths:
        raise RuntimeError(f"no shards found under {args.input_dir}")

    windows, regimes = load_windows_from_shards(
        shard_paths=shard_paths,
        window_len=cfg.window_len,
        stride=args.stride,
        max_windows=args.max_real_windows,
    )
    ref = compute_reference_stats(windows)
    model = train_diffusion_model(windows, regimes, cfg)
    synth, reports = generate_valid_synthetic_windows(
        model=model,
        reference=ref,
        cfg=cfg,
        window_len=cfg.window_len,
        feature_dim=cfg.feature_dim,
    )
    out_paths = write_synthetic_shards(synth, args.output_dir, split="train")
    manifest = write_fda_manifest(args.output_dir, cfg, ref, reports, out_paths)
    print(f"Generated {len(out_paths)} FDA shards. Manifest: {manifest}")


if __name__ == "__main__":
    main()
