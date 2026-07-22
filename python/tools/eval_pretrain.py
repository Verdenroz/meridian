"""Pretrain checkpoint validation against a random-init control.

Runs the same evaluation on the trained checkpoint and a randomly-initialized
model, on real shards only, plus naive forward-vol baselines (persistence,
EWMA). A checkpoint only demonstrates learning where it beats BOTH the random
control and the naive baselines — contrastive accuracy alone proves nothing
(a random encoder saturates it via input similarity).

Usage:
    PYTHONPATH=python python -m tools.eval_pretrain \
        --checkpoint checkpoints/pretrain/backbone_pretrain.pt \
        --data-dir data/mmaps --batches 50
"""

import argparse
from pathlib import Path

import torch

from core.self_supervised_pretraining import (
    SelfSupervisedPretrainer,
    TrainConfig,
    build_split_loaders,
    load_checkpoint,
)
from train.pretrain import build_model
from train.spec import DEFAULT_SPEC


def _build_pretrainer(device: torch.device) -> SelfSupervisedPretrainer:
    backbone, fusion, regime_detector, input_projection = build_model(
        DEFAULT_SPEC, device
    )
    return SelfSupervisedPretrainer(
        backbone,
        fusion,
        regime_detector,
        input_projection,
        d_model=DEFAULT_SPEC.d_model,
        n_assets=DEFAULT_SPEC.n_assets,
        device=device,
    )


def _supcon_grad_probe(
    pretrainer: SelfSupervisedPretrainer, train_loader, cfg: TrainConfig
) -> None:
    """Verify the SupCon term produces gradient in the backbone (issue #2)."""
    batch = next(iter(train_loader))
    (
        features,
        positives,
        vol_targets,
        asset_ids,
        regime_ids,
        sector_ids,
        sample_weights,
        time_indices,
    ) = pretrainer._unpack_batch(batch)
    if regime_ids is None:
        print("SUPCON PROBE: FAIL — regime_ids is None in training batches; "
              "the SupCon term is silently skipped during training.")
        return
    pretrainer._all_modules.zero_grad(set_to_none=True)
    losses = pretrainer._forward_losses(
        features,
        positives,
        vol_targets=vol_targets,
        asset_ids=asset_ids,
        regime_ids=regime_ids,
        sector_ids=sector_ids,
        sample_weights=sample_weights,
        time_indices=time_indices,
        temperature=cfg.contrastive_temperature,
        contrastive_weight=0.0,
        prediction_weight=0.0,
        supcon_weight=1.0,
        adversarial_lambda=0.0,
        return_aux=True,
    )
    losses["total"].backward()
    grad_sq = 0.0
    for p in pretrainer.backbone.parameters():
        if p.grad is not None:
            grad_sq += float(p.grad.detach().pow(2).sum())
    grad_norm = grad_sq**0.5
    verdict = "PASS" if grad_norm > 0 else "FAIL"
    print(
        f"SUPCON PROBE: {verdict} — supcon={losses['supcon'].item():.4f}  "
        f"backbone grad norm from supcon-only backward: {grad_norm:.3e}"
    )
    pretrainer._all_modules.zero_grad(set_to_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/pretrain/backbone_pretrain.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--positive-offset",
        type=int,
        default=DEFAULT_SPEC.pretrain_positive_offset,
        help="TCP positive offset for eval pairs (default: current spec value)",
    )
    parser.add_argument("--skip-supcon-probe", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  checkpoint={args.checkpoint}  data={args.data_dir}")

    train_loader, val_loader, _ = build_split_loaders(
        args.data_dir,
        seq_len=DEFAULT_SPEC.train_seq_len,
        batch_size=args.batch_size,
        num_workers=0,
        positive_offset=args.positive_offset,
        synthetic_fraction=0.0,
    )
    cfg = TrainConfig(eval_max_batches=args.batches)

    results: dict[str, dict[str, float]] = {}
    for name, load in (("random-init", False), ("pretrained", True)):
        pretrainer = _build_pretrainer(device)
        if load:
            ckpt = load_checkpoint(args.checkpoint)
            pretrainer._load_model_state(ckpt.model_state)
            print(f"loaded checkpoint step={ckpt.step}")
        results[name] = pretrainer.evaluate(val_loader, cfg, max_batches=args.batches)
        if load and not args.skip_supcon_probe:
            _supcon_grad_probe(pretrainer, train_loader, cfg)

    keys = [
        "contrastive_acc",
        "contrastive",
        "supcon",
        "prediction_r2",
        "baseline_r2_persistence",
        "baseline_r2_ewma",
        "prediction_delta_r2",
        "total",
    ]
    print(f"\n{'metric':<28}{'random-init':>14}{'pretrained':>14}")
    for k in keys:
        r = results["random-init"].get(k, float("nan"))
        p = results["pretrained"].get(k, float("nan"))
        print(f"{k:<28}{r:>14.4f}{p:>14.4f}")

    delta = results["pretrained"].get("prediction_delta_r2", 0.0)
    print(
        f"\nVERDICT: prediction ΔR² vs best naive baseline = {delta:+.4f} "
        f"({'backbone beats naive vol baselines' if delta > 0 else 'backbone does NOT beat naive vol baselines'})"
    )


if __name__ == "__main__":
    main()
