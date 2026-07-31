"""Measure the asset-identity shortcut in pretrain embeddings.

For each val window embedding, find its nearest neighbour (cosine, excluding
itself) and report how often that neighbour is the SAME asset, vs the chance
rate implied by the pool composition. High same-asset match in the trained
model (vs random-init control) = the encoder organizes by asset identity,
i.e. the contrastive task is being solved by asset re-identification and an
asset-invariance (adversarial) term has real work to do.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from core.self_supervised_pretraining import (
    SelfSupervisedPretrainer,
    build_split_loaders,
    load_checkpoint,
)
from train.pretrain import build_model
from train.spec import DEFAULT_SPEC


def build(device: torch.device) -> SelfSupervisedPretrainer:
    backbone, fusion, regime_detector, input_projection = build_model(
        DEFAULT_SPEC, device
    )
    return SelfSupervisedPretrainer(
        backbone, fusion, regime_detector, input_projection,
        d_model=DEFAULT_SPEC.d_model, n_assets=DEFAULT_SPEC.n_assets, device=device,
    )


@torch.no_grad()
def collect(pretrainer, loader, n_batches: int):
    embs, projs, assets = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        (features, _pos, _vt, asset_ids, *_rest) = pretrainer._unpack_batch(batch)
        view = pretrainer._encode_sequence(features.to(pretrainer.device))
        embs.append(F.normalize(view.float(), dim=-1).cpu())
        projs.append(pretrainer.projection_head(view).float().cpu())
        assets.append(asset_ids.cpu())
    return torch.cat(embs), torch.cat(projs), torch.cat(assets)


def nn_asset_match(z: torch.Tensor, assets: torch.Tensor) -> float:
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    sim.fill_diagonal_(-2.0)
    nn_idx = sim.argmax(dim=1)
    return (assets[nn_idx] == assets).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    p.add_argument("--batches", type=int, default=30)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_loader, val_loader, _ = build_split_loaders(
        args.data_dir,
        seq_len=DEFAULT_SPEC.train_seq_len,
        batch_size=32,
        num_workers=0,
        positive_offset=DEFAULT_SPEC.pretrain_positive_offset,
        synthetic_fraction=0.0,
    )

    for name, load in (("random-init", False), ("trained", True)):
        model = build(device)
        if load:
            ckpt = load_checkpoint(args.checkpoint)
            model._load_model_state(ckpt.model_state)
        model._all_modules.eval()
        emb, proj, assets = collect(model, val_loader, args.batches)
        # chance = probability two uniformly drawn pool members share an asset
        counts = torch.bincount(assets)
        n = assets.numel()
        chance = float(((counts * (counts - 1)).sum()) / (n * (n - 1)))
        print(
            f"{name:<12} windows={n}  chance={chance:.3f}  "
            f"nn_same_asset(backbone emb)={nn_asset_match(emb, assets):.3f}  "
            f"nn_same_asset(contrastive proj)={nn_asset_match(proj, assets):.3f}"
        )


if __name__ == "__main__":
    main()
