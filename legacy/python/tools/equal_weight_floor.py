"""Equal-weight DSR floor on the portfolio validation set.

Mirrors PortfolioTrainer.evaluate() exactly — same val shard selection, loader,
cold-start trim, loss config, and sharpe/max-dd computation — but with constant
uniform 1/N weights instead of model output. Every Phase-2 canary or training
run must beat this floor (and the random-backbone control) to mean anything.

Usage:
    PYTHONPATH=python python -m tools.equal_weight_floor [--data-dir data/mmaps]
"""
import argparse
import math
from pathlib import Path

import torch

from market_data.dataset_ingestion import enumerate_shards
from optimization.losses import return_head_loss
from train.portfolio_train import (
    build_portfolio_loader,
    load_portfolio_symbols,
    prepare_return_targets,
    select_portfolio_shards,
)

SEQ_LEN = 64
BATCH = 4
STATE_WARMUP_STEPS = 8
N_ASSETS = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument(
        "--symbols-file", type=Path, default=Path("portfolio_symbols.txt")
    )
    args = parser.parse_args()

    symbols = load_portfolio_symbols(args.symbols_file)
    val_paths = select_portfolio_shards(
        enumerate_shards(args.data_dir, "validation"),
        max_assets=N_ASSETS,
        required_assets=N_ASSETS,
        allowed_symbols=set(symbols),
    )
    loader = build_portfolio_loader(
        val_paths, SEQ_LEN, BATCH, shuffle=False, num_workers=0
    )

    totals = {"total": 0.0, "dsr": 0.0}
    series: list[torch.Tensor] = []
    n = 0
    with torch.no_grad():
        for _features, shifted_returns in loader:
            returns = prepare_return_targets(shifted_returns)  # (B, T, N)
            b, t, n_assets = returns.shape
            weights = torch.full((b, t, n_assets), 1.0 / n_assets)
            weights = weights[:, STATE_WARMUP_STEPS:, :]
            returns = returns[:, STATE_WARMUP_STEPS:, :]
            series.append((weights * returns).sum(dim=-1).reshape(-1))
            loss_dict = return_head_loss(
                weights,
                returns,
                pred_vol=None,
                lambda_ret=0.1,
                lambda_turn=0.1,
                dsr_eta=0.01,
            )
            totals["total"] += loss_dict["total"].item()
            totals["dsr"] += loss_dict["dsr"].item()
            n += 1

    rets = torch.cat(series, dim=0)
    mu = float(rets.mean().item())
    sigma = float(rets.std(unbiased=False).clamp(min=1e-6).item())
    sharpe = (mu / sigma) * math.sqrt(252.0)
    equity = torch.exp(torch.cumsum(rets, dim=0))
    running_max = torch.cummax(equity, dim=0).values.clamp(min=1e-6)
    max_dd = float((1.0 - equity / running_max).max().item())

    print(
        f"EQUAL-WEIGHT FLOOR  batches={n}  "
        f"val total={totals['total'] / max(n, 1):.4f}  "
        f"val dsr={totals['dsr'] / max(n, 1):.4f}  "
        f"val_sharpe={sharpe:.3f}  val_max_dd={max_dd:.3f}"
    )


if __name__ == "__main__":
    main()
