"""Measure whether pretrain embeddings carry CROSS-SECTIONAL return signal.

The pretrain objective is built from three terms -- instance contrastive
(match an asset to itself 64 bars later), regime CE/SupCon (which market-wide
regime we are in), and a vol auxiliary (how volatile *this* asset will be).
None of them require knowing which asset will OUTPERFORM which, yet that is
exactly what the Phase-2 portfolio heads consume. A backbone can therefore
score perfectly on every pretrain gate and still carry zero alpha.

This probe tests that directly: freeze the backbone, embed calendar-aligned
windows across the portfolio universe, fit a linear map from embedding to
cross-sectionally demeaned forward return on early dates, and measure the rank
information coefficient (IC) on later dates. A random-init control runs the
same pipeline, so the number that matters is trained IC MINUS control IC.

Interpreting the result:
  trained IC ~= control IC ~= 0  -> objective is orthogonal to the task;
                                    tuning pretrain hyperparameters cannot
                                    help, the objective itself must change
  trained IC > control IC        -> pretraining does encode relative-return
                                    structure; the Phase-2 probe was the
                                    insensitive part

Usage:
    PYTHONPATH=python python -m tools.cross_sectional_ic_probe \
        --checkpoint backbone_pretrain.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from core.self_supervised_pretraining import (
    SelfSupervisedPretrainer,
    load_checkpoint,
)
from train.feature_schema import RETURN_FEATURE_INDEX
from train.pretrain import build_model
from train.spec import DEFAULT_SPEC

TOTAL_FEATURE_DIM = 280


def load_universe(symbols_file: Path) -> list[str]:
    """Symbols only: lines carry trailing inline comments (`AAPL  # Apple ...`)."""
    out: list[str] = []
    for raw in symbols_file.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line.split()[0])
    return out


def build(device: torch.device) -> SelfSupervisedPretrainer:
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


def collect_panel(
    shard_dir: Path,
    symbols: list[str],
    seq_len: int,
    horizon: int,
    n_dates: int,
    date_stride: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build calendar-aligned (date, asset) windows and their forward returns.

    Shards are right-aligned to a shared cutoff, so indexing backwards from
    each shard's END puts every asset on the same calendar date. Indexing
    forwards would not: shards have different lengths (250-365 rows here).

    Returns (windows[D, A, seq_len, F], fwd_returns[D, A], kept_symbols).
    """
    paths = {s: shard_dir / f"{s}.f32bin" for s in symbols}
    available = {s: p for s, p in paths.items() if p.exists()}

    rows: dict[str, np.ndarray] = {}
    for sym, path in available.items():
        raw = np.memmap(path, dtype="float32", mode="r")
        rows[sym] = np.asarray(raw).reshape(-1, TOTAL_FEATURE_DIM)

    # j = bars back from the shard end at which the anchor window closes.
    # j >= horizon keeps the forward-return slice inside the shard.
    j_values = [horizon + i * date_stride for i in range(n_dates)]
    min_len = seq_len + max(j_values) + 1
    kept = sorted(s for s, r in rows.items() if r.shape[0] >= min_len)
    if len(kept) < 10:
        raise RuntimeError(
            f"only {len(kept)} assets have >= {min_len} rows; "
            "reduce --n-dates or --date-stride"
        )

    windows = np.zeros((len(j_values), len(kept), seq_len, TOTAL_FEATURE_DIM), "float32")
    fwd = np.zeros((len(j_values), len(kept)), "float64")
    for di, j in enumerate(j_values):
        for ai, sym in enumerate(kept):
            r = rows[sym]
            end = r.shape[0] - j  # anchor window closes here (exclusive)
            windows[di, ai] = r[end - seq_len : end]
            # Forward simple-sum log return over the next `horizon` bars.
            fwd[di, ai] = r[end : end + horizon, RETURN_FEATURE_INDEX].sum()
    return windows, fwd, kept


@torch.no_grad()
def embed(
    pretrainer: SelfSupervisedPretrainer,
    windows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Encode [D, A, T, F] windows to [D, A, d_model]."""
    D, A, T, F = windows.shape
    flat = torch.from_numpy(windows.reshape(D * A, T, F))
    out: list[torch.Tensor] = []
    pretrainer._all_modules.eval()
    for i in range(0, flat.shape[0], batch_size):
        chunk = flat[i : i + batch_size].to(pretrainer.device)
        with torch.autocast(
            device_type=pretrainer.device.type,
            dtype=torch.bfloat16,
            enabled=pretrainer.device.type == "cuda",
        ):
            z = pretrainer._encode_sequence(chunk)
        out.append(z.float().cpu())
    return torch.cat(out).numpy().reshape(D, A, -1)


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def _fit_ridge(
    X: np.ndarray, y: np.ndarray, alpha: float
) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ y)


def _mean_ic(
    feats: np.ndarray, y: np.ndarray, idx: np.ndarray, w: np.ndarray,
    mu: np.ndarray, sd: np.ndarray,
) -> tuple[float, np.ndarray]:
    ics = np.asarray(
        [spearman(((feats[di].astype(np.float64) - mu) / sd) @ w, y[di]) for di in idx]
    )
    return float(ics.mean()), ics


def ridge_ic(
    feats: np.ndarray, fwd: np.ndarray, train_frac: float, alphas: list[float]
) -> tuple[float, float, float, int, float]:
    """Fit features -> cross-sectionally demeaned return; IC on later dates.

    Demeaning per date removes the market factor, leaving exactly the relative
    performance the portfolio heads must rank. Splits are chronological, so no
    future information reaches the fit, and alpha is chosen on an inner slice
    of the TRAIN dates -- never on the test dates -- so a weak result cannot be
    an artefact of bad regularization, and a strong one cannot be test-set
    selection.
    """
    D = feats.shape[0]
    n_train = max(2, int(D * train_frac))
    # Dates run most-recent-first (j counts backwards), so larger index =
    # earlier in time. Fit on the earliest slice, select alpha on the middle,
    # evaluate on the most recent.
    n_sel = max(2, (D - n_train) // 3)
    inner_fit = np.arange(n_train + n_sel, D)
    inner_sel = np.arange(n_train, n_train + n_sel)
    train_idx = np.arange(n_train, D)
    test_idx = np.arange(0, n_train)

    y = fwd - fwd.mean(axis=1, keepdims=True)  # cross-sectional demean

    def _standardize(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = feats[idx].reshape(-1, feats.shape[-1]).astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = X.mean(0), X.std(0) + 1e-8
        return (X - mu) / sd, mu, sd

    # Pick alpha on the inner selection slice.
    Xif, mu_i, sd_i = _standardize(inner_fit)
    yif = y[inner_fit].reshape(-1)
    best_alpha, best_sel = alphas[0], -np.inf
    for a in alphas:
        w = _fit_ridge(Xif, yif, a)
        sel_ic, _ = _mean_ic(feats, y, inner_sel, w, mu_i, sd_i)
        if sel_ic > best_sel:
            best_sel, best_alpha = sel_ic, a

    # Refit on the full train span at the chosen alpha, evaluate on test.
    Xtr, mu, sd = _standardize(train_idx)
    w = _fit_ridge(Xtr, y[train_idx].reshape(-1), best_alpha)
    mean, ics = _mean_ic(feats, y, test_idx, w, mu, sd)
    sd_ic = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    t = mean / (sd_ic / np.sqrt(len(ics))) if sd_ic > 0 else 0.0
    return mean, sd_ic, float(t), len(ics), best_alpha


def raw_features(windows: np.ndarray) -> np.ndarray:
    """Level / trend / dispersion summary of the raw window: [D, A, 3*F].

    Gives the raw inputs a fair shot without a 64x280 flatten: last row (current
    state), window mean (level), window std (dispersion).
    """
    last = windows[:, :, -1, :]
    mean = windows.mean(axis=2)
    std = windows.std(axis=2)
    return np.concatenate([last, mean, std], axis=-1)


def momentum_ic(windows: np.ndarray, fwd: np.ndarray, train_frac: float) -> float:
    """Naive control: rank by trailing return. Any real signal must beat this."""
    D = windows.shape[0]
    n_train = max(2, int(D * train_frac))
    y = fwd - fwd.mean(axis=1, keepdims=True)
    ics = [
        spearman(windows[di, :, :, RETURN_FEATURE_INDEX].sum(axis=1), y[di])
        for di in range(0, n_train)
    ]
    return float(np.mean(ics))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    p.add_argument("--split", default="validation")
    p.add_argument("--symbols-file", type=Path, default=Path("portfolio_symbols.txt"))
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n-dates", type=int, default=60)
    p.add_argument("--date-stride", type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e1, 1e2, 1e3, 1e4, 1e5],
        help="ridge grid; chosen on an inner slice of train dates",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--skip-model",
        action="store_true",
        help="raw-feature arm only (no GPU, seconds)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = load_universe(args.symbols_file)
    windows, fwd, kept = collect_panel(
        args.data_dir / args.split,
        symbols,
        args.seq_len,
        args.horizon,
        args.n_dates,
        args.date_stride,
    )
    print(
        f"panel: {windows.shape[0]} dates x {len(kept)} assets  "
        f"(seq_len={args.seq_len}, horizon={args.horizon}, device={device})"
    )
    mom = momentum_ic(windows, fwd, args.train_frac)
    print(f"{'trailing-return control':<24} IC={mom:+.4f}")

    results = {}
    # Raw-feature arm: is the signal in the INPUTS at all? This separates
    # "pretraining discards cross-sectional information" from "the target is
    # not predictable from this feature set", which need opposite responses.
    mean, sd, t, n, a = ridge_ic(
        raw_features(windows), fwd, args.train_frac, args.alphas
    )
    results["raw-features"] = mean
    print(
        f"{'raw features':<24} IC={mean:+.4f}  sd={sd:.4f}  t={t:+.2f}  "
        f"({n} test dates, alpha={a:g})"
    )

    if not args.skip_model:
        for name, load in (("random-init", False), ("trained", True)):
            model = build(device)
            if load:
                model._load_model_state(load_checkpoint(args.checkpoint).model_state)
            emb = embed(model, windows, args.batch_size)
            mean, sd, t, n, a = ridge_ic(emb, fwd, args.train_frac, args.alphas)
            results[name] = mean
            print(
                f"{name:<24} IC={mean:+.4f}  sd={sd:.4f}  t={t:+.2f}  "
                f"({n} test dates, alpha={a:g})"
            )
            del model, emb
            if device.type == "cuda":
                torch.cuda.empty_cache()

        delta = results["trained"] - results["random-init"]
        print(f"\ndelta (trained - random) = {delta:+.4f}")

    print(
        "\nraw ~= 0  -> target not predictable from these features; no pretrain\n"
        "             objective change helps (revisit horizon/features/target)\n"
        "raw >  0 and trained ~= 0 -> pretraining DISCARDS cross-sectional\n"
        "             information that is present in the inputs; fix the objective"
    )


if __name__ == "__main__":
    main()
