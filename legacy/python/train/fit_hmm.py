"""Pre-fit the 4-state HMM on historical market data for regime initialization.

Spec §7.2: the RegimeDetector is bootstrapped with a pre-fit HMM so that regime
posteriors are meaningful at the start of training, rather than random.

The 4 regimes are:
  0 — Bull      (positive trend, low volatility)
  1 — Bear      (negative trend, elevated volatility)
  2 — HighVol   (high volatility, mean-reverting)
  3 — LowVol    (low volatility, low trend magnitude)

Observation features (5-dim daily vector derived from SPY):
  [daily_return, realized_vol_20d, trend_20d, vix_level_z, breadth_z]

The fitted HMM is saved as a .npz file containing:
  transmat   : (4, 4)  transition probability matrix
  means      : (4, 5)  Gaussian emission means per state
  covars     : (4, 5)  Gaussian emission diagonal covariances per state
  startprob  : (4,)    initial state distribution

Usage:
    python -m python.train.fit_hmm \\
        --shard-dir   data/mmaps/train \\
        --output      models/hmm/regime_hmm.npz \\
        --n-iter      200

If --shard-dir is not available (no training data yet), a sensible default
HMM is saved using hand-crafted regime statistics from literature.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_REGIMES = 4
OBS_DIM   = 5   # [daily_return, realized_vol_20d, trend_20d, vix_level_z, breadth_z]

# ── Default regime statistics (from literature / domain knowledge) ────────────
# Used when no historical shard data is available.
#
# Means: [daily_ret, realized_vol, trend_20d, vix_z, breadth_z]
_DEFAULT_MEANS = np.array([
    [ 0.0008,  0.008,  0.010,  -0.5,  0.4],   # 0: Bull
    [-0.0012,  0.020, -0.012,   0.8, -0.5],   # 1: Bear
    [-0.0002,  0.025, -0.002,   1.2, -0.2],   # 2: HighVol
    [ 0.0003,  0.005,  0.003,  -0.8,  0.1],   # 3: LowVol
], dtype=np.float64)

_DEFAULT_COVARS = np.array([
    [1e-5, 1e-4, 1e-4, 0.25, 0.04],   # 0: Bull
    [4e-5, 4e-4, 4e-4, 0.64, 0.16],   # 1: Bear
    [6e-5, 6e-4, 4e-4, 1.00, 0.09],   # 2: HighVol
    [5e-6, 2e-5, 2e-5, 0.16, 0.01],   # 3: LowVol
], dtype=np.float64)

# Transition matrix: regimes are persistent; transitions favour staying
_DEFAULT_TRANSMAT = np.array([
    [0.96, 0.02, 0.01, 0.01],  # Bull → stays Bull most of the time
    [0.02, 0.93, 0.04, 0.01],  # Bear
    [0.03, 0.06, 0.88, 0.03],  # HighVol (more transient)
    [0.03, 0.01, 0.02, 0.94],  # LowVol
], dtype=np.float64)

_DEFAULT_STARTPROB = np.array([0.40, 0.20, 0.15, 0.25], dtype=np.float64)


# ── Feature extraction from shard files ──────────────────────────────────────

def _extract_obs_from_shards(shard_dir: Path) -> np.ndarray | None:
    """
    Load per-asset shard files and extract SPY-like aggregate daily features.

    Shards store rows of 280 floats (184 per-asset + 96 market-wide).
    We use the market-wide block (last 96 dims) to derive daily observations.

    Market-wide block layout (see feature_schema.py MARKET_BLOCKS):
      [0:12]  volatility_sentiment
      [12:24] yield_curve
      [24:36] fred_macro
      [36:52] sector_industry
      [52:72] world_indices
      [72:84] breadth_flow
      [84:96] rss_news_aggregate

    We downsample minute-resolution shards to daily by averaging 390-step windows.
    """
    from python.train.feature_schema import PER_ASSET_FEATURE_DIM, MARKET_FEATURE_DIM

    total_dim = PER_ASSET_FEATURE_DIM + MARKET_FEATURE_DIM  # 280
    steps_per_day = 390

    all_obs: list[np.ndarray] = []

    for path in sorted(shard_dir.glob("*.f32bin")):
        data = np.memmap(path, dtype="float32", mode="r")
        n_rows = len(data) // total_dim
        if n_rows < steps_per_day:
            continue
        data = data[: n_rows * total_dim].reshape(n_rows, total_dim)

        # Market-wide block
        market = data[:, PER_ASSET_FEATURE_DIM:].astype(np.float64)

        # Aggregate to daily windows
        n_days = n_rows // steps_per_day
        for d in range(n_days):
            window = market[d * steps_per_day : (d + 1) * steps_per_day]
            # Use mean of window as daily feature proxy
            day_feats = window.mean(axis=0)  # (96,)

            # Extract 5-dim observation:
            # [0] daily_return     ~ mean of vol_sentiment[0] (SPY return-like)
            # [1] realized_vol_20d ~ std of vol_sentiment[0] within window
            # [2] trend_20d        ~ mean of breadth_flow[0]
            # [3] vix_level_z      ~ vol_sentiment[1] (VIX proxy)
            # [4] breadth_z        ~ breadth_flow[1]
            daily_ret  = float(day_feats[0])
            realized_v = float(window[:, 0].std())
            trend_20d  = float(day_feats[72])   # breadth_flow start
            vix_z      = float(day_feats[1])
            breadth_z  = float(day_feats[73])
            all_obs.append(np.array([daily_ret, realized_v, trend_20d, vix_z, breadth_z]))

    if not all_obs:
        return None
    return np.stack(all_obs, axis=0)


# ── HMM fitting ──────────────────────────────────────────────────────────────

def fit_hmm(obs: np.ndarray, n_iter: int = 200) -> dict[str, np.ndarray]:
    """Fit a diagonal-covariance Gaussian HMM via Baum-Welch."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError(
            "hmmlearn not installed — run: pip install hmmlearn"
        ) from e

    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-4,
        random_state=42,
        init_params="stmc",  # init startprob, transmat, means, covars from data
    )
    model.fit(obs)
    log.info("HMM converged=%s  score=%.2f", model.monitor_.converged, model.score(obs))

    return {
        "transmat":   model.transmat_,
        "means":      model.means_,
        "covars":     model.covars_,
        "startprob":  model.startprob_,
    }


def default_hmm() -> dict[str, np.ndarray]:
    """Return the hand-crafted default HMM when no training data is available."""
    return {
        "transmat":  _DEFAULT_TRANSMAT,
        "means":     _DEFAULT_MEANS,
        "covars":    _DEFAULT_COVARS,
        "startprob": _DEFAULT_STARTPROB,
    }


def save_hmm(params: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **params)
    log.info(
        "Saved HMM → %s  transmat:\n%s",
        output_path,
        np.array2string(params["transmat"], precision=3),
    )


def load_hmm(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path))
    return {k: data[k] for k in ("transmat", "means", "covars", "startprob")}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Pre-fit 4-state HMM for regime initialization")
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("data/mmaps/train"),
        help="Directory containing *.f32bin shard files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/hmm/regime_hmm.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Maximum Baum-Welch iterations",
    )
    args = parser.parse_args()

    obs = None
    if args.shard_dir.exists():
        log.info("Extracting observations from %s …", args.shard_dir)
        obs = _extract_obs_from_shards(args.shard_dir)

    if obs is not None and len(obs) >= N_REGIMES * 10:
        log.info("Fitting HMM on %d daily observations …", len(obs))
        params = fit_hmm(obs, n_iter=args.n_iter)
    else:
        log.warning(
            "Insufficient data (%s observations) — using default HMM parameters.",
            len(obs) if obs is not None else 0,
        )
        params = default_hmm()

    save_hmm(params, args.output)


if __name__ == "__main__":
    main()
