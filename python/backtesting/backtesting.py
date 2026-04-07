"""Historical backtesting framework — Section 12 of meridian-spec.md.

Integrates with finance-query's Rust backtesting engine via PyO3 bindings
to evaluate Meridian's portfolio weights against standard benchmarks.

Usage:
    # Full backtest against benchmarks
    python -m backtesting.backtesting \
        --checkpoint checkpoints/portfolio/return_head_best.pt \
        --data-dir data/mmaps --head return

    # Smoke test with synthetic weights
    python -m backtesting.backtesting --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import meridian_py
import numpy as np
import torch
import torch.nn.functional as F
from train.spec import DEFAULT_SPEC, MeridianSpec

# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class BacktestSpec:
    """Section 12 backtesting parameters."""

    initial_capital: float = 1_000_000.0
    commission_bps: float = 10.0  # 10 basis points
    slippage_bps: float = 5.0  # 5 basis points
    bars_per_year: float = 252 * 390  # minute bars
    max_positions: int = 100
    mc_simulations: int = 1000
    mc_seed: int = 42


# ── Candle loading ───────────────────────────────────────────────────────────

OHLCV_INDICES = {
    "timestamp": 0,
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
}


def load_candles_from_shards(
    data_dir: Path,
    symbols: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    """Load OHLCV candle arrays from f32bin shards.

    Returns:
        candles: {symbol: (T, 6) float64 array of [ts, O, H, L, C, V]}
        min_bars: shortest shard length across all symbols
    """
    feature_dim = 280
    candles = {}
    min_bars = float("inf")

    for sym in symbols:
        path = data_dir / f"{sym}.f32bin"
        if not path.exists():
            continue

        raw = np.memmap(path, dtype="float32", mode="r")
        n = len(raw) // feature_dim
        data = raw[: n * feature_dim].reshape(n, feature_dim)
        min_bars = min(min_bars, n)

        # Feature indices 0-5 are the OHLCV block in the shard layout.
        # Timestamp is synthetic (minute-bar indexed from epoch).
        arr = np.zeros((n, 6), dtype=np.float64)
        arr[:, 0] = np.arange(n, dtype=np.float64) * 60 + 1609459200  # synthetic ts
        arr[:, 1] = data[:, 1].astype(np.float64)  # open
        arr[:, 2] = data[:, 2].astype(np.float64)  # high
        arr[:, 3] = data[:, 3].astype(np.float64)  # low
        arr[:, 4] = data[:, 4].astype(np.float64)  # close (index 4 in shard)
        arr[:, 5] = data[:, 5].astype(np.float64)  # volume

        candles[sym] = arr

    # Tail-align to shortest common length
    min_bars = int(min_bars)
    candles = {sym: arr[-min_bars:] for sym, arr in candles.items()}

    return candles, min_bars


# ── Model weight generation ──────────────────────────────────────────────────


def load_model_and_produce_weights(
    checkpoint_path: Path,
    data_dir: Path,
    head: str = "return",
    spec: MeridianSpec = DEFAULT_SPEC,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Load a trained Meridian model and compute portfolio weights.

    Returns:
        weights: (T, N) float64 array of target portfolio weights
        symbols: list of N ticker strings
        candles: dict mapping symbol to (T, 6) OHLCV array
    """
    from core.cross_asset_fusion import CrossAssetFusion
    from core.regime_aware_backbone import (
        InputProjection,
        RAGDNBackbone,
        RegimeDetector,
    )
    from optimization.portfolio_heads import MeridianR, MeridianS

    # Discover symbols from available shard files
    shard_files = sorted(data_dir.glob("*.f32bin"))
    symbols = [p.stem for p in shard_files]
    if not symbols:
        raise FileNotFoundError(f"no .f32bin shards in {data_dir}")

    candles, min_bars = load_candles_from_shards(data_dir, symbols)
    symbols = [s for s in symbols if s in candles]
    N = len(symbols)

    # Load shard features for model inference
    feature_dim = 280
    arrays = []
    for sym in symbols:
        path = data_dir / f"{sym}.f32bin"
        raw = np.memmap(path, dtype="float32", mode="r")
        n = len(raw) // feature_dim
        data = raw[: n * feature_dim].reshape(n, feature_dim)
        arrays.append(data[-min_bars:])

    # (N, T, F) aligned features
    features = np.stack(arrays, axis=0)

    # Build model
    input_proj = InputProjection(spec)
    regime_det = RegimeDetector(spec)
    backbone = RAGDNBackbone(spec)
    fusion = CrossAssetFusion(spec)
    head_module = (
        MeridianR(spec.d_model) if head == "return" else MeridianS(spec.d_model)
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = (
        ckpt
        if isinstance(ckpt, dict) and "model_state_dict" not in ckpt
        else ckpt.get("model_state_dict", ckpt)
    )

    # Partial load: match keys across components
    for module, prefix in [
        (input_proj, "input_proj."),
        (regime_det, "regime_det."),
        (backbone, "backbone."),
        (fusion, "fusion."),
        (head_module, "head."),
    ]:
        sub_state = {
            k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)
        }
        if sub_state:
            module.load_state_dict(sub_state, strict=False)

    for m in [input_proj, regime_det, backbone, fusion, head_module]:
        m.to(device).eval()

    # Produce weights in chunks to avoid OOM
    seq_len = spec.train_seq_len
    all_weights = []
    feat_tensor = torch.from_numpy(features).to(device)  # (N, T, F)

    with torch.no_grad():
        for t in range(0, min_bars - seq_len, seq_len):
            chunk = feat_tensor[:, t : t + seq_len, :].unsqueeze(0)  # (1, N, T, F)
            B, Nc, T, Fd = chunk.shape

            # Dummy inputs for components that need them
            market_f = chunk[:, 0, :, -96:]  # (B, T, 96) market features
            portfolio_s = torch.zeros(B, spec.portfolio_state_dim, device=device)

            regime_logits = regime_det(
                market_f[:, -1, :],
                portfolio_s[:, :3] if portfolio_s.shape[-1] >= 3 else portfolio_s,
            )
            regime_probs = F.softmax(regime_logits, dim=-1)

            proj = input_proj(chunk, regime_probs)  # (B, N, D)
            backbone_out = backbone(proj, regime_probs)  # (B, N, D)
            fused = fusion(backbone_out, regime_probs)  # (B, N, D)
            w = head_module(fused)  # (B, N)
            all_weights.append(w.cpu().numpy())

    weights = np.concatenate(all_weights, axis=0)  # (n_chunks, N)
    # Repeat each chunk's weights for seq_len bars (constant within chunk)
    weights_expanded = np.repeat(weights, seq_len, axis=0)[:min_bars]

    return weights_expanded.astype(np.float64), symbols, candles


# ── Benchmark suite ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    name: str
    sharpe: float
    sortino: float
    total_return: float
    max_drawdown: float
    total_trades: int


def run_benchmark_suite(
    candles: dict[str, np.ndarray],
    bs: BacktestSpec = BacktestSpec(),
) -> list[BenchmarkResult]:
    """Run all Section 12.2 benchmarks and return results."""
    config = meridian_py.PortfolioConfig(
        base=meridian_py.BacktestConfig(
            initial_capital=bs.initial_capital,
            commission_pct=bs.commission_bps / 10_000,
            slippage_pct=bs.slippage_bps / 10_000,
            bars_per_year=bs.bars_per_year,
        ),
        max_positions=bs.max_positions,
    )

    benchmarks = [
        ("BuyAndHold", lambda: meridian_py.run_buy_and_hold(candles, config=config)),
        ("SMA_10_30", lambda: meridian_py.run_sma_crossover(candles, config=config)),
        ("RSI_14", lambda: meridian_py.run_rsi_reversal(candles, config=config)),
        ("Bollinger_20_2", lambda: meridian_py.run_bollinger(candles, config=config)),
    ]

    results = []
    for name, fn in benchmarks:
        r = fn()
        m = r.metrics
        results.append(
            BenchmarkResult(
                name=name,
                sharpe=m.sharpe_ratio,
                sortino=m.sortino_ratio,
                total_return=m.total_return_pct,
                max_drawdown=m.max_drawdown_pct,
                total_trades=m.total_trades,
            )
        )

    return results


# ── Monte Carlo ──────────────────────────────────────────────────────────────


def run_monte_carlo_analysis(
    result: meridian_py.PortfolioResult,
    symbols: list[str],
    bs: BacktestSpec = BacktestSpec(),
) -> dict[str, meridian_py.MonteCarloResult]:
    """Run Monte Carlo on each symbol's result, return {symbol: MCResult}."""
    mc_results = {}
    for sym in symbols:
        try:
            sr = result.symbol_result(sym)
            mc = meridian_py.run_monte_carlo(
                sr,
                num_simulations=bs.mc_simulations,
                seed=bs.mc_seed,
            )
            mc_results[sym] = mc
        except (KeyError, RuntimeError):
            continue
    return mc_results


# ── Main ─────────────────────────────────────────────────────────────────────


def print_results(
    model_result: meridian_py.PortfolioResult,
    benchmarks: list[BenchmarkResult],
    mc_results: dict[str, meridian_py.MonteCarloResult] | None = None,
) -> None:
    """Print formatted backtest comparison."""
    m = model_result.metrics
    print("\n" + "=" * 70)
    print("MERIDIAN BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Sharpe:        {m.sharpe_ratio:>10.4f}")
    print(f"  Sortino:       {m.sortino_ratio:>10.4f}")
    print(f"  Total Return:  {m.total_return_pct:>10.2f}%")
    print(f"  Max Drawdown:  {m.max_drawdown_pct:>10.4f}")
    print(f"  Total Trades:  {m.total_trades:>10d}")
    print(f"  Win Rate:      {m.win_rate:>10.2%}")
    print(f"  Calmar:        {m.calmar_ratio:>10.4f}")

    print("\n" + "-" * 70)
    print(
        f"{'Benchmark':<20} {'Sharpe':>10} {'Return':>10} {'MaxDD':>10} {'Trades':>8}"
    )
    print("-" * 70)
    for b in benchmarks:
        print(
            f"{b.name:<20} {b.sharpe:>10.4f} {b.total_return:>9.2f}% {b.max_drawdown:>10.4f} {b.total_trades:>8d}"
        )

    if mc_results:
        print("\n" + "-" * 70)
        print("MONTE CARLO DEPLOYMENT CRITERIA")
        print("-" * 70)
        passing = sum(
            1 for mc in mc_results.values() if mc.passes_deployment_criteria()
        )
        total = len(mc_results)
        print(f"  Symbols passing: {passing}/{total}")
        for sym, mc in sorted(mc_results.items()):
            status = "PASS" if mc.passes_deployment_criteria() else "FAIL"
            print(
                f"    {sym}: {status}  (return_p5={mc.total_return.p5:.2f}%, "
                f"sharpe_p25={mc.sharpe_ratio.p25:.3f}, dd_p95={mc.max_drawdown.p95:.4f})"
            )


def smoke_test() -> None:
    """Quick validation with synthetic weights — no checkpoint needed."""
    T, N = 500, 5
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    weights = np.random.dirichlet(np.ones(N), size=T).astype(np.float64)

    candles = {}
    for i, sym in enumerate(symbols):
        arr = np.zeros((T, 6), dtype=np.float64)
        price = 100.0 + i * 30
        for t in range(T):
            arr[t, 0] = 1609459200 + t * 60
            price *= 1 + 0.0002 + np.random.randn() * 0.003
            arr[t, 1] = price * 0.999
            arr[t, 2] = price * 1.002
            arr[t, 3] = price * 0.998
            arr[t, 4] = price
            arr[t, 5] = np.random.randint(1000, 100000)
        candles[sym] = arr

    result = meridian_py.run_backtest(weights, symbols, candles)
    benchmarks = run_benchmark_suite(candles)
    print_results(result, benchmarks)

    # Monte Carlo on one symbol
    sr = result.symbol_result("AAPL")
    mc = meridian_py.run_monte_carlo(sr, num_simulations=200, seed=42)
    print(f"\nMonte Carlo (AAPL): {mc}")
    print("Smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meridian backtesting framework")
    parser.add_argument(
        "--checkpoint", type=Path, help="path to trained model checkpoint"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument("--head", choices=["return", "sharpe"], default="return")
    parser.add_argument(
        "--smoke", action="store_true", help="run smoke test with synthetic data"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint required (or use --smoke)")

    bs = BacktestSpec()
    weights, symbols, candles = load_model_and_produce_weights(
        args.checkpoint,
        args.data_dir,
        args.head,
        device=args.device,
    )

    result = meridian_py.run_backtest(weights, symbols, candles)
    benchmarks = run_benchmark_suite(candles, bs)
    mc_results = run_monte_carlo_analysis(result, symbols, bs)
    print_results(result, benchmarks, mc_results)


if __name__ == "__main__":
    main()
