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
    bars_per_year: float = 252  # daily bars
    max_positions: int = 100
    mc_simulations: int = 1000
    mc_seed: int = 42


# ── Candle loading ───────────────────────────────────────────────────────────

def load_candles_from_shards(
    data_dir: Path,
    symbols: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    """Reconstruct OHLCV candle arrays from f32bin shard log-returns.

    Shard price_volume block layout (10 floats):
      [0] open  = ln(open / prev_close)
      [1] high  = ln(high / open)
      [2] low   = ln(low / open)
      [3] close = ln(close / prev_close)   ← primary return
      [4] volume (normalized vs SMA20)
      [5-9] multi-horizon returns

    We reconstruct synthetic prices via cumulative exponentiation from a
    base price of 100.  The backtesting engine only needs relative movements.

    Returns:
        candles: {symbol: (T, 6) float64 array of [ts, O, H, L, C, V]}
        min_bars: shortest shard length across all symbols
    """
    feature_dim = 184
    candles = {}
    min_bars = float("inf")

    for sym in symbols:
        path = data_dir / f"{sym}.f32bin"
        if not path.exists():
            continue

        raw = np.memmap(path, dtype="float32", mode="r")
        n = len(raw) // feature_dim
        data = raw[: n * feature_dim].reshape(n, feature_dim).astype(np.float64)
        min_bars = min(min_bars, n)

        # Reconstruct close prices from log-returns: close[t] = base * exp(cumsum(lr))
        close_lr = data[:, 3]  # ln(close / prev_close)
        close = 100.0 * np.exp(np.cumsum(close_lr))

        # Reconstruct open/high/low relative to close
        open_lr = data[:, 0]   # ln(open / prev_close)
        high_lr = data[:, 1]   # ln(high / open)
        low_lr = data[:, 2]    # ln(low / open)

        # prev_close for each bar (shift close by 1, first bar uses base)
        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = 100.0
        prev_close[1:] = close[:-1]

        open_price = prev_close * np.exp(open_lr)
        high_price = open_price * np.exp(high_lr)
        low_price = open_price * np.exp(low_lr)

        # Volume: de-normalize from SMA20-normalized (approximate, use exp)
        vol_norm = data[:, 4]
        volume = 1_000_000.0 * np.exp(vol_norm)  # synthetic absolute volume

        # Synthetic daily timestamps (86400s apart)
        ts = np.arange(n, dtype=np.float64) * 86400 + 1609459200

        arr = np.column_stack([ts, open_price, high_price, low_price, close, volume])
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

    Mirrors the recurrent forward pass from PortfolioTrainer._forward():
    each timestep runs regime detection → input projection → backbone →
    cross-asset fusion → head, stepping the GDN recurrent state forward.

    Uses feature shards for model inference but fetches real OHLCV prices
    from Yahoo Finance for the backtesting engine.

    Returns:
        weights: (T, N) float64 array of target portfolio weights
        symbols: list of N ticker strings
        candles: dict mapping symbol to (T, 6) real OHLCV array
    """
    from optimization.portfolio_heads import MeridianR, MeridianS
    from train.pretrain import build_model

    # Discover symbols from available shard files
    shard_files = sorted(data_dir.glob("*.f32bin"))
    symbols = [p.stem for p in shard_files]
    if not symbols:
        raise FileNotFoundError(f"no .f32bin shards in {data_dir}")

    # Load shard features for model inference
    feature_dim = 184
    shard_arrays = {}
    for sym in symbols:
        path = data_dir / f"{sym}.f32bin"
        raw = np.memmap(path, dtype="float32", mode="r")
        n = len(raw) // feature_dim
        shard_arrays[sym] = raw[: n * feature_dim].reshape(n, feature_dim)

    # Tail-align shards to shortest common length
    min_bars = min(a.shape[0] for a in shard_arrays.values())
    arrays = [shard_arrays[sym][-min_bars:] for sym in symbols]
    N = len(symbols)

    # Fetch real OHLCV candles from Yahoo Finance for backtesting
    print(f"Fetching real OHLCV for {N} symbols...")
    real_candles = meridian_py.fetch_candles(symbols, interval="1d", range="10y")
    # Keep only symbols that were successfully fetched
    fetched = [s for s in symbols if s in real_candles]
    if len(fetched) < N:
        missing = set(symbols) - set(fetched)
        print(f"  warning: {len(missing)} symbols not fetched: {missing}")
        # Re-align: drop missing symbols from inference arrays too
        idx_map = {s: i for i, s in enumerate(symbols)}
        arrays = [arrays[idx_map[s]] for s in fetched]
        symbols = fetched
        N = len(symbols)
        min_bars = min(a.shape[0] for a in arrays)
        arrays = [a[-min_bars:] for a in arrays]

    # Tail-align real candles to shard length
    candles = {}
    for sym in symbols:
        rc = real_candles[sym]
        candles[sym] = rc[-min_bars:] if rc.shape[0] >= min_bars else rc
    print(f"  {N} symbols, {min_bars} bars (shard), candle lengths: "
          f"{min(c.shape[0] for c in candles.values())}-"
          f"{max(c.shape[0] for c in candles.values())}")

    # (N, T, F) aligned features
    features = np.stack(arrays, axis=0)

    # Build model components (same construction as portfolio_train)
    dev = torch.device(device)
    backbone, fusion, regime_det, input_proj = build_model(spec, dev)
    head_r = MeridianR(spec.d_model).to(dev)
    head_s = MeridianS(spec.d_model).to(dev)
    head_module = head_r if head == "return" else head_s

    # Load checkpoint — format: {model_state: {backbone: ..., fusion: ..., ...}}
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state", ckpt)

    component_map = {
        "backbone": backbone,
        "fusion": fusion,
        "regime_detector": regime_det,
        "input_projection": input_proj,
        "head_return": head_r,
        "head_sharpe": head_s,
    }
    for key, module in component_map.items():
        if key in model_state:
            module.load_state_dict(model_state[key], strict=False)

    for m in [input_proj, regime_det, backbone, fusion, head_module]:
        m.eval()

    # Run recurrent forward pass (mirrors PortfolioTrainer._forward)
    feat_tensor = torch.from_numpy(features).float().to(dev)  # (N, T, F)
    B = 1

    # Split per-asset (184) features — no market block in 184-dim shards
    # Market features are the last 96 dims when feature_dim=280; with 184 we
    # use zeros (same as training when market block is absent from shard).
    asset_feats = feat_tensor.unsqueeze(0)  # (1, N, T, 184)

    # Bootstrap recurrent state
    portfolio_s = torch.zeros(B, spec.portfolio_state_dim, device=dev)
    calendar = torch.zeros(B, spec.calendar_dim, device=dev)
    sector_ids = torch.zeros(N, dtype=torch.long, device=dev)
    episodic = torch.zeros(B, spec.n_episodic_retrievals, spec.d_model, device=dev)
    regime_ema = torch.zeros(B, spec.regime_emb_dim, device=dev)
    prev_regime_emb = None
    prev_port_return = torch.zeros(B, device=dev)
    portfolio_value = torch.ones(B, device=dev)
    states = backbone.init_state(B, N, dev)

    n_full = spec.n_assets  # 100
    _zeros_n = torch.zeros(B, n_full, device=dev)
    _zeros_1 = torch.zeros(B, 1, device=dev)

    # Pre-compute sector bias
    with torch.no_grad():
        n_episodic = episodic.shape[1]
        n_total = N + 1 + 1 + n_episodic
        n_extras = n_total - N
        extra_ids = torch.full((n_extras,), -1, dtype=torch.long, device=dev)
        full_ids = torch.cat([sector_ids, extra_ids], dim=0)
        precomputed_bias = fusion.sector_bias.bias_matrix(full_ids, n_total)

    all_weights = []
    use_autocast = dev.type == "cuda"

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        for t in range(min_bars):
            af_t = asset_feats[:, :, t, :]  # (B, N, 184)
            # Market features: zeros (not present in 184-dim shards)
            mf_t = torch.zeros(B, 96, device=dev)

            # Regime detection
            portfolio_value = portfolio_value * (1.0 + prev_port_return)
            portfolio_summary = torch.stack([
                portfolio_value, prev_port_return,
                torch.ones(B, device=dev),
            ], dim=-1)  # (B, 3)
            regime_emb, posteriors, _ = regime_det(
                mf_t, portfolio_summary, regime_ema, prev_regime_emb,
            )
            regime_ema = (0.98 * regime_ema + 0.02 * regime_emb).detach()
            prev_regime_emb = regime_emb.detach()

            # Input projection
            asset_tok, mkt_tok, port_tok, _ = input_proj(
                af_t, mf_t, portfolio_s, calendar, regime_emb
            )
            out, states = backbone(asset_tok, regime_emb, posteriors, states)
            fused, mkt_out = fusion(
                out, mkt_tok, port_tok, episodic, sector_ids,
                precomputed_bias=precomputed_bias,
            )

            # Head
            w_t = head_module(fused)  # (B, N)
            all_weights.append(w_t.cpu().numpy())

            # Update portfolio state
            w_padded = F.pad(w_t.detach(), (0, n_full - N))
            portfolio_s = torch.cat([
                w_padded, _zeros_n, _zeros_1,
            ], dim=-1)

            prev_port_return = (w_t.detach() * af_t[:, :, 3]).sum(dim=-1)

            if t > 0 and t % 100 == 0:
                print(f"  inference progress: {t}/{min_bars}")

    weights = np.concatenate(all_weights, axis=0)  # (T_shard, N)

    # Align weights and candles to the same length (take tail of each)
    min_candle = min(c.shape[0] for c in candles.values())
    T = min(weights.shape[0], min_candle)
    weights = weights[-T:]
    candles = {sym: arr[-T:] for sym, arr in candles.items()}
    print(f"  aligned {T} bars for backtest")

    return weights.astype(np.float64), symbols, candles


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
                method="block_bootstrap",
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
    mc = meridian_py.run_monte_carlo(sr, num_simulations=200, method="block_bootstrap", seed=42)
    print(f"\nMonte Carlo (AAPL): {mc}")
    print("Smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meridian backtesting framework")
    parser.add_argument(
        "--checkpoint", type=Path, help="path to trained model checkpoint"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmaps"))
    parser.add_argument("--split", default="test",
                        help="data split to backtest on (default: test)")
    parser.add_argument("--head", choices=["return", "sharpe"], default="return")
    parser.add_argument(
        "--smoke", action="store_true", help="run smoke test with synthetic data"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint required (or use --smoke)")

    bs = BacktestSpec()
    shard_dir = args.data_dir / args.split
    if not shard_dir.exists():
        parser.error(f"split directory not found: {shard_dir}")
    print(f"Backtesting on {args.split} split: {shard_dir}")
    weights, symbols, candles = load_model_and_produce_weights(
        args.checkpoint,
        shard_dir,
        args.head,
        device=args.device,
    )

    result = meridian_py.run_backtest(weights, symbols, candles)
    benchmarks = run_benchmark_suite(candles, bs)
    mc_results = run_monte_carlo_analysis(result, symbols, bs)
    print_results(result, benchmarks, mc_results)


if __name__ == "__main__":
    main()
