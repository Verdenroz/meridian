use crate::features::sector_map::sector_returns_idx_for;
use crate::store::DataStore;
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// Block 7: Cross-Asset Context — 8 features.
///
/// Per-asset features derived from the asset's relationship to the broader market.
///
/// | Dim | Feature                   | Live mode source                          |
/// |-----|---------------------------|-------------------------------------------|
/// | 112 | RS vs SPY (20d)           | store.prices (SPY price buffer)           |
/// | 113 | RS vs sector (daily)      | store.sector.returns[sector_idx]          |
/// | 114 | Correlation to SPY (60d)  | store.prices (Pearson over 60d window)    |
/// | 115 | Correlation Δ (20d)       | store.prices (corr at t − corr at t−20d)  |
/// | 116 | Trending membership       | store.trending_symbols                    |
/// | 117 | Screener membership       | store.screener_membership                 |
/// | 118 | Recommendation overlap    | store.analyst (strong_buy+buy fraction)   |
/// | 119 | Dividend analytics trend  | store.fundamentals (dividend_yield proxy) |
pub fn assemble(symbol: &str, store: &Arc<DataStore>, _now: DateTime<Utc>) -> [f32; 8] {
    // ── dim 112: Relative strength (20d) vs SPY ──────────────────────────────
    let asset_ret_20d = compute_return_nd(&store.prices, symbol, 20 * 390);
    let spy_ret_20d = compute_return_nd(&store.prices, "SPY", 20 * 390);
    let rel_strength_market = if spy_ret_20d.abs() > 1e-8 {
        ((asset_ret_20d - spy_ret_20d) as f32).clamp(-2.0, 2.0)
    } else {
        0.0
    };

    // ── dim 113: Sector relative strength (1d, live mode) ────────────────────
    // Uses the daily sector return from the sector poller (finance::sector()) as
    // the benchmark; asset's 1d return from price buffer for parity.
    // Note: historical mode uses a 20d sector ETF lookback — the daily proxy is
    // the best available in live inference without subscribing sector ETF feeds.
    let sector_rel_strength: f32 = {
        let asset_ret_1d = compute_return_nd(&store.prices, symbol, 390);
        if let Some(idx) = sector_returns_idx_for(symbol) {
            let sector_ret = store.sector.read().value.returns[idx] as f64;
            ((asset_ret_1d - sector_ret) as f32).clamp(-2.0, 2.0)
        } else {
            // ETFs, crypto, or unrecognized symbols default to market RS
            rel_strength_market
        }
    };

    // ── dim 114: Rolling 60-day correlation to SPY ───────────────────────────
    let corr_spy_60d = rolling_corr(&store.prices, symbol, "SPY", 60 * 390);

    // ── dim 115: Correlation change (vs 20d ago) ─────────────────────────────
    let corr_change = {
        let corr_now = corr_spy_60d;
        let corr_prior = rolling_corr_offset(&store.prices, symbol, "SPY", 60 * 390, 20 * 390);
        (corr_now - corr_prior).clamp(-2.0, 2.0)
    };

    // ── dim 116: Trending membership ─────────────────────────────────────────
    let is_trending: f32 = store.trending_symbols.read().contains(symbol) as i32 as f32;

    // ── dim 117: Screener membership count (normalised) ──────────────────────
    let screener_count: f32 = store
        .screener_membership
        .get(symbol)
        .map(|v| *v)
        .unwrap_or(0.0);

    // ── dim 118: Recommendation overlap (bullish consensus) ──────────────────
    let rec_overlap: f32 = store
        .analyst
        .get(symbol)
        .and_then(|cached| {
            let p = cached.value.recommendation_period.as_ref()?;
            let bullish = (p.strong_buy + p.buy) as f32;
            let total = (p.strong_buy + p.buy + p.hold + p.sell + p.strong_sell) as f32;
            if total > 0.0 { Some((bullish / total).clamp(0.0, 1.0)) } else { None }
        })
        .unwrap_or(0.0);

    // ── dim 119: Dividend analytics trend ────────────────────────────────────
    // Track yield vs its own rolling history via z-score; positive = growing,
    // negative = declining, zero = no dividend or flat. Matches the -1/0/+1
    // semantics of the historical mode without requiring stored payment history.
    let div_trend: f32 = store
        .fundamentals
        .get(symbol)
        .and_then(|f| f.value.dividend_yield)
        .map(|y| {
            store.rolling_stats.zscore(symbol, "div_yield", y).clamp(-1.0, 1.0)
        })
        .unwrap_or(0.0);

    [
        rel_strength_market,
        sector_rel_strength,
        corr_spy_60d,
        corr_change,
        is_trending,
        screener_count,
        rec_overlap,
        div_trend,
    ]
}

/// Compute ln-return over the last `n_bars` bars for `symbol`.
fn compute_return_nd(
    prices: &Arc<dashmap::DashMap<String, crate::store::PriceBuffer>>,
    symbol: &str,
    n_bars: usize,
) -> f64 {
    let Some(buf) = prices.get(symbol) else {
        return 0.0;
    };
    let closes = buf.closes(n_bars + 1);
    let n = closes.len();
    if n < 2 {
        return 0.0;
    }
    let first = closes[0];
    let last = closes[n - 1];
    if first <= 0.0 { 0.0 } else { (last / first).ln() }
}

/// Pearson correlation over the last `n_bars` bars between two symbols.
fn rolling_corr(
    prices: &Arc<dashmap::DashMap<String, crate::store::PriceBuffer>>,
    sym_a: &str,
    sym_b: &str,
    n_bars: usize,
) -> f32 {
    pearson_from_buffers(prices, sym_a, sym_b, n_bars, 0)
}

/// Pearson correlation using a window that ends `offset_bars` before the latest bar.
/// Used to compute correlation at t−offset for the correlation-change feature.
fn rolling_corr_offset(
    prices: &Arc<dashmap::DashMap<String, crate::store::PriceBuffer>>,
    sym_a: &str,
    sym_b: &str,
    n_bars: usize,
    offset_bars: usize,
) -> f32 {
    pearson_from_buffers(prices, sym_a, sym_b, n_bars, offset_bars)
}

fn pearson_from_buffers(
    prices: &Arc<dashmap::DashMap<String, crate::store::PriceBuffer>>,
    sym_a: &str,
    sym_b: &str,
    n_bars: usize,
    offset_bars: usize,
) -> f32 {
    let Some(a_buf) = prices.get(sym_a) else {
        return 0.0;
    };
    let Some(b_buf) = prices.get(sym_b) else {
        return 0.0;
    };

    let fetch = n_bars + offset_bars;
    let a_closes = a_buf.closes(fetch);
    let b_closes = b_buf.closes(fetch);

    // Trim off the `offset_bars` most-recent bars so both series end at t−offset.
    let a_end = a_closes.len().saturating_sub(offset_bars);
    let b_end = b_closes.len().saturating_sub(offset_bars);
    let n = a_end.min(b_end);
    if n < 5 {
        return 0.0;
    }

    let a = &a_closes[a_end - n..a_end];
    let b = &b_closes[b_end - n..b_end];

    let mean_a = a.iter().sum::<f64>() / n as f64;
    let mean_b = b.iter().sum::<f64>() / n as f64;

    let mut num = 0.0f64;
    let mut da2 = 0.0f64;
    let mut db2 = 0.0f64;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }

    let denom = (da2 * db2).sqrt();
    if denom < 1e-10 { 0.0 } else { (num / denom).clamp(-1.0, 1.0) as f32 }
}
