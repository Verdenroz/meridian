use crate::store::DataStore;
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// Block 7: Cross-Asset Context — 8 features.
///
/// Per-asset features derived from the asset's relationship to the broader market.
pub fn assemble(symbol: &str, store: &Arc<DataStore>, _now: DateTime<Utc>) -> [f32; 8] {
    // ── Relative strength (20d) vs SPY ───────────────────────────────────────
    let asset_ret_20d = compute_return_nd(&store.prices, symbol, 20 * 390);
    let spy_ret_20d = compute_return_nd(&store.prices, "SPY", 20 * 390);
    let rel_strength_market = if spy_ret_20d.abs() > 1e-8 {
        ((asset_ret_20d - spy_ret_20d) as f32).clamp(-2.0, 2.0)
    } else {
        0.0
    };

    // ── Sector relative strength ──────────────────────────────────────────────
    // (stub: uses SPY as proxy until sector mapping is wired)
    let sector_rel_strength: f32 = 0.0;

    // ── Rolling 60-day correlation to SPY ────────────────────────────────────
    let corr_spy_60d = rolling_corr(&store.prices, symbol, "SPY", 60 * 390);

    // ── Correlation change (vs 20d ago) ─────────────────────────────────────
    // Approximate: corr(last 60d) - corr(20d-offset 60d window)
    // Stub for now: 0
    let corr_change: f32 = 0.0;

    // ── Trending membership ───────────────────────────────────────────────────
    let is_trending: f32 = {
        let rss = store.rss_feed.read();
        // Trending is tracked separately by the market poller
        // Check the market_summary data's trending overlap
        drop(rss);
        0.0 // Filled by market poller updating a separate trending set
    };

    // ── Screener membership count ─────────────────────────────────────────────
    let screener_count: f32 = 0.0; // stub

    // ── Recommendation overlap ────────────────────────────────────────────────
    let rec_overlap: f32 = 0.0; // stub

    // ── Dividend analytics trend ──────────────────────────────────────────────
    let div_trend: f32 = 0.0; // stub

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
    if first <= 0.0 {
        0.0
    } else {
        (last / first).ln()
    }
}

/// Pearson correlation over the last `n_bars` bars between two symbols.
fn rolling_corr(
    prices: &Arc<dashmap::DashMap<String, crate::store::PriceBuffer>>,
    sym_a: &str,
    sym_b: &str,
    n_bars: usize,
) -> f32 {
    let Some(a_buf) = prices.get(sym_a) else {
        return 0.0;
    };
    let Some(b_buf) = prices.get(sym_b) else {
        return 0.0;
    };

    let a_closes = a_buf.closes(n_bars);
    let b_closes = b_buf.closes(n_bars);

    let n = a_closes.len().min(b_closes.len());
    if n < 5 {
        return 0.0;
    }

    let a = &a_closes[a_closes.len() - n..];
    let b = &b_closes[b_closes.len() - n..];

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
    if denom < 1e-10 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0) as f32
    }
}
