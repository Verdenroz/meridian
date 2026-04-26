//! Historical-mode Block 7 (Cross-Asset Context) computation.
//!
//! Spec §6.2 Block 7 (line 580-589) lists 8 features. 5 are computable
//! from existing OHLCV + dividend history; the remaining 3 (trending
//! membership, screener count, recommendation overlap) are
//! current-snapshot-only on the live finance-query API and have no
//! historical equivalent. They stay 0.0 in historical shards with a
//! documented note at the channel.

use crate::store::Bar;

/// Compute the 8 cross-asset features at a point in time given the
/// trailing-prefix bar slices. Inputs:
///   asset_bars   — asset OHLCV ending at the bar of interest
///   spy_bars     — SPY OHLCV aligned (most recent last)
///   sector_bars  — sector ETF OHLCV aligned (or empty for ETFs)
///   dividend_amounts — per-period dividend amounts ordered chronologically
///                     up to and including the current bar
pub fn compute_cross_asset_at(
    asset_bars: &[Bar],
    spy_bars: &[Bar],
    sector_bars: &[Bar],
    dividend_amounts: &[f64],
) -> [f32; 8] {
    let rs_market = relative_strength(asset_bars, spy_bars, 20);
    let rs_sector = relative_strength(asset_bars, sector_bars, 20);
    let corr_60 = rolling_corr(asset_bars, spy_bars, 60);
    let corr_change = corr_change_20d(asset_bars, spy_bars, 60, 20);
    let div_trend = dividend_trend(dividend_amounts);

    [
        rs_market,
        rs_sector,
        corr_60,
        corr_change,
        0.0, // trending — current-snapshot only on finance-query
        0.0, // screener_count — current-snapshot only
        0.0, // rec_overlap — requires current portfolio knowledge
        div_trend,
    ]
}

fn closes_tail(bars: &[Bar], n: usize) -> Vec<f64> {
    let take = bars.len().min(n);
    bars.iter().rev().take(take).map(|b| b.close).rev().collect()
}

fn log_returns(closes: &[f64]) -> Vec<f64> {
    if closes.len() < 2 {
        return Vec::new();
    }
    closes
        .windows(2)
        .filter_map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                Some((w[1] / w[0]).ln())
            } else {
                None
            }
        })
        .collect()
}

fn relative_strength(asset_bars: &[Bar], bench_bars: &[Bar], lookback: usize) -> f32 {
    if asset_bars.len() < 2 || bench_bars.len() < 2 {
        return 0.0;
    }
    let a_ret = trailing_return(asset_bars, lookback);
    let b_ret = trailing_return(bench_bars, lookback);
    ((a_ret - b_ret) as f32).clamp(-2.0, 2.0)
}

fn trailing_return(bars: &[Bar], lookback: usize) -> f64 {
    let take = bars.len().min(lookback + 1).max(2);
    let slice = &bars[bars.len() - take..];
    let first = slice[0].close;
    let last = slice[slice.len() - 1].close;
    if first <= 0.0 {
        0.0
    } else {
        (last / first).ln()
    }
}

fn rolling_corr(a: &[Bar], b: &[Bar], lookback: usize) -> f32 {
    let a_closes = closes_tail(a, lookback + 1);
    let b_closes = closes_tail(b, lookback + 1);
    let a_ret = log_returns(&a_closes);
    let b_ret = log_returns(&b_closes);
    pearson_corr(&a_ret, &b_ret)
}

fn corr_change_20d(a: &[Bar], b: &[Bar], window: usize, offset: usize) -> f32 {
    if a.len() < window + offset + 1 || b.len() < window + offset + 1 {
        return 0.0;
    }
    let now_corr = rolling_corr(a, b, window);
    let prev_a = &a[..a.len() - offset];
    let prev_b = &b[..b.len() - offset];
    let prev_corr = rolling_corr(prev_a, prev_b, window);
    (now_corr - prev_corr).clamp(-2.0, 2.0)
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f32 {
    let n = x.len().min(y.len());
    if n < 5 {
        return 0.0;
    }
    let x = &x[x.len() - n..];
    let y = &y[y.len() - n..];
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0) as f32
    }
}

/// Spec §6.2 Block 7 line 589: -1 (declining), 0 (flat / unknown), +1
/// (growing). Compares the most recent dividend to the median of the
/// previous (up to) 4 payments.
fn dividend_trend(amounts: &[f64]) -> f32 {
    if amounts.len() < 2 {
        return 0.0;
    }
    let last = *amounts.last().unwrap();
    let history_end = amounts.len() - 1;
    let history_start = history_end.saturating_sub(4);
    let mut hist: Vec<f64> = amounts[history_start..history_end].to_vec();
    if hist.is_empty() {
        return 0.0;
    }
    hist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = hist[hist.len() / 2];
    if median <= 0.0 {
        return 0.0;
    }
    let ratio = last / median;
    if ratio > 1.05 {
        1.0
    } else if ratio < 0.95 {
        -1.0
    } else {
        0.0
    }
}
