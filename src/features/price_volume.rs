use crate::args::HistoricalInterval;
use crate::store::{PriceBuffer, RollingStatsStore};
use std::sync::Arc;

/// Determine the lookback window size in bars based on interval.
/// Returns the number of bars to request for computing price/volume features.
fn bar_lookback_window(interval: &HistoricalInterval) -> usize {
    match interval {
        HistoricalInterval::OneMinute => 390,    // 1 trading day (6.5h × 60m)
        HistoricalInterval::FiveMinutes => 78,   // 1 trading day (6.5h × 12 × 5m)
        HistoricalInterval::OneHour => 24,       // ~3 trading days  
        HistoricalInterval::OneDay => 252,       // 1 trading year
    }
}

/// Block 1: Price & Volume — 10 features.
///
/// All log-return features. Volume is normalized relative to a rolling SMA.
pub fn assemble(
    buf: Option<&PriceBuffer>,
    stats: &Arc<RollingStatsStore>,
    symbol: &str,
    interval: &HistoricalInterval,
) -> [f32; 10] {
    let Some(buf) = buf else {
        return [0.0f32; 10];
    };
    if buf.len() < 2 {
        return [0.0f32; 10];
    }

    let window = bar_lookback_window(interval);
    let closes = buf.closes(window); // up to 1 year of bars (adjusted for interval)
    let n = closes.len();
    if n < 2 {
        return [0.0f32; 10];
    }

    let last = closes[n - 1];
    let prev = closes[n - 2];

    // Safely compute log-return with zero-guard
    let ln_ret = |a: f64, b: f64| -> f32 {
        if b <= 0.0 || a <= 0.0 {
            0.0f32
        } else {
            (a / b).ln() as f32
        }
    };

    // Return over N bars ago (index offset)
    let ret_n = |offset: usize| -> f32 {
        if n <= offset {
            0.0
        } else {
            ln_ret(last, closes[n - 1 - offset])
        }
    };

    // Open of the current bar (use first close in the buffer as proxy)
    let opens = buf.opens(window);
    let m = opens.len();
    let open_price = if m > 0 { opens[m - 1] } else { last };

    // High / low of the current bar
    let highs = buf.highs(window);
    let lows = buf.lows(window);
    let nh = highs.len();
    let nl = lows.len();
    let high_price = if nh > 0 { highs[nh - 1] } else { last };
    let low_price = if nl > 0 { lows[nl - 1] } else { last };

    // Volume normalized by 20-period SMA
    let volumes = buf.volumes(20);
    let vol_last = buf.volumes(1).first().copied().unwrap_or(1.0);
    let sma20_vol = if volumes.is_empty() {
        1.0
    } else {
        volumes.iter().sum::<f64>() / volumes.len() as f64
    };
    let vol_norm = if sma20_vol > 0.0 {
        (vol_last / sma20_vol).ln().clamp(-5.0, 5.0) as f32
    } else {
        0.0
    };

    // Update rolling stats for volume normalization
    stats.update(symbol, "vol_norm", vol_norm as f64);

    [
        ln_ret(open_price, prev),       // open (normalized vs prev close)
        ln_ret(high_price, open_price), // high vs open
        ln_ret(low_price, open_price),  // low vs open
        ln_ret(last, prev),             // close vs prev close
        vol_norm,                       // volume vs SMA20
        ln_ret(last, prev),             // return 1 bar (same as close)
        ret_n(5),                       // return 5 bars
        ret_n(15),                      // return 15 bars
        ret_n(60),                      // return 60 bars
        ret_n(390),                     // return 390 bars (1 year equiv in daily)
    ]
}

