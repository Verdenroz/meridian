//! Today's 5-minute bars for the trigger symbol: intraday realized vol and
//! the session-open timestamp that anchors the volume projection.

use anyhow::{Context, Result};
use finance_query::{Candle, Interval, Ticker, TimeRange};

pub struct TodayBars {
    pub open_ts: i64,
    pub last_ts: i64,
    pub bar_count: usize,
    /// Realized vol from 5m log returns, scaled to a full-day equivalent —
    /// comparable to the EWMA daily sigma.
    pub realized_vol_day: Option<f64>,
}

pub async fn fetch_today(symbol: &str, ticker: &Ticker) -> Result<TodayBars> {
    let chart = ticker
        .chart(Interval::FiveMinutes, TimeRange::OneDay)
        .await
        .with_context(|| format!("today 5m bars for {symbol}"))?;
    from_candles(&chart.candles)
}

pub fn from_candles(candles: &[Candle]) -> Result<TodayBars> {
    anyhow::ensure!(!candles.is_empty(), "no intraday bars");
    let open_ts = candles[0].timestamp;
    let last_ts = candles.last().unwrap().timestamp;

    let mut sum_sq = 0.0;
    let mut n = 0usize;
    let mut prev: Option<f64> = None;
    for c in candles {
        if c.close <= 0.0 {
            continue;
        }
        if let Some(p) = prev {
            let r = (c.close / p).ln();
            sum_sq += r * r;
            n += 1;
        }
        prev = Some(c.close);
    }
    // Day-scale the partial sum: with k of ~77 return intervals observed,
    // E[full-day sum] ≈ sum * 77/k (ignores the U-shape; diagnostic only).
    let realized_vol_day = if n >= 3 {
        Some((sum_sq * 77.0 / n as f64).sqrt())
    } else {
        None
    };

    Ok(TodayBars {
        open_ts,
        last_ts,
        bar_count: candles.len(),
        realized_vol_day,
    })
}
