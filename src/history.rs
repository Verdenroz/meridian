//! Daily-history baselines, rebuilt at startup and refreshed each session:
//! EWMA(0.94) daily vol, trailing 20-day mean volume, and the prior close —
//! the normalizers behind every severity component. Also the intraday volume
//! profile (cumulative share of a day's volume by 5-minute bucket, averaged
//! over the trailing month) used to project full-day volume mid-session.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use finance_query::{Chart, Interval, Ticker, TimeRange};

const EWMA_LAMBDA: f64 = 0.94;
const VOLUME_LOOKBACK: usize = 20;
/// 78 five-minute buckets in a regular 9:30–16:00 session.
pub const SESSION_BUCKETS: usize = 78;

#[derive(Debug, Clone)]
pub struct Baseline {
    pub symbol: String,
    /// EWMA(0.94) daily vol of adj-close log returns, as of the last close.
    pub sigma: f64,
    pub avg_volume_20d: f64,
    pub prev_close: f64,
    pub prev_date: NaiveDate,
    pub built_at: DateTime<Utc>,
}

pub async fn fetch_baseline(symbol: &str, ticker: &Ticker) -> Result<Baseline> {
    let chart = ticker
        .chart(Interval::OneDay, TimeRange::OneYear)
        .await
        .with_context(|| format!("daily chart for {symbol}"))?;
    baseline_from_chart(symbol, &chart)
}

fn baseline_from_chart(symbol: &str, chart: &Chart) -> Result<Baseline> {
    let candles = &chart.candles;
    anyhow::ensure!(
        candles.len() >= VOLUME_LOOKBACK + 2,
        "{symbol}: only {} daily bars",
        candles.len()
    );

    let mut var: Option<f64> = None;
    let mut prev: Option<f64> = None;
    for c in candles {
        // Total-return basis where upstream provides it; close is the fallback.
        let px = c.adj_close.unwrap_or(c.close);
        if px <= 0.0 {
            continue;
        }
        if let Some(p) = prev {
            let r = (px / p).ln();
            var = Some(match var {
                Some(v) => EWMA_LAMBDA * v + (1.0 - EWMA_LAMBDA) * r * r,
                None => r * r,
            });
        }
        prev = Some(px);
    }
    let sigma = var.map(f64::sqrt).context("no returns for EWMA")?;
    anyhow::ensure!(sigma > 0.0, "{symbol}: zero EWMA vol");

    let tail = &candles[candles.len() - VOLUME_LOOKBACK..];
    let avg_volume_20d =
        tail.iter().map(|c| c.volume as f64).sum::<f64>() / VOLUME_LOOKBACK as f64;
    anyhow::ensure!(avg_volume_20d > 0.0, "{symbol}: zero average volume");

    let last = candles.last().unwrap();
    let prev_date = DateTime::from_timestamp(last.timestamp, 0)
        .context("bad last-bar timestamp")?
        .date_naive();

    Ok(Baseline {
        symbol: symbol.to_string(),
        sigma,
        avg_volume_20d,
        prev_close: last.close,
        prev_date,
        built_at: Utc::now(),
    })
}

/// Average cumulative share of a session's volume at the end of each
/// 5-minute bucket. U-shaped intraday volume makes a linear clock a bad
/// projector; this profile is the corrector.
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    shares: Vec<f64>,
}

impl VolumeProfile {
    pub async fn fetch(symbol: &str, ticker: &Ticker) -> Result<Self> {
        let chart = ticker
            .chart(Interval::FiveMinutes, TimeRange::OneMonth)
            .await
            .with_context(|| format!("5m chart for {symbol}"))?;
        Self::from_bars(&chart)
    }

    pub fn from_bars(chart: &Chart) -> Result<Self> {
        // Split into sessions on gaps > 1h between consecutive bars.
        let mut sessions: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut prev_ts: Option<i64> = None;
        for c in &chart.candles {
            let new_session = prev_ts.is_none_or(|p| c.timestamp - p > 3600);
            if new_session {
                sessions.push(Vec::new());
            }
            let s = sessions.last_mut().unwrap();
            let bucket = s.len();
            s.push((bucket, c.volume as f64));
            prev_ts = Some(c.timestamp);
        }
        // Only complete regular sessions contribute; partial days would skew
        // the tail buckets toward zero.
        sessions.retain(|s| s.len() >= SESSION_BUCKETS);
        anyhow::ensure!(!sessions.is_empty(), "no complete sessions in 5m history");

        let mut shares = vec![0.0; SESSION_BUCKETS];
        for s in &sessions {
            let total: f64 = s.iter().take(SESSION_BUCKETS).map(|(_, v)| v).sum();
            if total <= 0.0 {
                continue;
            }
            let mut cum = 0.0;
            for &(bucket, v) in s.iter().take(SESSION_BUCKETS) {
                cum += v;
                shares[bucket] += cum / total;
            }
        }
        let n = sessions.len() as f64;
        for s in &mut shares {
            *s /= n;
        }
        Ok(Self { shares })
    }

    /// Expected cumulative volume share after `elapsed_buckets` 5m buckets.
    /// Floored so an early-session projection cannot divide by ~0.
    pub fn expected_share(&self, elapsed_buckets: usize) -> f64 {
        if elapsed_buckets == 0 {
            return self.shares.first().copied().unwrap_or(0.05).max(0.02);
        }
        let idx = (elapsed_buckets - 1).min(self.shares.len() - 1);
        self.shares[idx].max(0.02)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_share_is_monotonic_floor() {
        let vp = VolumeProfile {
            shares: (1..=SESSION_BUCKETS).map(|i| i as f64 / SESSION_BUCKETS as f64).collect(),
        };
        assert!(vp.expected_share(0) >= 0.02);
        assert!(vp.expected_share(39) < vp.expected_share(SESSION_BUCKETS));
        // Past the end of the session clamps to the final (full) share.
        assert!((vp.expected_share(500) - 1.0).abs() < 1e-9);
    }
}
