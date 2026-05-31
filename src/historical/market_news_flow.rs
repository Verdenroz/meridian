//! Universe-wide daily news-flow signal for the market RSS/News block (spec §6.3).
//!
//! The 12 "RSS & News Aggregate Signals" dims (market block `[84..96]`) were a
//! near-constant FRED-release PCA proxy that only moved on FRED release dates
//! (~monthly), so the block carried almost no daily variation
//! (DATA_PIPELINE_ISSUES.md M1). Now that the FNSPID news corpus is cached per
//! symbol (`.fnspid_cache/`, see `fnspid.rs`), the Python ingestion also emits a
//! universe-wide **daily aggregate** at `$MERIDIAN_DATA_ROOT/.market_news_daily.json`:
//! for each UTC day, the total article count and the breadth (# distinct symbols
//! with news) across the whole 100-asset universe.
//!
//! This module loads that aggregate and turns it into two genuinely
//! daily-varying market-attention dims:
//!   * `[0]` news **intensity** — `log1p(count)`, causally EMA-detrended and
//!     z-scored (spec §6.3 normalizes news-intensity dims against a rolling-EMA
//!     baseline so source-cadence differences wash out);
//!   * `[1]` news **breadth** — same transform on the distinct-symbol count.
//!
//! Both are causal (each day's baseline uses only prior days), clamped to ±5 to
//! match the rest of the pre-normalized feature pipeline. A missing aggregate
//! file ⇒ empty lookup ⇒ the block falls back to the FRED-PCA dims unchanged
//! (honest absence, never fabricated). Loading from one JSON file (built once by
//! `make fnspid`) keeps the 100 parallel per-symbol collectors cheap.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// EMA smoothing factor for the trailing baseline + variance estimate. At daily
/// cadence α = 0.97 gives a ~23-day half-life — long enough to track the slow
/// drift in news volume, short enough that day-to-day spikes still register.
const EMA_ALPHA: f32 = 0.97;
const CLAMP: f32 = 5.0;
/// Floor on the z-score scale (std of log1p deviations). Prevents a low-variance
/// warm-up prefix from collapsing the denominator to ~0 and swallowing a genuine
/// spike. In log1p-count space a 0.25 floor treats a ~28% volume jump as ≈1σ.
const MIN_SCALE: f32 = 0.25;

#[derive(Deserialize)]
struct DailyAggregate {
    /// `{ "<day_ts>": [count, breadth], ... }` — day_ts is UTC midnight unix secs.
    #[serde(default)]
    days: BTreeMap<String, [f64; 2]>,
}

/// Forward-fillable daily news-flow lookup: `day_ts → [z_intensity, z_breadth]`.
#[derive(Default)]
pub struct NewsFlowLookup {
    map: BTreeMap<i64, [f32; 2]>,
}

impl NewsFlowLookup {
    pub fn cache_path() -> PathBuf {
        let root =
            std::env::var("MERIDIAN_DATA_ROOT").unwrap_or_else(|_| "./data".to_string());
        PathBuf::from(root).join(".market_news_daily.json")
    }

    /// Load + normalize the universe daily aggregate. Missing/unreadable/empty ⇒
    /// an empty lookup (the news block keeps its FRED-PCA values untouched).
    pub fn load() -> Self {
        let path = Self::cache_path();
        let Ok(bytes) = std::fs::read(&path) else {
            return Self::default();
        };
        let Ok(agg) = serde_json::from_slice::<DailyAggregate>(&bytes) else {
            tracing::warn!(path = %path.display(), "market_news_daily: parse failed → block falls back to FRED-PCA");
            return Self::default();
        };
        Self::from_aggregate(&agg)
    }

    fn from_aggregate(agg: &DailyAggregate) -> Self {
        // Sort by day so the EMA baseline is strictly causal.
        let mut rows: Vec<(i64, f64, f64)> = Vec::with_capacity(agg.days.len());
        for (k, v) in &agg.days {
            if let Ok(day_ts) = k.parse::<i64>() {
                rows.push((day_ts, v[0].max(0.0), v[1].max(0.0)));
            }
        }
        rows.sort_by_key(|(ts, _, _)| *ts);

        let mut map = BTreeMap::new();
        let mut intensity = CausalZScore::default();
        let mut breadth = CausalZScore::default();
        for (ts, count, br) in rows {
            let zi = intensity.push((count as f32).ln_1p());
            let zb = breadth.push((br as f32).ln_1p());
            map.insert(ts, [zi, zb]);
        }
        Self { map }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Most recent `[z_intensity, z_breadth]` at-or-before `ts`, else `None`
    /// (caller leaves the FRED-PCA dims in place).
    pub fn value_at(&self, ts: i64) -> Option<[f32; 2]> {
        self.map.range(..=ts).next_back().map(|(_, v)| *v)
    }
}

/// Streaming causal z-score: deviation from a trailing EMA mean, scaled by a
/// trailing EMA of squared deviations. The first sample is defined as 0 (no
/// baseline yet) rather than an undefined spike.
#[derive(Default)]
struct CausalZScore {
    initialized: bool,
    ema: f32,
    ema_var: f32,
}

impl CausalZScore {
    fn push(&mut self, x: f32) -> f32 {
        if !self.initialized {
            self.initialized = true;
            self.ema = x;
            self.ema_var = 0.0;
            return 0.0;
        }
        let dev = x - self.ema; // deviation from the trailing (prior) baseline
        let scale = self.ema_var.sqrt().max(MIN_SCALE);
        let z = (dev / scale).clamp(-CLAMP, CLAMP);
        // Update baselines AFTER scoring so the score uses only prior days.
        self.ema = EMA_ALPHA * self.ema + (1.0 - EMA_ALPHA) * x;
        self.ema_var = EMA_ALPHA * self.ema_var + (1.0 - EMA_ALPHA) * dev * dev;
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agg(pairs: &[(i64, f64, f64)]) -> DailyAggregate {
        let mut days = BTreeMap::new();
        for (ts, c, b) in pairs {
            days.insert(ts.to_string(), [*c, *b]);
        }
        DailyAggregate { days }
    }

    #[test]
    fn empty_aggregate_is_empty_lookup() {
        let lk = NewsFlowLookup::from_aggregate(&agg(&[]));
        assert!(lk.is_empty());
        assert!(lk.value_at(123).is_none());
    }

    #[test]
    fn produces_daily_varying_zscores() {
        // A realistically wobbling baseline (~80–120 articles/day) with a clear
        // spike on day 5 — universe news volume always varies day to day.
        let day = 86_400i64;
        let baseline = [90.0, 110.0, 85.0, 120.0, 95.0, 1000.0, 105.0, 88.0, 115.0, 92.0];
        let mut pairs = Vec::new();
        for (d, &count) in baseline.iter().enumerate() {
            pairs.push((d as i64 * day, count, 30.0 + (d % 3) as f64 * 5.0));
        }
        let lk = NewsFlowLookup::from_aggregate(&agg(&pairs));
        assert_eq!(lk.map.len(), 10);
        // The spike day must score a large positive intensity z.
        let spike = lk.value_at(5 * day).unwrap()[0];
        let calm = lk.value_at(8 * day).unwrap()[0];
        assert!(spike > 1.0, "spike z should be clearly positive, got {spike}");
        assert!(spike > calm, "spike must exceed a calm day");
        // The block genuinely varies day to day (not the old ~constant proxy).
        let uniq: std::collections::HashSet<_> =
            lk.map.values().map(|v| v[0].to_bits()).collect();
        assert!(uniq.len() > 3, "intensity should take many distinct values");
    }

    #[test]
    fn forward_fills_between_days() {
        let day = 86_400i64;
        let lk = NewsFlowLookup::from_aggregate(&agg(&[(0, 100.0, 30.0), (2 * day, 120.0, 31.0)]));
        // A ts between day 0 and day 2 picks up day 0's value.
        assert_eq!(lk.value_at(day).unwrap(), lk.value_at(0).unwrap());
        // Before the first day ⇒ None.
        assert!(lk.value_at(-1).is_none());
    }

    #[test]
    fn first_sample_is_zero_baseline() {
        let mut z = CausalZScore::default();
        assert_eq!(z.push(5.0), 0.0);
    }
}
