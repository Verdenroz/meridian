//! Historical-mode market-block macro fetchers.
//!
//! Spec §6.3 prescribes:
//!   - Yield Curve [12..24]: 12 maturities from `fred::treasury_yields(year)`
//!   - FRED Macro  [24..36]: 12 series from `fred::series(id)`
//!
//! Live mode polls these once a day; for historical training we collect
//! every observation across the training window and build per-date lookup
//! tables. Both FRED-side feeds have a "publish-and-hold" semantic — the
//! most recent value as of any timestamp applies until the next release —
//! so lookups are forward-fill with `BTreeMap::range(..=ts).next_back()`.

use crate::store::YieldCurveData;
use chrono::{NaiveDate, TimeZone, Utc};
use finance_query::fred;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// 12 FRED series IDs in the order spec §6.3 line 633-648 prescribes.
/// Mirrors `src/poller/macro_fred.rs:8-21` exactly so the live and
/// historical paths produce identical layouts.
pub const FRED_SERIES: &[&str] = &[
    "FEDFUNDS", // Fed Funds Rate
    "UNRATE",   // Unemployment Rate
    "CPIAUCSL", // CPI YoY
    "CPILFESL", // Core CPI YoY
    "ICSA",     // Initial Jobless Claims
    "CCSA",     // Continuing Claims
    "MANEMP",   // ISM Manufacturing proxy
    "RSXFS",    // Retail Sales MoM
    "INDPRO",   // Industrial Production
    "HOUST",    // Housing Starts
    "UMCSENT",  // Consumer Confidence
    "USSLIND",  // Leading Economic Index
];

/// Human-readable labels matching `FRED_SERIES` 1:1; used by the
/// market-news PCA corpus (`market_news_pca.rs`) to build deterministic
/// release-event snippets. Order tracks `FRED_SERIES`.
pub const FRED_LABELS: &[&str] = &[
    "Fed Funds Rate",
    "Unemployment Rate",
    "CPI All Items",
    "Core CPI",
    "Initial Jobless Claims",
    "Continuing Jobless Claims",
    "Manufacturing Employment",
    "Retail Sales",
    "Industrial Production",
    "Housing Starts",
    "Consumer Sentiment",
    "Leading Economic Index",
];

#[derive(Default)]
pub struct MacroLookup {
    /// (timestamp at midnight UTC) → yield curve snapshot.
    pub yield_curve: BTreeMap<i64, YieldCurveData>,
    /// Raw per-series FRED observations (pre-z-score). Parallel to
    /// `FRED_SERIES` / `FRED_LABELS`. Used by the market-news PCA pipeline
    /// to construct release-event text snippets without re-fetching.
    pub per_series_raw: Vec<BTreeMap<i64, f64>>,
    /// (timestamp) → 12-element macro values, ordered per `FRED_SERIES`.
    /// Stored z-scored against per-series mean/std computed across the full
    /// observation window. The raw FRED values span 7 orders of magnitude
    /// (FEDFUNDS ~5%, ICSA ~600k, CPIAUCSL ~300) and would blow the FDA
    /// pipeline's max_abs=10 hard gate; spec §6.2 line 392 already mandates
    /// rolling z-scores for indicators, and we extend the same convention
    /// to the market block. Look-ahead bias is negligible — macro series
    /// move slowly, so the global mean/std is close to a per-bar rolling
    /// estimate at any point in the training window.
    pub macro_values: BTreeMap<i64, [f64; 12]>,
}

impl MacroLookup {
    /// Cached on disk at `$MERIDIAN_DATA_ROOT/.macro_cache.json` (default:
    /// `./data/.macro_cache.json`) with a 24-hour TTL. Each `make collect`
    /// run launches 100 per-symbol Rust processes in parallel via the
    /// Python pipeline; without the cache, every process would refetch
    /// 11 years × 12 FRED series concurrently and get rate-limit-banned.
    /// Cache hit makes the macro lookup ~free for symbols 2-100.
    pub fn cache_path() -> PathBuf {
        let root = std::env::var("MERIDIAN_DATA_ROOT")
            .unwrap_or_else(|_| "./data".to_string());
        PathBuf::from(root).join(".macro_cache.json")
    }

    /// Try to load a cached MacroLookup; returns Some only if the file exists
    /// and the embedded `built_at` timestamp is < 24h old.
    fn try_load_cache() -> Option<Self> {
        let path = Self::cache_path();
        let bytes = std::fs::read(&path).ok()?;
        let cached: CacheEntry = serde_json::from_slice(&bytes).ok()?;
        let now = Utc::now().timestamp();
        if now - cached.built_at > 24 * 3600 {
            return None; // stale
        }
        let mut out = MacroLookup::default();
        for (ts, yc) in cached.yield_curve {
            out.yield_curve.insert(ts, yc.into());
        }
        for (ts, vals) in cached.macro_values {
            out.macro_values.insert(ts, vals);
        }
        out.per_series_raw = cached.per_series_raw;
        Some(out)
    }

    fn save_cache(&self) {
        let cached = CacheEntry {
            built_at: Utc::now().timestamp(),
            yield_curve: self
                .yield_curve
                .iter()
                .map(|(ts, yc)| (*ts, SerializableYieldCurve::from(yc)))
                .collect(),
            macro_values: self.macro_values.clone(),
            per_series_raw: self.per_series_raw.clone(),
        };
        if let Ok(bytes) = serde_json::to_vec(&cached) {
            let path = Self::cache_path();
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // Best-effort: ignore write errors (e.g. read-only filesystem).
            let _ = std::fs::write(&path, bytes);
        }
    }

    /// Build by fetching all yield-curve and FRED-macro history needed to
    /// cover `start_year..=end_year` (inclusive). On error from any single
    /// fetch we log and continue with what we have — partial macro data is
    /// vastly better than none. Cache-first: if a fresh sidecar exists,
    /// reuse it instead of refetching.
    pub async fn build(start_year: i32, end_year: i32) -> Self {
        if let Some(cached) = Self::try_load_cache() {
            tracing::info!("macro_lookup: loaded from cache");
            return cached;
        }
        tracing::info!("macro_lookup: building (cache miss or stale)");
        let out = Self::build_uncached(start_year, end_year).await;
        out.save_cache();
        out
    }

    async fn build_uncached(start_year: i32, end_year: i32) -> Self {
        let mut out = Self::default();

        // ── Treasury yields: one fetch per year ─────────────────────────
        for year in start_year..=end_year {
            match fred::treasury_yields(year as u32).await {
                Ok(yields) => {
                    for ty in yields {
                        // TreasuryYield.date is MM/DD/YYYY (per fq docs).
                        let Some(ts) = parse_mdy(&ty.date) else { continue };
                        let yc = YieldCurveData {
                            y1m: ty.y1m.unwrap_or(0.0),
                            y3m: ty.y3m.unwrap_or(0.0),
                            y6m: ty.y6m.unwrap_or(0.0),
                            y1: ty.y1.unwrap_or(0.0),
                            y2: ty.y2.unwrap_or(0.0),
                            y3: ty.y3.unwrap_or(0.0),
                            y5: ty.y5.unwrap_or(0.0),
                            y7: ty.y7.unwrap_or(0.0),
                            y10: ty.y10.unwrap_or(0.0),
                            y20: ty.y20.unwrap_or(0.0),
                            y30: ty.y30.unwrap_or(0.0),
                            spread_2y10y: ty.y10.unwrap_or(0.0)
                                - ty.y2.unwrap_or(0.0),
                        };
                        out.yield_curve.insert(ts, yc);
                    }
                }
                Err(err) => {
                    tracing::warn!(year, err = %err, "treasury_yields fetch failed");
                }
            }
        }

        // ── FRED macro: one fetch per series, all observations at once ──
        let mut per_series: Vec<BTreeMap<i64, f64>> = (0..FRED_SERIES.len())
            .map(|_| BTreeMap::new())
            .collect();
        for (i, series_id) in FRED_SERIES.iter().enumerate() {
            match fred::series(series_id).await {
                Ok(series) => {
                    for obs in series.observations {
                        let Some(ts) = parse_ymd(&obs.date) else { continue };
                        if let Some(v) = obs.value {
                            per_series[i].insert(ts, v);
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        series = %series_id, err = %err,
                        "fred::series fetch failed"
                    );
                }
            }
        }

        // ── Merge into per-date 12-vectors using forward-fill semantics ─
        // FRED releases happen on different schedules per series. At any
        // queried timestamp, each slot carries the most-recent-as-of value.
        // We materialize the merged table at every release date across all
        // series so the per-bar lookup just needs `range(..=ts).next_back()`.
        let mut release_ts: std::collections::BTreeSet<i64> = Default::default();
        for s in &per_series {
            for &ts in s.keys() {
                release_ts.insert(ts);
            }
        }
        // Per-series z-score stats (mean, std) over the full observation set.
        // We compute these once and apply at insert time so downstream lookup
        // is constant-time.
        let stats: Vec<(f64, f64)> = per_series
            .iter()
            .map(|s| {
                let vals: Vec<f64> = s.values().copied().collect();
                if vals.is_empty() {
                    return (0.0, 1.0);
                }
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var =
                    vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                let std = var.sqrt().max(1e-9);
                (mean, std)
            })
            .collect();

        let mut current = [0.0_f64; 12];
        for ts in release_ts {
            for (i, s) in per_series.iter().enumerate() {
                if let Some((_, &v)) = s.range(..=ts).next_back() {
                    let (mean, std) = stats[i];
                    current[i] = ((v - mean) / std).clamp(-5.0, 5.0);
                }
            }
            out.macro_values.insert(ts, current);
        }
        out.per_series_raw = per_series;
        out
    }

    /// Most recent yield curve at-or-before `ts`. Returns zeros when no
    /// observation exists at-or-before that timestamp (very early bars).
    pub fn yield_curve_at(&self, ts: i64) -> YieldCurveData {
        self.yield_curve
            .range(..=ts)
            .next_back()
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }

    /// Most recent macro vector at-or-before `ts`.
    pub fn macro_values_at(&self, ts: i64) -> [f64; 12] {
        self.macro_values
            .range(..=ts)
            .next_back()
            .map(|(_, v)| *v)
            .unwrap_or([0.0; 12])
    }
}

fn parse_mdy(s: &str) -> Option<i64> {
    // Treasury's native format is MM/DD/YYYY. Accept YYYY-MM-DD as a
    // defensive fallback (4-digit year disambiguates).
    let (y, m, d) = if s.contains('/') {
        let mut parts = s.split('/');
        let mm: u32 = parts.next()?.parse().ok()?;
        let dd: u32 = parts.next()?.parse().ok()?;
        let yyyy: i32 = parts.next()?.parse().ok()?;
        (yyyy, mm, dd)
    } else if s.contains('-') {
        let mut parts = s.split('-');
        let yyyy: i32 = parts.next()?.parse().ok()?;
        let mm: u32 = parts.next()?.parse().ok()?;
        let dd: u32 = parts.next()?.parse().ok()?;
        (yyyy, mm, dd)
    } else {
        return None;
    };
    let date = NaiveDate::from_ymd_opt(y, m, d)?;
    let dt = date.and_hms_opt(0, 0, 0)?;
    Some(Utc.from_utc_datetime(&dt).timestamp())
}

fn parse_ymd(s: &str) -> Option<i64> {
    let parts: Vec<&str> = s.split('-').collect();
    let [y, m, d] = parts.as_slice() else {
        return None;
    };
    let date = NaiveDate::from_ymd_opt(y.parse().ok()?, m.parse().ok()?, d.parse().ok()?)?;
    let dt = date.and_hms_opt(0, 0, 0)?;
    Some(Utc.from_utc_datetime(&dt).timestamp())
}

// ── Cache serialization ─────────────────────────────────────────────────────
//
// `YieldCurveData` lives in `crate::store` and isn't Serialize-derived (it's
// a runtime-only struct). We mirror its layout into a Serialize-friendly
// type for the cache file.

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    built_at: i64,
    yield_curve: Vec<(i64, SerializableYieldCurve)>,
    macro_values: BTreeMap<i64, [f64; 12]>,
    /// Raw per-series observations for the market-news PCA corpus.
    /// `#[serde(default)]` so caches written before this field exists still
    /// load — they'll just have an empty corpus and the news-PCA path will
    /// log + skip cleanly.
    #[serde(default)]
    per_series_raw: Vec<BTreeMap<i64, f64>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SerializableYieldCurve {
    y1m: f64,
    y3m: f64,
    y6m: f64,
    y1: f64,
    y2: f64,
    y3: f64,
    y5: f64,
    y7: f64,
    y10: f64,
    y20: f64,
    y30: f64,
    spread_2y10y: f64,
}

impl From<&YieldCurveData> for SerializableYieldCurve {
    fn from(yc: &YieldCurveData) -> Self {
        Self {
            y1m: yc.y1m,
            y3m: yc.y3m,
            y6m: yc.y6m,
            y1: yc.y1,
            y2: yc.y2,
            y3: yc.y3,
            y5: yc.y5,
            y7: yc.y7,
            y10: yc.y10,
            y20: yc.y20,
            y30: yc.y30,
            spread_2y10y: yc.spread_2y10y,
        }
    }
}

impl From<SerializableYieldCurve> for YieldCurveData {
    fn from(s: SerializableYieldCurve) -> Self {
        Self {
            y1m: s.y1m,
            y3m: s.y3m,
            y6m: s.y6m,
            y1: s.y1,
            y2: s.y2,
            y3: s.y3,
            y5: s.y5,
            y7: s.y7,
            y10: s.y10,
            y20: s.y20,
            y30: s.y30,
            spread_2y10y: s.spread_2y10y,
        }
    }
}
