//! Pure-Rust PCA(12) over FRED-release text embeddings → 12-dim
//! market-news block.
//!
//! Spec §6.3 RSS & News Aggregate Signals (12 dims) requires sentiment-scored
//! market-wide news. finance-query exposes only current articles — no dated
//! historical archive — so for historical training we substitute with **FRED
//! macro release events**, which *are* dated, deterministic, and finance-query
//! native.
//!
//! Each FRED observation becomes a short text snippet
//! ("FEDFUNDS release 2024-03-15: rate 5.33%"). The snippet is embedded via
//! the local `Embedder`, then dimensionality-reduced from the embedder's native
//! width (1024-d for `harrier-oss`) → 12 by a fixed PCA fit once over the full
//! corpus and persisted to disk.
//!
//! Lookup at runtime: forward-fill against a `BTreeMap<release_ts,
//! [f32; 12]>` whose values are the trailing-90d time-decay-weighted mean of
//! release projections at that date.
//!
//! The NIPALS PCA core lives in `crate::historical::pca`.

use crate::historical::embedder::{EMBED_DIM, Embedder};
use crate::historical::pca::PcaModel;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const NEWS_DIM: usize = 12;
const TRAILING_SECS: i64 = 90 * 86_400;
const DECAY_DAYS: f32 = 30.0;

/// On-disk location of the persisted market-news PCA basis.
fn cache_path() -> PathBuf {
    let root = std::env::var("MERIDIAN_DATA_ROOT").unwrap_or_else(|_| "./data".to_string());
    PathBuf::from(root).join("market_news_pca.bin")
}

/// One FRED-derived release event: text snippet + release timestamp.
#[derive(Clone, Debug)]
pub struct ReleaseEvent {
    pub text: String,
    pub ts: i64,
}

/// Build a corpus of release events from the per-series observation maps
/// already collected by `MacroLookup`. Snippet templates are stable across
/// runs so the embedder cache hits 100% after the first collection.
pub fn build_release_corpus(
    fred_series: &[&str],
    series_labels: &[&str],
    per_series: &[BTreeMap<i64, f64>],
) -> Vec<ReleaseEvent> {
    let mut out = Vec::new();
    for (i, series_id) in fred_series.iter().enumerate() {
        let label = series_labels.get(i).copied().unwrap_or(series_id);
        for (&ts, &v) in &per_series[i] {
            let date = chrono::DateTime::from_timestamp(ts, 0)
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or_default();
            let text = format!("{} ({}) release {}: value {:.4}", label, series_id, date, v);
            out.push(ReleaseEvent { text, ts });
        }
    }
    out
}

/// Materialize a forward-fillable `BTreeMap<release_ts, [f32; 12]>`.
///
/// At each release date `r`, the value is the time-decay-weighted-mean of all
/// release projections in the trailing 90-day window, i.e. exactly the value
/// that would be emitted by `vector_at(r)`. Bars between release dates pick
/// up the most recent table entry via `range(..=ts).next_back()`.
pub async fn build_news_pca_lookup(
    embedder: &Embedder,
    pca: &PcaModel,
    releases: &[ReleaseEvent],
) -> Result<BTreeMap<i64, [f32; NEWS_DIM]>> {
    if releases.is_empty() {
        return Ok(BTreeMap::new());
    }
    let texts: Vec<String> = releases.iter().map(|r| r.text.clone()).collect();
    let embs = embedder.embed_many(&texts).await?;

    // Pre-project every release once.
    let mut projected: Vec<(i64, [f32; NEWS_DIM])> = Vec::with_capacity(releases.len());
    for (rel, emb) in releases.iter().zip(embs.iter()) {
        let mut v = [0.0f32; NEWS_DIM];
        pca.project_into(emb, &mut v);
        projected.push((rel.ts, v));
    }
    projected.sort_by_key(|(ts, _)| *ts);

    let mut out: BTreeMap<i64, [f32; NEWS_DIM]> = BTreeMap::new();
    for (i, &(r_ts, _)) in projected.iter().enumerate() {
        let cutoff = r_ts - TRAILING_SECS;
        // Walk backward from i to find the trailing-window slice (ts ascending).
        let mut start = i;
        while start > 0 && projected[start - 1].0 >= cutoff {
            start -= 1;
        }
        let mut acc = [0.0f32; NEWS_DIM];
        let mut wsum = 0.0f32;
        for j in start..=i {
            let (j_ts, j_vec) = &projected[j];
            let age_days = ((r_ts - j_ts).max(0) as f32) / 86_400.0;
            let w = (-age_days / DECAY_DAYS).exp();
            for (a, v) in acc.iter_mut().zip(j_vec.iter()) {
                *a += w * v;
            }
            wsum += w;
        }
        if wsum > 0.0 {
            for a in &mut acc {
                *a /= wsum;
            }
        }
        out.insert(r_ts, acc);
    }
    Ok(out)
}

/// Fit-or-load entry point. Tries the on-disk cache first; on miss, embeds
/// the corpus and fits NIPALS PCA(12), then persists.
pub async fn fit_or_load(
    embedder: &Embedder,
    releases: &[ReleaseEvent],
) -> Result<PcaModel> {
    let path = cache_path();
    if let Some(m) = PcaModel::try_load(&path, EMBED_DIM, NEWS_DIM) {
        tracing::info!(path = %path.display(), "market_news_pca: loaded from cache");
        return Ok(m);
    }
    if releases.is_empty() {
        tracing::warn!("market_news_pca: empty corpus, returning zero model");
        return Ok(PcaModel::zeros(EMBED_DIM, NEWS_DIM));
    }
    tracing::info!(corpus = releases.len(), "market_news_pca: fitting (cache miss)");
    let texts: Vec<String> = releases.iter().map(|r| r.text.clone()).collect();
    let embs = embedder.embed_many(&texts).await?;
    let m = PcaModel::fit(&embs, EMBED_DIM, NEWS_DIM);
    if let Err(e) = m.save(&path) {
        tracing::warn!(err = %e, "market_news_pca: failed to persist cache");
    } else {
        tracing::info!(path = %path.display(), "market_news_pca: cached");
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_release_corpus_dedups_per_series() {
        let mut s0 = BTreeMap::new();
        s0.insert(1_700_000_000, 5.0);
        s0.insert(1_700_086_400, 5.1);
        let per_series = vec![s0];
        let releases = build_release_corpus(&["FEDFUNDS"], &["Fed Funds"], &per_series);
        assert_eq!(releases.len(), 2);
        assert!(releases[0].text.contains("Fed Funds"));
        assert!(releases[0].text.contains("FEDFUNDS"));
    }
}
