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
//! the local `Embedder`, then dimensionality-reduced from 768 → 12 by a fixed
//! PCA fit once over the full corpus and persisted to disk.
//!
//! Lookup at runtime: forward-fill against a `BTreeMap<release_ts,
//! [f32; 12]>` whose values are the trailing-90d time-decay-weighted mean of
//! release projections at that date.
//!
//! PCA is fit by **NIPALS** (power iteration with deflation). Pure scalar
//! arithmetic, no LAPACK / BLAS dependency. Convergence ε = 1e-6 or 200
//! iterations per component.

use crate::historical::embedder::{EMBED_DIM, Embedder};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const NEWS_DIM: usize = 12;
const PCA_MAX_ITER: usize = 200;
const PCA_TOL: f32 = 1e-6;
const TRAILING_SECS: i64 = 90 * 86_400;
const DECAY_DAYS: f32 = 30.0;

/// Persisted PCA model: per-feature mean + 12 unit-norm row components.
#[derive(Serialize, Deserialize)]
pub struct PcaModel {
    pub mean: Vec<f32>,             // length EMBED_DIM
    pub components: Vec<Vec<f32>>,  // 12 × EMBED_DIM, each row L2-unit
}

impl PcaModel {
    pub fn cache_path() -> PathBuf {
        let root = std::env::var("MERIDIAN_DATA_ROOT")
            .unwrap_or_else(|_| "./data".to_string());
        PathBuf::from(root).join("market_news_pca.bin")
    }

    pub fn try_load(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let m: PcaModel = bincode::deserialize(&bytes).ok()?;
        if m.mean.len() != EMBED_DIM || m.components.len() != NEWS_DIM {
            return None;
        }
        if m.components.iter().any(|c| c.len() != EMBED_DIM) {
            return None;
        }
        Some(m)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = bincode::serialize(self).context("serialize PcaModel")?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Fit PCA(12) via NIPALS on `data` (n_samples × EMBED_DIM).
    /// Skips fitting on degenerate input (n < 2) and falls back to a zero
    /// model — caller should treat that as "PCA unavailable, leave block 0".
    pub fn fit(data: &[Vec<f32>]) -> Self {
        if data.len() < 2 {
            return Self::zeros();
        }
        let n = data.len();
        // Center by per-column mean.
        let mut mean = vec![0.0f32; EMBED_DIM];
        for row in data {
            for (m, &x) in mean.iter_mut().zip(row.iter()) {
                *m += x;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }
        let mut x: Vec<Vec<f32>> = data
            .iter()
            .map(|row| {
                row.iter().zip(mean.iter()).map(|(a, m)| a - m).collect()
            })
            .collect();

        let mut components: Vec<Vec<f32>> = Vec::with_capacity(NEWS_DIM);
        for _k in 0..NEWS_DIM {
            // Initial t: any column with non-zero variance.
            let mut t: Vec<f32> = x.iter().map(|row| row.first().copied().unwrap_or(0.0)).collect();
            if t.iter().all(|v| v.abs() < 1e-12) {
                // pick the column with max variance instead
                let mut best_c = 0usize;
                let mut best_var = -1.0f32;
                for c in 0..EMBED_DIM {
                    let m: f32 = x.iter().map(|r| r[c]).sum::<f32>() / n as f32;
                    let v: f32 = x.iter().map(|r| (r[c] - m).powi(2)).sum::<f32>();
                    if v > best_var {
                        best_var = v;
                        best_c = c;
                    }
                }
                t = x.iter().map(|r| r[best_c]).collect();
            }
            let mut p = vec![0.0f32; EMBED_DIM];

            for _iter in 0..PCA_MAX_ITER {
                let t_norm_sq: f32 = t.iter().map(|v| v * v).sum::<f32>().max(1e-12);
                // p = X.T @ t / (t.T @ t)
                for c in 0..EMBED_DIM {
                    let mut s = 0.0f32;
                    for r in 0..n {
                        s += x[r][c] * t[r];
                    }
                    p[c] = s / t_norm_sq;
                }
                // p = p / ||p||
                let p_norm: f32 = p.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                for v in &mut p {
                    *v /= p_norm;
                }
                // t_new = X @ p
                let mut t_new = vec![0.0f32; n];
                for r in 0..n {
                    let mut s = 0.0f32;
                    for c in 0..EMBED_DIM {
                        s += x[r][c] * p[c];
                    }
                    t_new[r] = s;
                }
                let diff: f32 = t.iter().zip(t_new.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
                t = t_new;
                if diff < PCA_TOL {
                    break;
                }
            }

            // Deflate: X = X - t * p.T
            for r in 0..n {
                let tr = t[r];
                for c in 0..EMBED_DIM {
                    x[r][c] -= tr * p[c];
                }
            }
            components.push(p);
        }

        Self { mean, components }
    }

    pub fn zeros() -> Self {
        Self {
            mean: vec![0.0; EMBED_DIM],
            components: (0..NEWS_DIM).map(|_| vec![0.0; EMBED_DIM]).collect(),
        }
    }

    /// Project a 768-d embedding to 12-d via (emb - mean) · components.T.
    pub fn project(&self, emb: &[f32]) -> [f32; NEWS_DIM] {
        let mut out = [0.0f32; NEWS_DIM];
        if emb.len() != EMBED_DIM {
            return out;
        }
        for (k, comp) in self.components.iter().enumerate() {
            let mut s = 0.0f32;
            for c in 0..EMBED_DIM {
                s += (emb[c] - self.mean[c]) * comp[c];
            }
            out[k] = s;
        }
        out
    }
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
        projected.push((rel.ts, pca.project(emb)));
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
    let path = PcaModel::cache_path();
    if let Some(m) = PcaModel::try_load(&path) {
        tracing::info!(path = %path.display(), "market_news_pca: loaded from cache");
        return Ok(m);
    }
    if releases.is_empty() {
        tracing::warn!("market_news_pca: empty corpus, returning zero model");
        return Ok(PcaModel::zeros());
    }
    tracing::info!(corpus = releases.len(), "market_news_pca: fitting (cache miss)");
    let texts: Vec<String> = releases.iter().map(|r| r.text.clone()).collect();
    let embs = embedder.embed_many(&texts).await?;
    let m = PcaModel::fit(&embs);
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
    fn pca_recovers_dominant_axis() {
        // Synthetic: data lies on a 2D plane in 768-d ambient space.
        // Top component should align with the dominant axis.
        let n = 50;
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0f32; EMBED_DIM];
            v[0] = (i as f32) - (n as f32) / 2.0;        // dominant axis
            v[1] = ((i as f32) - (n as f32) / 2.0) * 0.1; // weaker axis
            data.push(v);
        }
        let m = PcaModel::fit(&data);
        // The first component should put almost all its weight on dim 0.
        let c0 = &m.components[0];
        assert!(c0[0].abs() > 0.99, "expected |c0[0]| ≈ 1, got {}", c0[0]);
        assert!(c0[2].abs() < 0.01);
    }

    #[test]
    fn project_zero_model_returns_zero() {
        let m = PcaModel::zeros();
        let e = vec![1.0f32; EMBED_DIM];
        let p = m.project(&e);
        assert!(p.iter().all(|&x| x == 0.0));
    }

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
