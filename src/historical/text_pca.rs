//! Block-8 text-embedding dimensionality reduction (spec §6.2 Block 8).
//!
//! `harrier-oss` outputs a 1024-d embedding and is **not** a Matryoshka model,
//! so the pooled snippet vector cannot be truncated to the 64-d Block-8 width —
//! doing so would discard untrained coordinates. Instead we learn a fixed
//! PCA(64) basis and project every pooled snippet embedding through it (see
//! `pca::PcaModel`).
//!
//! The basis is fit **once over the whole symbol universe** by the
//! `fit-text-pca` pre-pass (`meridian-bus fit-text-pca`, which `make collect`
//! runs after `fnspid` and before per-symbol collection). Per-symbol collection
//! only *loads* the cached basis via [`load_cached`] — it never fits, because a
//! per-symbol fit would silently give each asset incompatible axes and the
//! backbone shares weights across assets.

use crate::historical::embedder::{EMBED_DIM, Embedder};
use crate::historical::fnspid;
use crate::historical::pca::PcaModel;
use anyhow::Result;
use std::collections::HashSet;
use std::path::PathBuf;

/// Block-8 text width. Matches `PER_ASSET_TEXT_DIM` in
/// `python/train/feature_schema.py` and `FeatureRow::text`.
pub const TEXT_PCA_DIM: usize = 64;

/// Cap on snippets used to *fit* the basis. A 64-component basis is well
/// determined by a few thousand samples; capping keeps the scalar NIPALS fit
/// tractable. Every snippet is still embedded during collection regardless of
/// this cap — it only bounds the fit, not the data.
const PCA_FIT_MAX_SAMPLES: usize = 8192;

pub fn cache_path() -> PathBuf {
    let root = std::env::var("MERIDIAN_DATA_ROOT").unwrap_or_else(|_| "./data".to_string());
    PathBuf::from(root).join("block8_text_pca.bin")
}

/// Load the persisted basis if present and shape-valid. Returns `None` when the
/// `fit-text-pca` pre-pass has not been run — callers should treat that as
/// "Block 8 unavailable" rather than fitting a local (incompatible) basis.
pub fn load_cached() -> Option<PcaModel> {
    PcaModel::try_load(&cache_path(), EMBED_DIM, TEXT_PCA_DIM)
}

/// Fit the global Block-8 PCA basis over the FNSPID news corpus for the whole
/// `symbols` universe and persist it to [`cache_path`].
///
/// EDGAR filing text (fetched per-symbol at collection time) is projected onto
/// this news-fit basis — both are financial English, so the directions
/// generalize. A degenerate corpus (< 2 snippets) or an unavailable embedder
/// writes a zero basis so Block 8 collapses to honest zeros rather than a
/// fabricated or single-symbol-biased signal.
pub async fn fit_universe(symbols: &[String], embedder: &Embedder) -> Result<PcaModel> {
    // Gather unique news snippets across the universe (insertion order keeps
    // symbols grouped, so the stride-sample below spreads across the universe).
    let mut seen: HashSet<String> = HashSet::new();
    let mut corpus: Vec<String> = Vec::new();
    for sym in symbols {
        for (_ts, text) in fnspid::load_news(sym) {
            if seen.insert(text.clone()) {
                corpus.push(text);
            }
        }
    }

    if corpus.len() < 2 {
        tracing::warn!(
            snippets = corpus.len(),
            "fit-text-pca: corpus < 2 snippets (no FNSPID news?); writing zero basis — Block 8 → zeros"
        );
        let m = PcaModel::zeros(EMBED_DIM, TEXT_PCA_DIM);
        m.save(&cache_path())?;
        return Ok(m);
    }

    // Deterministic stride-sample down to the fit cap so the basis is built from
    // a universe-wide spread without an unbounded NIPALS cost.
    let sample: Vec<String> = if corpus.len() > PCA_FIT_MAX_SAMPLES {
        let stride = (corpus.len() / PCA_FIT_MAX_SAMPLES).max(1);
        corpus
            .iter()
            .step_by(stride)
            .take(PCA_FIT_MAX_SAMPLES)
            .cloned()
            .collect()
    } else {
        corpus.clone()
    };

    tracing::info!(
        universe_snippets = corpus.len(),
        fit_samples = sample.len(),
        "fit-text-pca: embedding fit sample"
    );

    let embs = match embedder.embed_many(&sample).await {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(
                err = %e,
                "fit-text-pca: embedding failed (is the embedder endpoint up?); \
                 writing zero basis — Block 8 → zeros"
            );
            let m = PcaModel::zeros(EMBED_DIM, TEXT_PCA_DIM);
            m.save(&cache_path())?;
            return Ok(m);
        }
    };

    tracing::info!(
        samples = embs.len(),
        dim = EMBED_DIM,
        k = TEXT_PCA_DIM,
        "fit-text-pca: fitting PCA basis"
    );
    let m = PcaModel::fit(&embs, EMBED_DIM, TEXT_PCA_DIM);
    let path = cache_path();
    m.save(&path)?;
    tracing::info!(path = %path.display(), "fit-text-pca: basis persisted");
    Ok(m)
}
