//! Live Block-8 text embedding via the shared harrier-oss embedder + PCA basis.
//!
//! This is the inference-path twin of the historical `compute_text_embedding_at`
//! (`src/historical/batch.rs`): for each symbol it gathers the recent RSS
//! headlines that mention the ticker, embeds them with the SAME local embedder
//! (`harrier-oss`, 1024-d), decay-pools over the trailing news window,
//! L2-normalizes, then reduces to the 64-d Block-8 width through the SAME
//! cached PCA basis fit by the `fit-text-pca` pre-pass. The result is written to
//! `store.text_embedding`, which the 1-minute assembler reads synchronously.
//!
//! Train/inference alignment (spec §6.2 Block 8): identical embedder, identical
//! decay pooling, identical PCA basis. No basis cached, embedder down, or no
//! matching headlines ⇒ the entry is left untouched and the assembler falls back
//! to honest zeros — never a fabricated or stale signal.

use crate::historical::embedder::{Embedder, EmbedderConfig};
use crate::historical::text_pca;
use crate::poller::news_sentiment::title_mentions_symbol;
use crate::store::DataStore;
use anyhow::Result;
use std::sync::Arc;

/// Trailing window for pooled headlines — matches `TEXT_WINDOW_SECS` in
/// `src/historical/batch.rs` (90 days) so live and historical Block 8 pool over
/// the same horizon.
const WINDOW_SECS: i64 = 90 * 86_400;
/// Cap on headlines pooled per symbol per pass (bounds the embedding calls).
const MAX_ARTICLES_PER_SYMBOL: usize = 32;

pub async fn poll_text_embeddings(store: &Arc<DataStore>) -> Result<()> {
    // Require the shared PCA basis (fit once by `make text-pca`). Without it we
    // cannot reduce 1024 → 64 on the same axes as training, so we leave Block 8
    // as honest zeros rather than inventing an incompatible per-run basis.
    let Some(pca) = text_pca::load_cached() else {
        tracing::debug!(
            "block8_text_pca basis not cached — live Block 8 stays zero (run `make text-pca`)"
        );
        return Ok(());
    };

    let embedder = Embedder::new(EmbedderConfig::default())?;

    let now = chrono::Utc::now();
    let now_ts = now.timestamp();
    let cutoff = now_ts - WINDOW_SECS;
    let articles = store.rss_feed.read().value.recent_articles.clone();

    for symbol in &store.symbols {
        // `(headline, days_since)` for articles in-window that mention the ticker.
        let snippets: Vec<(String, i64)> = articles
            .iter()
            .filter(|a| {
                a.fetched_at.timestamp() >= cutoff && title_mentions_symbol(&a.title, symbol)
            })
            .take(MAX_ARTICLES_PER_SYMBOL)
            .map(|a| {
                let days = ((now_ts - a.fetched_at.timestamp()).max(0)) / 86_400;
                (a.title.clone(), days)
            })
            .collect();

        if snippets.is_empty() {
            continue;
        }

        // Pool → 1024-d L2-normalized (decay handled inside embed_pool_decayed).
        let pooled = match embedder.embed_pool_decayed(&snippets).await {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(symbol = %symbol, err = %e, "block8 embed failed");
                continue;
            }
        };
        if pooled.iter().all(|x| *x == 0.0) {
            continue;
        }

        // Reduce 1024 → 64 through the shared basis, then re-normalize — exactly
        // the historical `compute_text_embedding_at` tail.
        let mut block8 = [0.0f32; text_pca::TEXT_PCA_DIM];
        pca.project_into(&pooled, &mut block8);
        let norm: f32 = block8.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in &mut block8 {
                *x /= norm;
            }
        }

        store.text_embedding.insert(symbol.clone(), block8);
    }

    Ok(())
}
