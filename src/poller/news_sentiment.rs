//! Per-asset news sentiment via the shared FinBERT client.
//!
//! Every 5 minutes: filters the RSS article buffer for per-symbol headline
//! mentions, scores each headline with the same `sentiment::SentimentClient`
//! the historical collector uses (so the train and inference sentiment slot
//! share one model), and writes a time-decay weighted average to
//! `store.news_sentiment`.

use crate::sentiment::{SentimentClient, SentimentConfig};
use crate::store::DataStore;
use anyhow::Result;
use std::sync::Arc;

const WINDOW_HOURS: i64 = 24;
const MAX_ARTICLES_PER_SYMBOL: usize = 20;
/// Decay half-life for time-weighted average: 6 hours.
const DECAY_HOURS: f32 = 6.0;

pub async fn poll_news_sentiment(store: &Arc<DataStore>) -> Result<()> {
    let client = SentimentClient::new(SentimentConfig::default())?;

    let now = chrono::Utc::now();
    let cutoff = now - chrono::Duration::hours(WINDOW_HOURS);
    let articles = store.rss_feed.read().value.recent_articles.clone();

    for symbol in &store.symbols {
        // Collect recent articles that mention the ticker in the headline
        let relevant: Vec<(i64, String)> = articles
            .iter()
            .filter(|a| a.fetched_at > cutoff && title_mentions_symbol(&a.title, symbol))
            .take(MAX_ARTICLES_PER_SYMBOL)
            .map(|a| (a.fetched_at.timestamp(), a.title.clone()))
            .collect();

        if relevant.is_empty() {
            continue;
        }

        let mut weighted_sum = 0.0f32;
        let mut weight_total = 0.0f32;

        for (ts, title) in &relevant {
            let score = match client.score(title).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!(symbol = %symbol, err = %e, "sentiment score failed");
                    continue;
                }
            };
            let hours_ago = ((now.timestamp() - ts) as f32 / 3600.0).max(0.0);
            let weight = (-hours_ago / DECAY_HOURS).exp();
            weighted_sum += score * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            store
                .news_sentiment
                .insert(symbol.clone(), (weighted_sum / weight_total).clamp(-1.0, 1.0));
        }
    }

    Ok(())
}

/// Whole-word, case-insensitive ticker match in the article title.
pub(crate) fn title_mentions_symbol(title: &str, symbol: &str) -> bool {
    let title_upper = title.to_uppercase();
    let sym_upper = symbol.to_uppercase();
    let sym_bytes = sym_upper.as_bytes();
    let title_bytes = title_upper.as_bytes();
    let sym_len = sym_bytes.len();

    let mut start = 0usize;
    while start + sym_len <= title_bytes.len() {
        if let Some(pos) = title_upper[start..].find(&sym_upper) {
            let abs = start + pos;
            let before_ok = abs == 0 || !title_bytes[abs - 1].is_ascii_alphanumeric();
            let after = abs + sym_len;
            let after_ok = after >= title_bytes.len() || !title_bytes[after].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
            start = abs + 1;
        } else {
            break;
        }
    }
    false
}
