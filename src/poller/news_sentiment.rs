//! Per-asset news sentiment via llama-swap (`/v1/chat/completions`).
//!
//! Every 5 minutes: filters the RSS article buffer for per-symbol headline
//! mentions, scores each unseen article with a small GGUF sentiment model,
//! and writes a time-decay weighted average to `store.news_sentiment`.
//!
//! ## Model
//!
//! Defaults to `qwen2.5:3b` (Qwen2.5 3B Instruct) which must be present in
//! the llama-swap server. Override via the SENTIMENT_MODEL environment variable.

use crate::store::DataStore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc};

const ENDPOINT: &str = "http://127.0.0.1:12434/v1/chat/completions";
const MODEL_ENV: &str = "SENTIMENT_MODEL";
const DEFAULT_MODEL: &str = "qwen2.5:3b";
const CACHE_DIR: &str = "data/sentiment_cache";
const WINDOW_HOURS: i64 = 24;
const MAX_ARTICLES_PER_SYMBOL: usize = 20;
/// Decay half-life for time-weighted average: 6 hours.
const DECAY_HOURS: f32 = 6.0;

pub async fn poll_news_sentiment(store: &Arc<DataStore>) -> Result<()> {
    let model = std::env::var(MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    std::fs::create_dir_all(CACHE_DIR).ok();

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
            let score = match score_title(title, &model, &http).await {
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
            store.news_sentiment.insert(symbol.clone(), (weighted_sum / weight_total).clamp(-1.0, 1.0));
        }
    }

    Ok(())
}

/// Whole-word, case-insensitive ticker match in the article title.
fn title_mentions_symbol(title: &str, symbol: &str) -> bool {
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

/// Score a headline; returns -1.0, 0.0, or +1.0.
/// Results are disk-cached by sha256(model+title) — only new articles hit the LLM.
async fn score_title(title: &str, model: &str, http: &reqwest::Client) -> Result<f32> {
    let hash = hex_sha256(&format!("{model}\x00{title}"));
    let cache_path = PathBuf::from(CACHE_DIR).join(&hash[..2]).join(&hash);

    if let Ok(bytes) = std::fs::read(&cache_path) {
        if bytes.len() == 4 {
            if let Ok(arr) = bytes.try_into() {
                return Ok(f32::from_le_bytes(arr));
            }
        }
    }

    let score = call_llm(title, model, http).await?;

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cache_path, score.to_le_bytes()).ok();

    Ok(score)
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageOut,
}

#[derive(Deserialize)]
struct ChatMessageOut {
    content: String,
}

async fn call_llm(title: &str, model: &str, http: &reqwest::Client) -> Result<f32> {
    let req = ChatRequest {
        model,
        messages: vec![
            ChatMessage {
                role: "system",
                content: "You are a financial news sentiment classifier. Classify the headline sentiment as it relates to the mentioned company. Respond with exactly one word: positive, neutral, or negative.",
            },
            ChatMessage { role: "user", content: title },
        ],
        max_tokens: 4,
        temperature: 0.0,
    };

    let body = serde_json::to_vec(&req)?;
    let resp: ChatResponse = http
        .post(ENDPOINT)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let content = resp
        .choices
        .first()
        .map(|c| c.message.content.trim().to_lowercase())
        .unwrap_or_default();

    Ok(if content.contains("positive") {
        1.0
    } else if content.contains("negative") {
        -1.0
    } else {
        0.0
    })
}

fn hex_sha256(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}
