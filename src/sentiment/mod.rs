//! Shared financial-sentiment client — FinBERT (`ProsusAI/finbert`).
//!
//! ONE scorer for BOTH feature paths, so train and inference stay aligned:
//!   - live:       `src/poller/news_sentiment.rs` (RSS headlines)
//!   - historical: `src/historical/batch.rs` (EDGAR filing text)
//!
//! Hits a sequence-classification endpoint (`POST /predict`, body
//! `{"inputs": "<text>"}`) returning a `[{label, score}, …]` distribution. The
//! signed score is `P(positive) − P(negative)` clamped to `[-1, 1]`. Results are
//! disk-cached by `sha256(model + text)` as a single little-endian `f32` — the
//! same cache shape the embedder uses — so re-collection is free after the first
//! pass.
//!
//! The model is served by the FinBERT sidecar in `services/finbert/` (CPU-only
//! `transformers`, no GPU needed); container port 80 maps to the default 8088:
//! ```text
//! docker build -t meridian-finbert services/finbert
//! docker run -d --name meridian-finbert -p 8088:80 \
//!     -v "$PWD/data/finbert:/data" meridian-finbert
//! ```
//! Override the endpoint/model with `SENTIMENT_ENDPOINT` / `SENTIMENT_MODEL`.

use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8088/predict";
pub const DEFAULT_MODEL: &str = "ProsusAI/finbert";
const ENDPOINT_ENV: &str = "SENTIMENT_ENDPOINT";
const MODEL_ENV: &str = "SENTIMENT_MODEL";
const CACHE_DIR: &str = "data/sentiment_cache";
/// FinBERT (BERT) caps at 512 tokens; ~2000 chars stays comfortably under it.
const MAX_CHARS: usize = 2000;

#[derive(Clone, Debug)]
pub struct SentimentConfig {
    pub endpoint: String,
    pub model: String,
    pub cache_dir: PathBuf,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            endpoint: std::env::var(ENDPOINT_ENV).unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string()),
            model: std::env::var(MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            cache_dir: PathBuf::from(CACHE_DIR),
        }
    }
}

pub struct SentimentClient {
    cfg: SentimentConfig,
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct LabelScore {
    label: String,
    score: f32,
}

impl SentimentClient {
    pub fn new(cfg: SentimentConfig) -> Result<Self> {
        std::fs::create_dir_all(&cfg.cache_dir)
            .with_context(|| format!("create sentiment cache dir {:?}", cfg.cache_dir))?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        Ok(Self { cfg, http })
    }

    /// Reuse an existing HTTP client (e.g. the poller's) rather than build one.
    pub fn with_http(cfg: SentimentConfig, http: reqwest::Client) -> Result<Self> {
        std::fs::create_dir_all(&cfg.cache_dir)
            .with_context(|| format!("create sentiment cache dir {:?}", cfg.cache_dir))?;
        Ok(Self { cfg, http })
    }

    fn cache_path(&self, text: &str) -> PathBuf {
        let h = hex_sha256(&format!("{}\x00{}", self.cfg.model, text));
        self.cfg.cache_dir.join(&h[..2]).join(h)
    }

    /// Signed sentiment `P(pos) − P(neg)` in `[-1, 1]` for one text. Empty/blank
    /// text → `0.0` (neutral) without a network call. Disk-cached.
    pub async fn score(&self, text: &str) -> Result<f32> {
        let text = truncate_chars(text.trim(), MAX_CHARS);
        if text.is_empty() {
            return Ok(0.0);
        }
        let path = self.cache_path(&text);
        if let Some(s) = read_cached(&path) {
            return Ok(s);
        }
        let s = self.request(&text).await?;
        write_cached(&path, s);
        Ok(s)
    }

    async fn request(&self, text: &str) -> Result<f32> {
        let body = serde_json::json!({ "inputs": text });
        let resp_text = self
            .http
            .post(&self.cfg.endpoint)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {}", self.cfg.endpoint))?
            .error_for_status()?
            .text()
            .await
            .context("read sentiment response")?;
        let scores = parse_scores(&resp_text)
            .with_context(|| format!("decode sentiment response: {resp_text:.200}"))?;
        Ok(signed_score(&scores))
    }
}

/// TEI returns `[[{label,score},…]]` for a single input; some servers return a
/// flat `[{label,score},…]`. Accept either.
fn parse_scores(body: &str) -> Result<Vec<LabelScore>> {
    if let Ok(mut nested) = serde_json::from_str::<Vec<Vec<LabelScore>>>(body) {
        return Ok(nested.drain(..).next().unwrap_or_default());
    }
    serde_json::from_str::<Vec<LabelScore>>(body).context("not a label/score list")
}

/// `P(positive) − P(negative)`, clamped. Labels are matched case-insensitively
/// against the FinBERT class names (`positive` / `negative` / `neutral`).
fn signed_score(scores: &[LabelScore]) -> f32 {
    let mut pos = 0.0f32;
    let mut neg = 0.0f32;
    for s in scores {
        match s.label.to_ascii_lowercase().as_str() {
            "positive" => pos = s.score,
            "negative" => neg = s.score,
            _ => {}
        }
    }
    (pos - neg).clamp(-1.0, 1.0)
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

fn read_cached(path: &Path) -> Option<f32> {
    let bytes = std::fs::read(path).ok()?;
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(f32::from_le_bytes(arr))
}

fn write_cached(path: &Path, score: f32) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, score.to_le_bytes()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn hex_sha256(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_score_is_pos_minus_neg() {
        let s = vec![
            LabelScore { label: "positive".into(), score: 0.7 },
            LabelScore { label: "negative".into(), score: 0.2 },
            LabelScore { label: "neutral".into(), score: 0.1 },
        ];
        assert!((signed_score(&s) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn signed_score_clamps_and_ignores_unknown_labels() {
        let s = vec![
            LabelScore { label: "NEGATIVE".into(), score: 1.0 },
            LabelScore { label: "whatever".into(), score: 0.9 },
        ];
        assert_eq!(signed_score(&s), -1.0);
    }

    #[test]
    fn parse_scores_accepts_nested_and_flat() {
        let nested = r#"[[{"label":"positive","score":0.9},{"label":"negative","score":0.1}]]"#;
        let flat = r#"[{"label":"positive","score":0.9},{"label":"negative","score":0.1}]"#;
        assert_eq!(parse_scores(nested).unwrap().len(), 2);
        assert_eq!(parse_scores(flat).unwrap().len(), 2);
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let s = "café".repeat(1000);
        assert_eq!(truncate_chars(&s, 10).chars().count(), 10);
    }

    /// Live round-trip against the FinBERT sidecar (`services/finbert`). Ignored by
    /// default; run with the server up:
    ///   docker run -d -p 8088:80 -v "$PWD/data/finbert:/data" meridian-finbert
    ///   cargo test --lib sentiment -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn live_finbert_round_trip() {
        let client = SentimentClient::new(SentimentConfig::default()).unwrap();
        let pos = client.score("Profit beats estimates; company raises guidance.").await.unwrap();
        let neg = client.score("Firm slashes dividend amid mounting losses.").await.unwrap();
        eprintln!("pos={pos}  neg={neg}");
        assert!(pos > neg, "positive headline should score above negative");
        assert!((-1.0..=1.0).contains(&pos) && (-1.0..=1.0).contains(&neg));
        assert!(neg < -0.5, "clear negative should be strongly negative");
    }
}
