//! Local embedder client for Block 8 text features.
//!
//! Hits an OpenAI-compatible `/v1/embeddings` endpoint (defaulting to the
//! llama-swap server at `127.0.0.1:12434`, overridable via
//! `MERIDIAN_EMBED_ENDPOINT` / `MERIDIAN_EMBED_MODEL`) using `harrier-oss:0.6b`
//! (Microsoft multilingual embedding, 1024-dim, 32K context, last-token
//! pooling). The override lets `make collect` point at a CPU-pinned llama-server
//! so Block-8 embedding never touches the Thunderbolt eGPU. Snippet
//! outputs are cached on disk by `sha256(text)` so re-collection is free after
//! the first pass.
//!
//! Pooling for Block 8: time-decay weighted mean over snippets in the
//! trailing 90-day window, L2-normalize. Reduction to the 64-d Block-8 width is
//! done downstream by `text_pca` (a learned PCA basis) — harrier-oss is NOT a
//! Matryoshka model, so the pooled vector cannot simply be truncated.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const EMBED_DIM: usize = 1024;
const DECAY_DAYS: f32 = 30.0;

/// Per-input character cap. harrier-oss has a 32K-token context. EDGAR primary
/// documents can be far longer and are truncated to their lead (business /
/// risk-factor) sections. Overridable via `MERIDIAN_EMBED_MAX_CHARS`.
const DEFAULT_MAX_CHARS: usize = 80_000;
/// Pack inputs into one request up to this many characters total. The GSP/CUDA
/// hang that motivated the conservative 24K budget is now prevented server-side
/// by llama-server --ubatch-size 1024, so we can pack more text per request and
/// dramatically reduce the number of HTTP round-trips during collection.
/// Overridable via `MERIDIAN_EMBED_CHAR_BUDGET`.
const DEFAULT_REQUEST_CHAR_BUDGET: usize = 80_000;
/// Hard cap on items per request, bounding the all-headlines (many tiny inputs)
/// case. Overridable via `MERIDIAN_EMBED_MAX_ITEMS`.
const DEFAULT_MAX_BATCH_ITEMS: usize = 16;
/// Retry budget for a single request on transport errors / 5xx (e.g. the 502
/// llama-swap returns while it restarts a crashed backend). Overridable via
/// `MERIDIAN_EMBED_MAX_RETRIES`.
const DEFAULT_MAX_RETRIES: u32 = 6;
/// Exponential-backoff base / ceiling between retries (full-jitter). Overridable
/// via `MERIDIAN_EMBED_RETRY_BASE_MS` / `MERIDIAN_EMBED_RETRY_MAX_MS`.
const DEFAULT_RETRY_BASE_MS: u64 = 1_000;
const DEFAULT_RETRY_MAX_MS: u64 = 30_000;
/// HTTP client timeout. Overridable via `MERIDIAN_EMBED_TIMEOUT_SECS`.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Pause between successive embedding requests (millis). Set to 0 — the GSP/CUDA
/// hang this was mitigating is now prevented server-side by --ubatch-size 1024.
/// Overridable via `MERIDIAN_EMBED_BATCH_PAUSE_MS`.
const DEFAULT_BATCH_PAUSE_MS: u64 = 0;

#[derive(Clone, Debug)]
pub struct EmbedderConfig {
    pub endpoint: String,
    pub model: String,
    pub cache_dir: PathBuf,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            endpoint: env_string(
                "MERIDIAN_EMBED_ENDPOINT",
                "http://127.0.0.1:12434/v1/embeddings",
            ),
            model: env_string("MERIDIAN_EMBED_MODEL", "harrier-oss:0.6b"),
            cache_dir: PathBuf::from("data/embed_cache"),
        }
    }
}

pub struct Embedder {
    cfg: EmbedderConfig,
    http: reqwest::Client,
    max_chars: usize,
    request_char_budget: usize,
    max_batch_items: usize,
    max_retries: u32,
    retry_base_ms: u64,
    retry_max_ms: u64,
    /// Pause between successive embedding requests (millis).
    batch_pause_ms: u64,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

impl Embedder {
    pub fn new(cfg: EmbedderConfig) -> Result<Self> {
        std::fs::create_dir_all(&cfg.cache_dir)
            .with_context(|| format!("create embed cache dir {:?}", cfg.cache_dir))?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(env_u64(
                "MERIDIAN_EMBED_TIMEOUT_SECS",
                DEFAULT_TIMEOUT_SECS,
            )))
            .build()?;
        Ok(Self {
            cfg,
            http,
            max_chars: env_usize("MERIDIAN_EMBED_MAX_CHARS", DEFAULT_MAX_CHARS).max(1),
            request_char_budget: env_usize("MERIDIAN_EMBED_CHAR_BUDGET", DEFAULT_REQUEST_CHAR_BUDGET)
                .max(1),
            max_batch_items: env_usize("MERIDIAN_EMBED_MAX_ITEMS", DEFAULT_MAX_BATCH_ITEMS).max(1),
            max_retries: env_u32("MERIDIAN_EMBED_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            retry_base_ms: env_u64("MERIDIAN_EMBED_RETRY_BASE_MS", DEFAULT_RETRY_BASE_MS).max(1),
            retry_max_ms: env_u64("MERIDIAN_EMBED_RETRY_MAX_MS", DEFAULT_RETRY_MAX_MS).max(1),
            batch_pause_ms: env_u64("MERIDIAN_EMBED_BATCH_PAUSE_MS", DEFAULT_BATCH_PAUSE_MS),
        })
    }

    fn cache_path(&self, hash: &str) -> PathBuf {
        self.cfg.cache_dir.join(&hash[..2]).join(hash)
    }

    fn read_cached(&self, path: &Path) -> Option<Vec<f32>> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() != EMBED_DIM * 4 {
            return None;
        }
        let mut out = Vec::with_capacity(EMBED_DIM);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().ok()?));
        }
        Some(out)
    }

    fn write_cached(&self, path: &Path, vec: &[f32]) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut bytes = Vec::with_capacity(vec.len() * 4);
        for &v in vec {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Embed a slice of strings, hitting cache first. Uncached items are
    /// requested in batches of `BATCH_SIZE`. Output is in input order.
    pub async fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut hashes: Vec<String> = Vec::with_capacity(texts.len());
        let mut misses: Vec<usize> = Vec::new();

        for (i, t) in texts.iter().enumerate() {
            let h = hex_sha256(t);
            let path = self.cache_path(&h);
            if let Some(v) = self.read_cached(&path) {
                out[i] = Some(v);
            } else {
                misses.push(i);
            }
            hashes.push(h);
        }

        // Greedily pack each request up to `request_char_budget` (and
        // `max_batch_items`) so large documents don't all decode in one batch — the
        // CUDA/GSP hang seen under big batches. A single oversized input still goes
        // through alone (truncated to `max_chars`). The disk cache key remains
        // sha256(full source text); only the sent payload is truncated.
        let mut cursor = 0usize;
        while cursor < misses.len() {
            let mut batch: Vec<usize> = Vec::new();
            let mut budget = 0usize;
            while cursor < misses.len() && batch.len() < self.max_batch_items {
                let idx = misses[cursor];
                let len = texts[idx].chars().count().min(self.max_chars);
                if !batch.is_empty() && budget + len > self.request_char_budget {
                    break;
                }
                budget += len;
                batch.push(idx);
                cursor += 1;
            }

            let inputs: Vec<String> = batch
                .iter()
                .map(|&i| truncate_chars(&texts[i], self.max_chars))
                .collect();
            let resp = self.embed_batch_resilient(&inputs).await?;
            anyhow::ensure!(
                resp.data.len() == batch.len(),
                "embed response len {} != batch {}",
                resp.data.len(),
                batch.len()
            );
            for (slot, item) in batch.iter().zip(resp.data.into_iter()) {
                anyhow::ensure!(
                    item.embedding.len() == EMBED_DIM,
                    "embed dim {} != {}",
                    item.embedding.len(),
                    EMBED_DIM
                );
                let path = self.cache_path(&hashes[*slot]);
                self.write_cached(&path, &item.embedding)?;
                out[*slot] = Some(item.embedding);
            }

            // Pause between batches — gives the GSP RPC channel over
            // Thunderbolt time to drain before the next request.
            if cursor < misses.len() && self.batch_pause_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.batch_pause_ms)).await;
            }
        }

        Ok(out.into_iter().map(|o| o.expect("all slots filled")).collect())
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_many(std::slice::from_ref(&text.to_string())).await?;
        Ok(v.into_iter().next().unwrap())
    }

    /// POST a batch, transparently recovering from a context-window overflow that
    /// `post_embeddings` surfaces as `ContextExceededError` (a single EDGAR filing
    /// can tokenize past harrier's 32K window even within the per-item char cap,
    /// because dense XBRL/tables run well under the assumed chars-per-token). A
    /// multi-item batch is bisected and each half retried independently, then
    /// recombined in input order; a single oversized input has its character
    /// budget halved until the backend accepts it (the 32K window always admits a
    /// few-KB lead, so this terminates above the floor). Rows stay 1:1 with
    /// `inputs`, in order — the caller never sees a dropped or reordered result.
    fn embed_batch_resilient<'a>(
        &'a self,
        inputs: &'a [String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<EmbedResponse>> + Send + 'a>> {
        Box::pin(async move {
            match self.post_embeddings(inputs).await {
                Ok(resp) => Ok(resp),
                Err(e) if e.downcast_ref::<ContextExceededError>().is_some() => {
                    if inputs.len() > 1 {
                        let mid = inputs.len() / 2;
                        let mut left = self.embed_batch_resilient(&inputs[..mid]).await?;
                        let right = self.embed_batch_resilient(&inputs[mid..]).await?;
                        left.data.extend(right.data);
                        Ok(left)
                    } else {
                        // Single input over the window: shrink its char budget by a
                        // quarter each attempt (keep 3/4) so we retain as much lead
                        // text as possible, until the backend accepts it (or we
                        // bottom out at the floor).
                        const FLOOR_CHARS: usize = 2_000;
                        let mut budget = inputs[0].chars().count() * 3 / 4;
                        loop {
                            let trimmed = truncate_chars(&inputs[0], budget.max(FLOOR_CHARS));
                            match self.post_embeddings(std::slice::from_ref(&trimmed)).await {
                                Ok(resp) => {
                                    tracing::warn!(
                                        kept_chars = trimmed.chars().count(),
                                        "embedder input exceeded context; truncated and retried"
                                    );
                                    return Ok(resp);
                                }
                                Err(e) if e.downcast_ref::<ContextExceededError>().is_some() => {
                                    if budget <= FLOOR_CHARS {
                                        return Err(e).context(
                                            "embedder input exceeds context even at floor budget",
                                        );
                                    }
                                    budget = budget * 3 / 4;
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                Err(e) => Err(e),
            }
        })
    }

    /// POST one batch to the embeddings endpoint with full-jitter exponential
    /// backoff. Transport errors (connect/timeout) and retryable HTTP statuses
    /// (5xx, 429) are retried up to `max_retries`; the 502 llama-swap returns
    /// while it restarts a crashed backend is the common case. A 4xx (e.g. a
    /// malformed request) fails immediately — retrying it is pointless.
    async fn post_embeddings(&self, inputs: &[String]) -> Result<EmbedResponse> {
        let req = EmbedRequest { model: &self.cfg.model, input: inputs };
        let body = serde_json::to_vec(&req).context("serialize embed request")?;

        let mut attempt: u32 = 0;
        loop {
            let send = self
                .http
                .post(&self.cfg.endpoint)
                .header("Content-Type", "application/json")
                .body(body.clone())
                .send()
                .await;

            let reason: String = match send {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let text = resp.text().await.context("read embedding response")?;
                        return serde_json::from_str(&text).context("decode embedding response");
                    }
                    let body: String = resp.text().await.unwrap_or_default();
                    // A context-window overflow is neither retryable (the input is
                    // deterministically too long) nor a hard failure — surface it as
                    // ContextExceededError so `embed_batch_resilient` can truncate
                    // the offending input and retry instead of dropping the filing.
                    if is_context_exceeded(&body) {
                        return Err(anyhow::Error::new(ContextExceededError));
                    }
                    let snippet: String = body.chars().take(160).collect();
                    if is_retryable_status(status) {
                        format!("HTTP {} ({})", status.as_u16(), snippet.trim())
                    } else {
                        anyhow::bail!(
                            "embedder returned non-retryable HTTP {}: {}",
                            status.as_u16(),
                            snippet.trim()
                        );
                    }
                }
                Err(e) if is_retryable_err(&e) => format!("transport error: {e}"),
                Err(e) => {
                    return Err(anyhow::Error::new(e)
                        .context(format!("POST {}", self.cfg.endpoint)));
                }
            };

            if attempt >= self.max_retries {
                anyhow::bail!(
                    "embedder request failed after {} retries (batch={} items): {}",
                    self.max_retries,
                    inputs.len(),
                    reason
                );
            }
            let delay = backoff_delay(self.retry_base_ms, self.retry_max_ms, attempt);
            tracing::warn!(
                attempt = attempt + 1,
                max_retries = self.max_retries,
                backoff_ms = delay.as_millis() as u64,
                items = inputs.len(),
                "embedder request retrying: {reason}"
            );
            tokio::time::sleep(delay).await;
            attempt += 1;
        }
    }

    /// Time-decay weighted mean pool of snippet embeddings → full 1024-dim
    /// L2-normalized vector. `snippets` is `(text, days_since_event)`.
    /// Empty input → zero vector (caller decides default-embedding policy).
    /// Reduction to the 64-d Block-8 width is the caller's job via `text_pca`
    /// (harrier-oss is not a Matryoshka model, so this vector is not truncated).
    pub async fn embed_pool_decayed(
        &self,
        snippets: &[(String, i64)],
    ) -> Result<[f32; EMBED_DIM]> {
        if snippets.is_empty() {
            return Ok([0.0; EMBED_DIM]);
        }
        let texts: Vec<String> = snippets.iter().map(|(t, _)| t.clone()).collect();
        let embs = self.embed_many(&texts).await?;

        let mut acc = vec![0.0f32; EMBED_DIM];
        let mut wsum = 0.0f32;
        for ((_, days), emb) in snippets.iter().zip(embs.iter()) {
            let w = (-((*days).max(0) as f32) / DECAY_DAYS).exp();
            for (a, &e) in acc.iter_mut().zip(emb.iter()) {
                *a += w * e;
            }
            wsum += w;
        }
        if wsum > 0.0 {
            for a in &mut acc {
                *a /= wsum;
            }
        }
        l2_normalize(&mut acc);

        let mut out = [0.0f32; EMBED_DIM];
        out.copy_from_slice(&acc);
        Ok(out)
    }
}

/// Retryable HTTP statuses: any 5xx (502/503/504 from llama-swap restarting a
/// crashed backend, 500 from a transient decode failure) plus 429 throttling.
fn is_retryable_status(s: reqwest::StatusCode) -> bool {
    s.is_server_error() || s == reqwest::StatusCode::TOO_MANY_REQUESTS
}

/// Retryable transport errors: connection refused/reset (backend bounced) and
/// request timeouts. A body/decode error is not retried here.
fn is_retryable_err(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request()
}

/// Marker error: the embeddings backend rejected a request because an input
/// exceeded the model's context window. Caught by `embed_batch_resilient`, which
/// truncates the offending input and retries rather than failing the symbol's
/// whole Block-8 text floor.
#[derive(Debug)]
struct ContextExceededError;

impl std::fmt::Display for ContextExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("embedder input exceeds model context window")
    }
}

impl std::error::Error for ContextExceededError {}

/// Detect a context-window overflow from the backend's error body. harrier /
/// llama-server reports it as a 400 `exceed_context_size_error` ("… exceeds the
/// available context size …") or, when it surfaces mid-decode, a 500 whose body
/// reads "Context size has been exceeded."
fn is_context_exceeded(body: &str) -> bool {
    body.contains("exceed_context_size_error")
        || body.contains("exceeds the available context size")
        || body.contains("Context size has been exceeded")
}

/// Full-jitter exponential backoff: capped exponential ceiling, then a uniform
/// random wait in `[ceil/2, ceil]` to avoid synchronized retries.
fn backoff_delay(base_ms: u64, max_ms: u64, attempt: u32) -> Duration {
    let ceil = base_ms
        .saturating_mul(1u64.checked_shl(attempt.min(20)).unwrap_or(u64::MAX))
        .min(max_ms)
        .max(1);
    let half = ceil / 2;
    let jitter = half + (pseudo_rand_u64() % (half + 1));
    Duration::from_millis(jitter)
}

/// Cheap non-crypto PRNG (splitmix64 seeded from the clock) for backoff jitter.
fn pseudo_rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Truncate to at most `max` characters on a char boundary (no-op if shorter).
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

fn hex_sha256(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let bytes = h.finalize();
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_respects_ceiling() {
        let (base, max) = (1_000u64, 30_000u64);
        // Full-jitter keeps each draw within [ceil/2, ceil]; sample repeatedly.
        for attempt in 0..16u32 {
            let ceil = base.saturating_mul(1u64 << attempt.min(20)).min(max).max(1);
            for _ in 0..64 {
                let ms = backoff_delay(base, max, attempt).as_millis() as u64;
                assert!(ms >= ceil / 2, "attempt {attempt}: {ms} < {}", ceil / 2);
                assert!(ms <= ceil, "attempt {attempt}: {ms} > {ceil}");
            }
        }
    }

    #[test]
    fn backoff_never_overflows_at_high_attempts() {
        // Large attempt counts must not panic via shift overflow.
        let ms = backoff_delay(1_000, 30_000, 1_000).as_millis() as u64;
        assert!(ms <= 30_000);
    }

    #[test]
    fn retryable_status_classification() {
        use reqwest::StatusCode;
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY)); // 502 — the llama-swap case
        assert!(is_retryable_status(StatusCode::SERVICE_UNAVAILABLE));
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST)); // 400 — don't retry
        assert!(!is_retryable_status(StatusCode::OK));
    }

    #[test]
    fn truncate_chars_is_char_safe() {
        let s = "naïve café — résumé";
        let t = truncate_chars(s, 5);
        assert_eq!(t.chars().count(), 5);
        assert_eq!(truncate_chars(s, 1000), s);
    }
}
