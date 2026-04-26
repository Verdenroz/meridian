//! Local embedder client for Block 8 text features.
//!
//! Hits an OpenAI-compatible `/v1/embeddings` endpoint (defaulting to the
//! llama-swap server at `127.0.0.1:12434`) using `nomic-embed-text` (Nomic
//! v1.5, 768-dim, Matryoshka). Snippet outputs are cached on disk by
//! `sha256(text)` so re-collection is free after the first pass.
//!
//! Pooling for Block 8: time-decay weighted mean over snippets in the
//! trailing 90-day window, L2-normalize, Matryoshka-truncate to 64,
//! re-normalize.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const EMBED_DIM: usize = 768;
pub const TEXT_DIM: usize = 64;
const BATCH_SIZE: usize = 32;
const DECAY_DAYS: f32 = 30.0;

#[derive(Clone, Debug)]
pub struct EmbedderConfig {
    pub endpoint: String,
    pub model: String,
    pub cache_dir: PathBuf,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:12434/v1/embeddings".to_string(),
            model: "nomic-embed-text".to_string(),
            cache_dir: PathBuf::from("data/embed_cache"),
        }
    }
}

pub struct Embedder {
    cfg: EmbedderConfig,
    http: reqwest::Client,
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
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        Ok(Self { cfg, http })
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

        for chunk in misses.chunks(BATCH_SIZE) {
            let inputs: Vec<String> = chunk.iter().map(|&i| texts[i].clone()).collect();
            let req = EmbedRequest { model: &self.cfg.model, input: &inputs };
            let body = serde_json::to_vec(&req).context("serialize embed request")?;
            let resp_text = self
                .http
                .post(&self.cfg.endpoint)
                .header("Content-Type", "application/json")
                .body(body)
                .send()
                .await
                .with_context(|| format!("POST {}", self.cfg.endpoint))?
                .error_for_status()?
                .text()
                .await
                .context("read embedding response")?;
            let resp: EmbedResponse = serde_json::from_str(&resp_text)
                .context("decode embedding response")?;
            anyhow::ensure!(
                resp.data.len() == chunk.len(),
                "embed response len {} != batch {}",
                resp.data.len(),
                chunk.len()
            );
            for (slot, item) in chunk.iter().zip(resp.data.into_iter()) {
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
        }

        Ok(out.into_iter().map(|o| o.expect("all slots filled")).collect())
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_many(std::slice::from_ref(&text.to_string())).await?;
        Ok(v.into_iter().next().unwrap())
    }

    /// Time-decay weighted mean pool of snippet embeddings → 64-dim
    /// L2-normalized vector. `snippets` is `(text, days_since_event)`.
    /// Empty input → zero vector (caller decides default-embedding policy).
    pub async fn embed_pool_decayed(
        &self,
        snippets: &[(String, i64)],
    ) -> Result<[f32; TEXT_DIM]> {
        if snippets.is_empty() {
            return Ok([0.0; TEXT_DIM]);
        }
        let texts: Vec<String> = snippets.iter().map(|(t, _)| t.clone()).collect();
        let embs = self.embed_many(&texts).await?;

        let mut acc = vec![0.0f32; EMBED_DIM];
        let mut wsum = 0.0f32;
        for ((_, days), emb) in snippets.iter().zip(embs.iter()) {
            let w = (-(days.max(&0).clone() as f32) / DECAY_DAYS).exp();
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

        let mut out = [0.0f32; TEXT_DIM];
        out.copy_from_slice(&acc[..TEXT_DIM]);
        l2_normalize_arr(&mut out);
        Ok(out)
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

fn l2_normalize_arr<const N: usize>(v: &mut [f32; N]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v {
            *x /= norm;
        }
    }
}
