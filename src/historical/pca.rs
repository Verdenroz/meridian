//! Pure-Rust NIPALS PCA shared by the Block-8 text encoder (`text_pca`) and the
//! market-news block (`market_news_pca`).
//!
//! Power iteration with deflation; no LAPACK / BLAS dependency. Convergence
//! ε = 1e-6 or 200 iterations per component.
//!
//! This is the right tool for reducing a **non-Matryoshka** embedding (such as
//! `harrier-oss`, whose 1024 coordinates are jointly trained with no nested
//! prefix structure) to a compact block: a learned orthonormal basis keeps the
//! variance-carrying directions, whereas naive dimension truncation would
//! discard untrained coordinates with no guarantee the first N carry signal.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

const PCA_MAX_ITER: usize = 200;
const PCA_TOL: f32 = 1e-6;

/// Persisted PCA model: per-feature mean + `k` unit-norm row components.
///
/// `mean.len()` is the input dimension; `components.len()` is `k`, and every
/// component row also has length `input_dim`.
#[derive(Clone, Serialize, Deserialize)]
pub struct PcaModel {
    pub mean: Vec<f32>,
    pub components: Vec<Vec<f32>>,
}

impl PcaModel {
    pub fn input_dim(&self) -> usize {
        self.mean.len()
    }

    pub fn k(&self) -> usize {
        self.components.len()
    }

    /// Zero model — `project_into` returns all-zeros. Caller should treat this
    /// as "PCA unavailable, leave block 0".
    pub fn zeros(input_dim: usize, k: usize) -> Self {
        Self {
            mean: vec![0.0; input_dim],
            components: (0..k).map(|_| vec![0.0; input_dim]).collect(),
        }
    }

    /// Load a persisted model, validating it matches the expected shape.
    pub fn try_load(path: &Path, input_dim: usize, k: usize) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let m: PcaModel = bincode::deserialize(&bytes).ok()?;
        if m.mean.len() != input_dim || m.components.len() != k {
            return None;
        }
        if m.components.iter().any(|c| c.len() != input_dim) {
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

    /// Fit `k`-component PCA via NIPALS on `data` (n_samples × `input_dim`).
    /// Degenerate input (n < 2) yields a zero model.
    pub fn fit(data: &[Vec<f32>], input_dim: usize, k: usize) -> Self {
        if data.len() < 2 {
            return Self::zeros(input_dim, k);
        }
        let n = data.len();
        // Center by per-column mean.
        let mut mean = vec![0.0f32; input_dim];
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
            .map(|row| row.iter().zip(mean.iter()).map(|(a, m)| a - m).collect())
            .collect();

        let mut components: Vec<Vec<f32>> = Vec::with_capacity(k);
        for _comp in 0..k {
            // Initial t: first column; fall back to the max-variance column if
            // that one is degenerate.
            let mut t: Vec<f32> = x.iter().map(|row| row.first().copied().unwrap_or(0.0)).collect();
            if t.iter().all(|v| v.abs() < 1e-12) {
                let mut best_c = 0usize;
                let mut best_var = -1.0f32;
                for c in 0..input_dim {
                    let m: f32 = x.iter().map(|r| r[c]).sum::<f32>() / n as f32;
                    let v: f32 = x.iter().map(|r| (r[c] - m).powi(2)).sum::<f32>();
                    if v > best_var {
                        best_var = v;
                        best_c = c;
                    }
                }
                t = x.iter().map(|r| r[best_c]).collect();
            }
            let mut p = vec![0.0f32; input_dim];

            for _iter in 0..PCA_MAX_ITER {
                let t_norm_sq: f32 = t.iter().map(|v| v * v).sum::<f32>().max(1e-12);
                // p = Xᵀ t / (tᵀ t)
                for c in 0..input_dim {
                    let mut s = 0.0f32;
                    for r in 0..n {
                        s += x[r][c] * t[r];
                    }
                    p[c] = s / t_norm_sq;
                }
                // p = p / ‖p‖
                let p_norm: f32 = p.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                for v in &mut p {
                    *v /= p_norm;
                }
                // t_new = X p
                let mut t_new = vec![0.0f32; n];
                for r in 0..n {
                    let mut s = 0.0f32;
                    for c in 0..input_dim {
                        s += x[r][c] * p[c];
                    }
                    t_new[r] = s;
                }
                let diff: f32 = t
                    .iter()
                    .zip(t_new.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                t = t_new;
                if diff < PCA_TOL {
                    break;
                }
            }

            // Deflate: X = X − t pᵀ
            for r in 0..n {
                let tr = t[r];
                for c in 0..input_dim {
                    x[r][c] -= tr * p[c];
                }
            }
            components.push(p);
        }

        Self { mean, components }
    }

    /// Project `emb` (length == `input_dim`) into the first `out.len()`
    /// components via `(emb − mean) · componentsᵀ`. Writes zeros on dimension
    /// mismatch.
    pub fn project_into(&self, emb: &[f32], out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        if emb.len() != self.input_dim() {
            return;
        }
        for (idx, comp) in self.components.iter().enumerate() {
            if idx >= out.len() {
                break;
            }
            let mut s = 0.0f32;
            for (c, &cw) in comp.iter().enumerate() {
                s += (emb[c] - self.mean[c]) * cw;
            }
            out[idx] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_recovers_dominant_axis() {
        // Synthetic: data lies on a 2D plane in 768-d ambient space.
        // Top component should align with the dominant axis.
        const DIM: usize = 768;
        let n = 50;
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0f32; DIM];
            v[0] = (i as f32) - (n as f32) / 2.0; // dominant axis
            v[1] = ((i as f32) - (n as f32) / 2.0) * 0.1; // weaker axis
            data.push(v);
        }
        let m = PcaModel::fit(&data, DIM, 12);
        let c0 = &m.components[0];
        assert!(c0[0].abs() > 0.99, "expected |c0[0]| ≈ 1, got {}", c0[0]);
        assert!(c0[2].abs() < 0.01);
    }

    #[test]
    fn project_zero_model_returns_zero() {
        const DIM: usize = 1024;
        let m = PcaModel::zeros(DIM, 64);
        let e = vec![1.0f32; DIM];
        let mut out = [0.0f32; 64];
        m.project_into(&e, &mut out);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn project_handles_dim_mismatch() {
        let m = PcaModel::zeros(1024, 64);
        let mut out = [1.0f32; 64];
        m.project_into(&[0.0f32; 10], &mut out); // wrong input dim
        assert!(out.iter().all(|&x| x == 0.0), "mismatch must zero the output");
    }
}
