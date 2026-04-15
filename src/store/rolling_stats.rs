use anyhow::Context;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Welford online algorithm for running mean and variance.
///
/// Stores only three scalars — no ring buffer — so memory is O(1) per instance.
/// This is an expanding-window statistic (all samples since startup or last load).
/// For Phase 0 this is acceptable; a windowed EWMA variant can be swapped in later.
#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct RollingStats {
    pub mean: f64,
    pub m2: f64, // Welford's M2 accumulator
    pub count: u64,
}

impl RollingStats {
    /// Returns a new `RollingStats` updated with sample `x`. Never mutates.
    pub fn update(&self, x: f64) -> Self {
        let count = self.count + 1;
        let delta = x - self.mean;
        let mean = self.mean + delta / count as f64;
        let delta2 = x - mean;
        let m2 = self.m2 + delta * delta2;
        Self { mean, m2, count }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            1.0
        } else {
            self.m2 / self.count as f64
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Z-score of `x` relative to the running distribution. Returns 0 if insufficient data.
    pub fn zscore(&self, x: f64) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        ((x - self.mean) / std).clamp(-10.0, 10.0) as f32
    }
}

/// Thread-safe store of per-symbol, per-feature rolling statistics.
///
/// Key structure: outer map is symbol → inner map is feature_name → `RollingStats`.
/// Uses a `DashMap` for concurrent per-symbol access during parallel feature assembly.
pub struct RollingStatsStore {
    // symbol → (feature_name → RollingStats)
    inner: DashMap<String, HashMap<String, RollingStats>>,
}

impl RollingStatsStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: DashMap::new(),
        })
    }

    /// Update the stats for `(symbol, feature)` with a new sample, returning the updated stats.
    pub fn update(&self, symbol: &str, feature: &str, x: f64) -> RollingStats {
        let mut entry = self.inner.entry(symbol.to_string()).or_default();
        let stats = entry.get(feature).cloned().unwrap_or_default();
        let updated = stats.update(x);
        entry.insert(feature.to_string(), updated.clone());
        updated
    }

    /// Get the current stats for `(symbol, feature)`, or default if not yet seen.
    pub fn get(&self, symbol: &str, feature: &str) -> RollingStats {
        self.inner
            .get(symbol)
            .and_then(|m| m.get(feature).cloned())
            .unwrap_or_default()
    }

    /// Compute and update z-score for `(symbol, feature)` in one call.
    pub fn zscore(&self, symbol: &str, feature: &str, x: f64) -> f32 {
        self.update(symbol, feature, x).zscore(x)
    }

    /// Persist the entire store to disk using bincode. Write is atomic via tmp→rename.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        // Snapshot the DashMap into a plain HashMap for serialization
        let snapshot: HashMap<String, HashMap<String, RollingStats>> = self
            .inner
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let bytes = bincode::serialize(&snapshot).context("bincode serialize rolling stats")?;

        // Atomic write: write to .tmp, then rename
        let tmp_path = path.with_extension("tmp");
        if let Some(parent) = tmp_path.parent() {
            std::fs::create_dir_all(parent).context("create rolling_stats directory")?;
        }
        std::fs::write(&tmp_path, &bytes).context("write rolling_stats.tmp")?;
        std::fs::rename(&tmp_path, path).context("rename rolling_stats.tmp → rolling_stats.bin")?;

        tracing::info!(path = %path.display(), symbols = self.inner.len(), "rolling stats saved");
        Ok(())
    }

    /// Load from disk. If the file is absent or corrupt, returns a fresh empty store.
    pub fn load(path: &Path) -> Arc<Self> {
        let store = Self::new();
        if !path.exists() {
            return store;
        }
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(path = %path.display(), err = %e, "failed to read rolling stats, starting fresh");
                return store;
            }
        };
        let snapshot: HashMap<String, HashMap<String, RollingStats>> =
            match bincode::deserialize(&bytes) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(err = %e, "failed to deserialize rolling stats, starting fresh");
                    return store;
                }
            };
        for (symbol, features) in snapshot {
            store.inner.insert(symbol, features);
        }
        tracing::info!(path = %path.display(), symbols = store.inner.len(), "rolling stats loaded");
        store
    }
}
