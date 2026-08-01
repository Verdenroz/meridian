//! Append-only JSONL journals plus the derived `latest.json` snapshot.
//! Monitor journals are heartbeats in the archive sense: every cycle writes a
//! row, so a coverage gap is visible as missing time, never as silence.

use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::io::Write as _;
use std::path::Path;

pub fn append_jsonl<T: Serialize>(path: &Path, row: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(row)?;
    writeln!(f, "{line}").with_context(|| format!("appending to {}", path.display()))
}

/// Overwrite via temp-file rename so readers never see a torn snapshot.
pub fn write_latest<T: Serialize>(path: &Path, snapshot: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(snapshot)?)?;
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(path)?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).with_context(|| format!("bad row in {}", path.display())))
        .collect()
}

/// Day-partitioned monitor journal path, e.g. `logs/bus/monitor-2026-08-01.jsonl`.
pub fn monitor_path(dir: &Path, date: NaiveDate) -> std::path::PathBuf {
    dir.join(format!("monitor-{date}.jsonl"))
}
