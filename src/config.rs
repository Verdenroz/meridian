//! Bus configuration: monitored universe, cadences, output paths, and the
//! break-glass thresholds with their calibration provenance.
//!
//! Threshold calibration (2026-08-01, SPY daily history 1993–2026, EOD): each
//! day's |log return| and |open gap| normalized by the prior day's EWMA(0.94)
//! vol, volume by its trailing 20-day mean. Episode counts (hits >7 calendar
//! days apart) over 33.5 years:
//!   watch union  (ret_z≥3 | gap_z≥2 | vol_ratio≥3)                ≈ 4.1/yr
//!   break-glass  (ret_z≥3.5 | gap_z≥2.5 | ret_z≥3 & vol_ratio≥2.5) ≈ 1.9/yr
//! matching the §4.5 base rate ("a few per year"). Intraday evaluation fires
//! somewhat more often than this EOD proxy; if it fires weekly, the threshold
//! is wrong (§1.6) and recalibration — not silence — is the fix.

use std::path::PathBuf;
use std::time::Duration;

/// Break-glass rule thresholds. Members are named after the rule they feed;
/// see the module docs for the calibrated base rates behind these numbers.
#[derive(Debug, Clone)]
pub struct Thresholds {
    pub watch_ret_z: f64,
    pub watch_gap_z: f64,
    pub watch_vol_ratio: f64,
    pub break_ret_z: f64,
    pub break_gap_z: f64,
    pub break_combo_ret_z: f64,
    pub break_combo_vol_ratio: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            watch_ret_z: 3.0,
            watch_gap_z: 2.0,
            watch_vol_ratio: 3.0,
            break_ret_z: 3.5,
            break_gap_z: 2.5,
            break_combo_ret_z: 3.0,
            break_combo_vol_ratio: 2.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BusConfig {
    /// Monitored symbols. The first is the market-level trigger symbol (SPY);
    /// the rest are the allocation legs, reported but not triggering.
    pub symbols: Vec<String>,
    pub thresholds: Thresholds,
    /// Active-cycle cadence during the regular session.
    pub cycle: Duration,
    /// Heartbeat cadence while the market is closed.
    pub idle_cycle: Duration,
    /// SEC 8-K feed poll cadence (event context, not a trigger).
    pub feed_cycle: Duration,
    /// Per-cycle monitor journal, one file per UTC day (gitignored).
    pub monitor_dir: PathBuf,
    /// Append-only break-glass alert record (git-tracked, point-in-time).
    pub alerts_path: PathBuf,
    /// Latest snapshot for Python consumers (derived, rebuildable).
    pub latest_path: PathBuf,
    /// finance-query API base for the feeds poll (quotes/bars go through the
    /// crate; feeds go through the API because sec.gov requires a proper UA).
    pub feed_base_url: String,
}

impl Default for BusConfig {
    fn default() -> Self {
        Self {
            symbols: ["SPY", "QQQ", "TLT", "GLD"].map(String::from).to_vec(),
            thresholds: Thresholds::default(),
            cycle: Duration::from_secs(60),
            idle_cycle: Duration::from_secs(300),
            feed_cycle: Duration::from_secs(600),
            monitor_dir: PathBuf::from("logs/bus"),
            alerts_path: PathBuf::from("data/breakglass/alerts.jsonl"),
            latest_path: PathBuf::from("data/bus/latest.json"),
            feed_base_url: "https://finance-query.com".to_string(),
        }
    }
}

impl BusConfig {
    pub fn market_symbol(&self) -> &str {
        &self.symbols[0]
    }
}
