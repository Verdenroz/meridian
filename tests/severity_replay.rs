//! Golden replay: the production severity rules over the full SPY daily
//! history must reproduce the episode rates the thresholds were calibrated
//! to (see src/config.rs module docs). Guards against the shipped code and
//! the calibration analysis silently drifting apart.
//!
//! Reads the market_data snapshot (gitignored, rebuildable via
//! `uv run python -m market_data fetch`); skips when absent so CI without
//! data still passes.

use meridian::config::Thresholds;
use meridian::severity::{Components, Level, evaluate};

const SPY_SNAPSHOT: &str = "data/market/charts/SPY.json";
const EWMA_LAMBDA: f64 = 0.94;
const VOLUME_LOOKBACK: usize = 20;

struct Day {
    date: String,
    ret_z: f64,
    gap_z: f64,
    vol_ratio: f64,
}

fn load_days() -> Option<Vec<Day>> {
    let raw = std::fs::read_to_string(SPY_SNAPSHOT).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let rows = doc["data"]["rows"].as_array()?;

    let mut days = Vec::new();
    let mut prev_adj: Option<f64> = None;
    let mut prev_close: Option<f64> = None;
    let mut prev_var: Option<f64> = None;
    let mut vols: Vec<f64> = Vec::new();
    for r in rows {
        let adj = r["adj_close"].as_f64()?;
        let close = r["close"].as_f64()?;
        let open = r["open"].as_f64()?;
        let vol = r["volume"].as_f64()?;
        if let (Some(pa), Some(pc)) = (prev_adj, prev_close) {
            let ret = (adj / pa).ln();
            let gap = if open > 0.0 { (open / pc).ln() } else { 0.0 };
            if let Some(v) = prev_var {
                let sigma = v.sqrt();
                let lookback = &vols[vols.len().saturating_sub(VOLUME_LOOKBACK)..];
                let v20 = lookback.iter().sum::<f64>() / lookback.len() as f64;
                if sigma > 0.0 && v20 > 0.0 {
                    days.push(Day {
                        date: r["date"].as_str()?.to_string(),
                        ret_z: ret.abs() / sigma,
                        gap_z: gap.abs() / sigma,
                        vol_ratio: vol / v20,
                    });
                }
            }
            prev_var = Some(match prev_var {
                Some(v) => EWMA_LAMBDA * v + (1.0 - EWMA_LAMBDA) * ret * ret,
                None => ret * ret,
            });
        }
        prev_adj = Some(adj);
        prev_close = Some(close);
        vols.push(vol);
    }
    Some(days)
}

/// Hit days more than 7 calendar days apart count as separate episodes —
/// the same clustering rule the calibration used.
fn episodes(dates: &[&str]) -> usize {
    let mut count = 0;
    let mut last: Option<chrono::NaiveDate> = None;
    for d in dates {
        let dd: chrono::NaiveDate = d.parse().unwrap();
        if last.is_none_or(|l| (dd - l).num_days() > 7) {
            count += 1;
        }
        last = Some(dd);
    }
    count
}

#[test]
fn production_rules_reproduce_calibration_rates() {
    let Some(days) = load_days() else {
        eprintln!("skipping: {SPY_SNAPSHOT} not present (run market_data fetch)");
        return;
    };
    assert!(days.len() > 8000, "unexpectedly short history: {}", days.len());
    let years = days.len() as f64 / 252.0;

    let t = Thresholds::default();
    let mut watch_days: Vec<&str> = Vec::new();
    let mut break_days: Vec<&str> = Vec::new();
    for d in &days {
        let c = Components {
            ret_z: Some(d.ret_z),
            gap_z: Some(d.gap_z),
            vol_ratio: Some(d.vol_ratio),
            rvol_ratio: None,
        };
        match evaluate(&c, &t).0 {
            Level::BreakGlass => break_days.push(&d.date),
            Level::Watch => watch_days.push(&d.date),
            Level::None => {}
        }
    }

    // Watch-or-worse must match the calibrated watch union (~4.1 episodes/yr);
    // break-glass alone the L2 union (~1.9/yr). Bands are generous enough to
    // absorb new data accruing, tight enough to catch a rule regression.
    let mut watch_or_worse: Vec<&str> = watch_days.iter().chain(break_days.iter()).copied().collect();
    watch_or_worse.sort();
    let watch_rate = episodes(&watch_or_worse) as f64 / years;
    let break_rate = episodes(&break_days) as f64 / years;
    assert!(
        (3.4..=4.8).contains(&watch_rate),
        "watch episode rate {watch_rate:.2}/yr outside calibration band"
    );
    assert!(
        (1.5..=2.4).contains(&break_rate),
        "break-glass episode rate {break_rate:.2}/yr outside calibration band"
    );

    // Ground-truth days the rules must classify. Note 2020-03-16 (the -12%
    // day) breaks glass on the *gap* rule — EWMA had already adapted, so its
    // close-to-close z is only 2.7; that morning gap is why the rule exists.
    let by_date: std::collections::HashMap<&str, &Day> =
        days.iter().map(|d| (d.date.as_str(), d)).collect();
    for crash in ["2018-02-05", "2020-03-09", "2020-03-16", "2025-04-04"] {
        let d = by_date.get(crash).expect("crash day missing from history");
        let c = Components {
            ret_z: Some(d.ret_z),
            gap_z: Some(d.gap_z),
            vol_ratio: Some(d.vol_ratio),
            rvol_ratio: None,
        };
        assert_eq!(evaluate(&c, &t).0, Level::BreakGlass, "{crash} must break glass");
    }
    let quiet = by_date.get("2024-07-03").expect("quiet day missing");
    let c = Components {
        ret_z: Some(quiet.ret_z),
        gap_z: Some(quiet.gap_z),
        vol_ratio: Some(quiet.vol_ratio),
        rvol_ratio: None,
    };
    assert_eq!(evaluate(&c, &t).0, Level::None, "2024-07-03 must stay quiet");
}
