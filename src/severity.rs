//! Break-glass severity evaluation (§4.5) — pure functions over the
//! per-cycle components, no I/O. Price/volume rules only: fully functional
//! without the LLM. Event-pipeline severity scores join as a component only
//! once live filing scoring is plumbed; today they are context, not triggers.

use crate::config::Thresholds;
use serde::{Deserialize, Serialize};

/// Monitor components for one symbol, normalized against its daily baseline.
/// `None` means the input was unavailable this cycle — a missing component
/// never satisfies a rule (blind inputs must not fire or suppress alarms).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Components {
    /// |log(price / prev_close)| / EWMA(0.94) daily vol.
    pub ret_z: Option<f64>,
    /// |log(open / prev_close)| / EWMA(0.94) daily vol.
    pub gap_z: Option<f64>,
    /// Projected full-day volume / trailing 20-day mean volume.
    pub vol_ratio: Option<f64>,
    /// Intraday realized vol (from 5m bars, day-scaled) / EWMA forecast.
    /// Diagnostic only in v0 — no historical intraday record to calibrate on.
    pub rvol_ratio: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    None,
    Watch,
    BreakGlass,
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Level::None => "none",
            Level::Watch => "watch",
            Level::BreakGlass => "break_glass",
        })
    }
}

/// Evaluate the calibrated trigger rules. Returns the level plus the list of
/// rules that fired (for the alert record and the human reading it).
pub fn evaluate(c: &Components, t: &Thresholds) -> (Level, Vec<String>) {
    let ge = |v: Option<f64>, th: f64| v.is_some_and(|x| x >= th);

    let mut drivers = Vec::new();
    if ge(c.ret_z, t.break_ret_z) {
        drivers.push(format!("ret_z {:.2} >= {}", c.ret_z.unwrap(), t.break_ret_z));
    }
    if ge(c.gap_z, t.break_gap_z) {
        drivers.push(format!("gap_z {:.2} >= {}", c.gap_z.unwrap(), t.break_gap_z));
    }
    if ge(c.ret_z, t.break_combo_ret_z) && ge(c.vol_ratio, t.break_combo_vol_ratio) {
        drivers.push(format!(
            "ret_z {:.2} >= {} & vol_ratio {:.2} >= {}",
            c.ret_z.unwrap(),
            t.break_combo_ret_z,
            c.vol_ratio.unwrap(),
            t.break_combo_vol_ratio
        ));
    }
    if !drivers.is_empty() {
        return (Level::BreakGlass, drivers);
    }

    if ge(c.ret_z, t.watch_ret_z) {
        drivers.push(format!("ret_z {:.2} >= {}", c.ret_z.unwrap(), t.watch_ret_z));
    }
    if ge(c.gap_z, t.watch_gap_z) {
        drivers.push(format!("gap_z {:.2} >= {}", c.gap_z.unwrap(), t.watch_gap_z));
    }
    if ge(c.vol_ratio, t.watch_vol_ratio) {
        drivers.push(format!(
            "vol_ratio {:.2} >= {}",
            c.vol_ratio.unwrap(),
            t.watch_vol_ratio
        ));
    }
    if !drivers.is_empty() {
        return (Level::Watch, drivers);
    }
    (Level::None, drivers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Thresholds {
        Thresholds::default()
    }

    #[test]
    fn quiet_day_is_none() {
        let c = Components {
            ret_z: Some(0.8),
            gap_z: Some(0.2),
            vol_ratio: Some(1.1),
            rvol_ratio: Some(1.0),
        };
        assert_eq!(evaluate(&c, &t()).0, Level::None);
    }

    #[test]
    fn missing_components_never_fire() {
        let c = Components::default();
        let (level, drivers) = evaluate(&c, &t());
        assert_eq!(level, Level::None);
        assert!(drivers.is_empty());
    }

    #[test]
    fn ret_z_watch_then_break() {
        let mut c = Components {
            ret_z: Some(3.1),
            ..Default::default()
        };
        assert_eq!(evaluate(&c, &t()).0, Level::Watch);
        c.ret_z = Some(3.6);
        assert_eq!(evaluate(&c, &t()).0, Level::BreakGlass);
    }

    #[test]
    fn combo_rule_requires_both() {
        let c = Components {
            ret_z: Some(3.1),
            vol_ratio: Some(2.6),
            ..Default::default()
        };
        let (level, drivers) = evaluate(&c, &t());
        assert_eq!(level, Level::BreakGlass);
        assert_eq!(drivers.len(), 1);

        let c = Components {
            ret_z: Some(3.1),
            vol_ratio: Some(2.0),
            ..Default::default()
        };
        assert_eq!(evaluate(&c, &t()).0, Level::Watch);
    }

    #[test]
    fn gap_alone_can_break() {
        let c = Components {
            gap_z: Some(2.7),
            ..Default::default()
        };
        assert_eq!(evaluate(&c, &t()).0, Level::BreakGlass);
    }

    #[test]
    fn rvol_is_diagnostic_only() {
        let c = Components {
            rvol_ratio: Some(9.0),
            ..Default::default()
        };
        assert_eq!(evaluate(&c, &t()).0, Level::None);
    }

    #[test]
    fn boundary_is_inclusive() {
        let c = Components {
            ret_z: Some(3.5),
            ..Default::default()
        };
        assert_eq!(evaluate(&c, &t()).0, Level::BreakGlass);
    }

    #[test]
    fn level_ordering() {
        assert!(Level::BreakGlass > Level::Watch);
        assert!(Level::Watch > Level::None);
    }
}
