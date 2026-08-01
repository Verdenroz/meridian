use super::{EventKind, EventRecord};
use crate::store::{OurGradeChange, OurInsiderTx, OurSecFiling};
use chrono::Utc;
use finance_query::Split;
use std::collections::HashSet;

// ── Insider transactions ──────────────────────────────────────────────────────

pub fn detect_insider_events(
    symbol: &str,
    old: &[OurInsiderTx],
    new: &[OurInsiderTx],
) -> Vec<EventRecord> {
    let old_dates: HashSet<String> = old.iter().map(filing_date_key).collect();

    new.iter()
        .filter(|t| !old_dates.contains(&filing_date_key(t)))
        .map(|t| {
            let is_purchase = t
                .transaction_text
                .as_deref()
                .map(|s| s.to_lowercase().contains("purchase") || s.to_lowercase().contains("buy"))
                .unwrap_or(false);

            let kind = if is_purchase {
                EventKind::InsiderBuy
            } else {
                EventKind::InsiderSell
            };

            let magnitude = t
                .value
                .map(|v| (v.unsigned_abs() as f64).ln().max(0.0) as f32)
                .unwrap_or(1.0);

            EventRecord {
                kind,
                symbol: Some(symbol.to_string()),
                magnitude,
                detected_at: Utc::now(),
                description: format!(
                    "{} {} shares",
                    t.filer_name.as_deref().unwrap_or("unknown"),
                    t.shares.unwrap_or(0)
                ),
            }
        })
        .collect()
}

fn filing_date_key(t: &OurInsiderTx) -> String {
    t.start_date_ts.map(|ts| ts.to_string()).unwrap_or_default()
}

// ── Analyst ratings ───────────────────────────────────────────────────────────

pub fn detect_analyst_events(
    symbol: &str,
    old: &[OurGradeChange],
    new: &[OurGradeChange],
) -> Vec<EventRecord> {
    let old_set: HashSet<String> = old.iter().map(grade_key).collect();

    new.iter()
        .filter(|g| !old_set.contains(&grade_key(g)))
        .filter_map(|g| {
            let action = g.action.as_deref()?.to_lowercase();
            let kind = if action == "up" || action.contains("upgrade") || action.contains("init") {
                EventKind::AnalystUpgrade
            } else if action == "down" || action.contains("downgrade") {
                EventKind::AnalystDowngrade
            } else {
                return None;
            };

            Some(EventRecord {
                kind,
                symbol: Some(symbol.to_string()),
                magnitude: 1.0,
                detected_at: Utc::now(),
                description: format!(
                    "{} → {}",
                    g.from_grade.as_deref().unwrap_or("?"),
                    g.to_grade.as_deref().unwrap_or("?")
                ),
            })
        })
        .collect()
}

fn grade_key(g: &OurGradeChange) -> String {
    format!(
        "{}{}{}",
        g.date.as_deref().unwrap_or(""),
        g.firm.as_deref().unwrap_or(""),
        g.to_grade.as_deref().unwrap_or("")
    )
}

// ── SEC filings ───────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub fn detect_sec_filing_events(
    symbol: &str,
    old: &[OurSecFiling],
    new: &[OurSecFiling],
) -> Vec<EventRecord> {
    let old_set: HashSet<String> = old.iter().map(filing_key).collect();

    new.iter()
        .filter(|f| !old_set.contains(&filing_key(f)))
        .map(|f| {
            let form = f.filing_type.as_deref().unwrap_or("");
            let (kind, magnitude) = if form == "10-K" {
                (EventKind::SecFiling10K, 1.0f32)
            } else if form == "10-Q" {
                (EventKind::SecFiling10Q, 1.0)
            } else if form == "8-K" {
                (EventKind::SecFiling8K, 1.5)
            } else {
                (EventKind::SecFilingOther, 0.5)
            };

            EventRecord {
                kind,
                symbol: Some(symbol.to_string()),
                magnitude,
                detected_at: Utc::now(),
                description: format!("{form} filing"),
            }
        })
        .collect()
}

#[allow(dead_code)]
fn filing_key(f: &OurSecFiling) -> String {
    format!(
        "{}{}",
        f.filing_type.as_deref().unwrap_or(""),
        f.date.as_deref().unwrap_or("")
    )
}

// ── Stock splits ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub fn detect_split_events(symbol: &str, old: &[Split], new: &[Split]) -> Vec<EventRecord> {
    let old_ts: HashSet<i64> = old.iter().map(|s| s.timestamp).collect();

    new.iter()
        .filter(|s| !old_ts.contains(&s.timestamp))
        .map(|s| {
            let ratio = if s.denominator != 0.0 {
                s.numerator / s.denominator
            } else {
                1.0
            };
            EventRecord {
                kind: EventKind::SplitAnnounced,
                symbol: Some(symbol.to_string()),
                magnitude: ratio as f32,
                detected_at: Utc::now(),
                description: format!("Split {}", s.ratio),
            }
        })
        .collect()
}

// ── Trending ticker ───────────────────────────────────────────────────────────

pub fn detect_trending_events(
    old_set: &HashSet<String>,
    new_set: &HashSet<String>,
) -> Vec<EventRecord> {
    new_set
        .difference(old_set)
        .map(|sym| EventRecord {
            kind: EventKind::TrendingEntry,
            symbol: Some(sym.clone()),
            magnitude: 1.0,
            detected_at: Utc::now(),
            description: "Entered trending list".to_string(),
        })
        .collect()
}

// ── Analyst rating consensus shift ───────────────────────────────────────────

pub fn detect_rating_change(symbol: &str, old_score: f64, new_score: f64) -> Option<EventRecord> {
    let delta = new_score - old_score;
    if delta.abs() < 0.1 {
        return None;
    }
    Some(EventRecord {
        kind: EventKind::RatingChange,
        symbol: Some(symbol.to_string()),
        magnitude: delta as f32,
        detected_at: Utc::now(),
        description: format!("Consensus Δ = {delta:.2}"),
    })
}

// ── Options unusual activity ──────────────────────────────────────────────────

pub fn detect_options_unusual(
    symbol: &str,
    total_vol: f64,
    total_oi: f64,
    vol_mean: f64,
    vol_std: f64,
) -> Option<EventRecord> {
    if vol_std < 1.0 {
        return None;
    }
    let zscore = (total_vol - vol_mean) / vol_std;
    let vol_oi_ratio = if total_oi > 0.0 {
        total_vol / total_oi
    } else {
        0.0
    };

    if zscore > 2.0 && vol_oi_ratio > 0.1 {
        Some(EventRecord {
            kind: EventKind::OptionsUnusualActivity,
            symbol: Some(symbol.to_string()),
            magnitude: zscore as f32,
            detected_at: Utc::now(),
            description: format!(
                "Vol {total_vol:.0} ({zscore:.1}σ above avg), V/OI={vol_oi_ratio:.2}"
            ),
        })
    } else {
        None
    }
}

// ── FOMC detection (from RSS) ─────────────────────────────────────────────────

fn is_fomc_title(title: &str) -> bool {
    let lower = title.to_lowercase();
    lower.contains("federal funds")
        || lower.contains("rate decision")
        || lower.contains("fomc")
        || lower.contains("fed raises")
        || lower.contains("fed cuts")
        || lower.contains("basis point")
}

pub fn detect_fomc_from_articles(
    new_articles: &[super::super::store::RssArticle],
) -> Vec<EventRecord> {
    new_articles
        .iter()
        .filter(|a| is_fomc_title(&a.title))
        .map(|a| EventRecord {
            kind: EventKind::FomcDecision,
            symbol: None,
            magnitude: 0.0,
            detected_at: Utc::now(),
            description: a.title.clone(),
        })
        .collect()
}

// ── FRED macro release ────────────────────────────────────────────────────────

pub fn detect_macro_release(
    series_id: &str,
    old_date: &str,
    new_date: &str,
    new_value: f64,
    old_value: f64,
    hist_std: f64,
) -> Option<EventRecord> {
    if old_date == new_date || new_date.is_empty() {
        return None;
    }
    let magnitude = if hist_std > 1e-10 {
        ((new_value - old_value).abs() / hist_std) as f32
    } else {
        0.0
    };
    Some(EventRecord {
        kind: EventKind::MacroRelease,
        symbol: None,
        magnitude,
        detected_at: Utc::now(),
        description: format!("{series_id}: {old_date} → {new_date} (value={new_value:.4})"),
    })
}
