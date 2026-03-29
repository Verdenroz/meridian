pub mod detector;
pub mod state;

pub use state::EventState;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// The 18 discrete financial event categories tracked by the DataBus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum EventKind {
    InsiderBuy,
    InsiderSell,
    AnalystUpgrade,
    AnalystDowngrade,
    EarningsBeat,
    EarningsMiss,
    SecFiling10K,
    SecFiling10Q,
    SecFiling8K,
    SecFilingOther,
    FomcDecision,
    MacroRelease,
    NewsHighImpact,
    DividendAnnounced,
    SplitAnnounced,
    OptionsUnusualActivity,
    TrendingEntry,
    RatingChange,
}

impl EventKind {
    /// Index into the `EventState.slots` array (0–17).
    pub fn index(self) -> usize {
        self as usize
    }

    /// Decay time constant τ in minutes for Block 6 exponential decay.
    pub fn decay_tau_minutes(self) -> f64 {
        match self {
            Self::InsiderBuy | Self::InsiderSell => 1440.0,        // 1 day
            Self::AnalystUpgrade | Self::AnalystDowngrade => 43200.0, // 30 days
            Self::EarningsBeat | Self::EarningsMiss => 1440.0,
            Self::SecFiling10K | Self::SecFiling10Q => 43200.0,
            Self::SecFiling8K => 10080.0,                          // 7 days
            Self::SecFilingOther => 43200.0,
            Self::FomcDecision => 43200.0,
            Self::MacroRelease => 10080.0,
            Self::NewsHighImpact => 60.0,                          // 1 hour
            Self::DividendAnnounced => 10080.0,
            Self::SplitAnnounced => 10080.0,
            Self::OptionsUnusualActivity => 15.0,                  // 15 minutes
            Self::TrendingEntry => 60.0,
            Self::RatingChange => 43200.0,
        }
    }
}

/// A detected discrete financial event.
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    pub kind: EventKind,
    /// Symbol this event pertains to, or `None` for market-wide events (FOMC, MACRO_RELEASE).
    pub symbol: Option<String>,
    /// Normalized magnitude — interpretation varies by event type (see detector docs).
    pub magnitude: f32,
    pub detected_at: DateTime<Utc>,
    pub description: String,
}
