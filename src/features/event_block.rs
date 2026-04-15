use crate::events::EventState;
use crate::store::{CachedValue, FundamentalData};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::Arc;

/// Block 6: Event State — 10 decayed event signals.
///
/// Each feature represents a time-decayed signal from the most recent event
/// of that type for the asset. See `EventState::to_block6` for the decay logic.
pub fn assemble(
    event_state: &Arc<DashMap<String, EventState>>,
    fundamentals: &Arc<DashMap<String, CachedValue<FundamentalData>>>,
    symbol: &str,
    now: DateTime<Utc>,
) -> [f32; 10] {
    let state = event_state
        .get(symbol)
        .map(|e| e.clone())
        .unwrap_or_default();

    // Earnings proximity: exp(-days_to_earnings / 14)
    let earnings_proximity = fundamentals
        .get(symbol)
        .and_then(|f| {
            // Use days_since_earnings as a proxy for next earnings distance
            f.value.days_since_earnings.map(|d| {
                let days_to_next = (90.0 - d).max(0.0); // rough quarterly cycle
                (-days_to_next / 14.0).exp() as f32
            })
        })
        .unwrap_or(0.0);

    // Ex-dividend proximity: stub (would come from calendar_events)
    let exdiv_proximity: f32 = 0.0;

    state.to_block6(now, earnings_proximity, exdiv_proximity)
}
