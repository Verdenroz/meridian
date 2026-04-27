use crate::events::EventState;
use crate::store::{CachedValue, FundamentalData, OurCalendar};
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
    calendar: &Arc<DashMap<String, CachedValue<OurCalendar>>>,
    news_sentiment: &Arc<DashMap<String, f32>>,
    symbol: &str,
    now: DateTime<Utc>,
) -> [f32; 10] {
    let state = event_state
        .get(symbol)
        .map(|e| e.clone())
        .unwrap_or_default();

    // Earnings proximity: exp(-days_to_next_earnings / 14)
    let earnings_proximity = fundamentals
        .get(symbol)
        .and_then(|f| {
            f.value.days_since_earnings.map(|d| {
                let days_to_next = (90.0 - d).max(0.0);
                (-days_to_next / 14.0).exp() as f32
            })
        })
        .unwrap_or(0.0);

    // Ex-dividend proximity: exp(-days_to_exdiv / 7)
    let exdiv_proximity: f32 = calendar
        .get(symbol)
        .and_then(|c| c.value.ex_div_ts)
        .map(|ts| {
            let days = ((ts - now.timestamp()) as f64 / 86400.0).max(0.0) as f32;
            (-days / 7.0).exp()
        })
        .unwrap_or(0.0);

    // Per-asset news sentiment from llama-swap sentiment model [-1, +1]
    let per_asset_sentiment = news_sentiment.get(symbol).map(|v| *v).unwrap_or(0.0);

    state.to_block6(now, earnings_proximity, exdiv_proximity, per_asset_sentiment)
}
