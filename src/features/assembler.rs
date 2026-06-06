use super::{
    FeatureRow, classify_asset, cross_asset, event_block, fundamental_block, market_wide,
    options_block, price_volume, risk_block, technical,
};
use crate::{args::HistoricalInterval, store::DataStore};
use chrono::Utc;
use std::sync::Arc;

/// Assemble feature vectors for every tracked symbol and return them as `FeatureRow`s.
///
/// Called by the 1-minute scheduler task. Produces one `FeatureRow` per symbol.
pub fn assemble_all(store: &Arc<DataStore>) -> Vec<FeatureRow> {
    let now = Utc::now();
    let market = market_wide::assemble_market_vector(store, now);

    // Emit only warmed-up symbols: the live active universe == ready set, so a
    // freshly added symbol never contributes a cold-start (garbage z-score) row.
    store
        .ready_snapshot()
        .iter()
        .map(|sym| assemble_one(sym, store, now, &market))
        .collect()
}

fn assemble_one(
    symbol: &str,
    store: &Arc<DataStore>,
    now: chrono::DateTime<Utc>,
    market: &[f32; 96],
) -> FeatureRow {
    let price_buf = store.prices.get(symbol);
    let price_buf_ref = price_buf.as_deref();

    // Live system operates at 1-minute interval
    let interval = HistoricalInterval::OneMinute;

    // Earnings proximity is shared between event_block and options_block
    let earnings_proximity = store
        .fundamentals
        .get(symbol)
        .and_then(|f| {
            f.value.days_since_earnings.map(|d| {
                let days_to_next = (90.0 - d).max(0.0);
                (-days_to_next / 14.0).exp() as f32
            })
        })
        .unwrap_or(0.0);

    FeatureRow {
        timestamp: now,
        symbol: symbol.to_string(),
        asset_class: classify_asset(symbol),
        price_volume: price_volume::assemble(price_buf_ref, &store.rolling_stats, symbol, &interval),
        technical: technical::assemble(price_buf_ref, &store.rolling_stats, symbol, &interval),
        options: options_block::assemble(&store.options, &store.prices, symbol, &store.rolling_stats, earnings_proximity),
        fundamental: fundamental_block::assemble(&store.fundamentals, symbol, &store.rolling_stats),
        risk_analytics: risk_block::assemble(&store.risk, symbol),
        event_state: event_block::assemble(&store.event_state, &store.fundamentals, &store.calendar, &store.news_sentiment, symbol, now),
        cross_asset_context: cross_asset::assemble(symbol, store, now),
        // Block 8: harrier-oss → PCA(64), produced by the `text_embed` poller
        // (same embedder + basis as the historical collector). Absent ⇒ honest
        // zeros (no in-window headlines yet, basis uncached, or embedder down).
        text: store
            .text_embedding
            .get(symbol)
            .map(|v| *v)
            .unwrap_or([0.0f32; 64]),
        market: *market,
    }
}
