use super::{
    FeatureRow, classify_asset, cross_asset, event_block, fundamental_block, market_wide,
    options_block, price_volume, risk_block, technical,
};
use crate::store::DataStore;
use chrono::Utc;
use std::sync::Arc;

/// Assemble feature vectors for every tracked symbol and return them as `FeatureRow`s.
///
/// Called by the 1-minute scheduler task. Produces one `FeatureRow` per symbol.
pub fn assemble_all(store: &Arc<DataStore>) -> Vec<FeatureRow> {
    let now = Utc::now();
    let market = market_wide::assemble_market_vector(store, now);

    store
        .symbols
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

    FeatureRow {
        timestamp: now,
        symbol: symbol.to_string(),
        asset_class: classify_asset(symbol),
        price_volume: price_volume::assemble(price_buf_ref, &store.rolling_stats, symbol),
        technical: technical::assemble(price_buf_ref, &store.rolling_stats, symbol),
        options: options_block::assemble(&store.options, symbol, &store.rolling_stats),
        fundamental: fundamental_block::assemble(&store.fundamentals, symbol, &store.rolling_stats),
        risk_analytics: risk_block::assemble(&store.risk, symbol),
        event_state: event_block::assemble(&store.event_state, &store.fundamentals, symbol, now),
        cross_asset_context: cross_asset::assemble(symbol, store, now),
        text: [0.0f32; 64], // Phase 0 stub — wired to MarketReader in Phase 1
        market: *market,
    }
}
