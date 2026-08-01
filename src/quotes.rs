//! Quote fetch against a reused, already-authenticated `Ticker`. Each snap
//! carries a `fetched_at` stamp — the point-in-time clock downstream.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use finance_query::Ticker;
use finance_query::format::Raw;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteSnap {
    pub symbol: String,
    pub price: f64,
    pub prev_close: Option<f64>,
    pub open: Option<f64>,
    pub day_volume: Option<i64>,
    /// Upstream session state, e.g. "REGULAR", "PRE", "POST", "CLOSED".
    pub market_state: Option<String>,
    /// Unix seconds of the quote's own market timestamp.
    pub market_time: Option<i64>,
    pub fetched_at: DateTime<Utc>,
}

pub async fn fetch_quote(symbol: &str, ticker: &Ticker) -> Result<QuoteSnap> {
    let q = ticker
        .quote::<Raw>()
        .await
        .with_context(|| format!("quote for {symbol}"))?;
    let price = q
        .regular_market_price
        .with_context(|| format!("{symbol}: quote has no price"))?;
    Ok(QuoteSnap {
        symbol: symbol.to_string(),
        price,
        prev_close: q.regular_market_previous_close,
        open: q.regular_market_open,
        day_volume: q.regular_market_volume,
        market_state: q.market_state.clone(),
        market_time: q.regular_market_time,
        fetched_at: Utc::now(),
    })
}
