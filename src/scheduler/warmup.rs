//! Warm up a newly added symbol before it emits feature rows.
//!
//! When the universe gains a symbol at runtime it has no price history, so its
//! returns/indicators would be degenerate. We backfill recent minute bars from
//! finance-query into the price buffer, then mark the symbol ready so the
//! assembler begins emitting it. Rolling-stat z-score baselines continue to warm
//! up live from there (same as the initial universe's first run).

use crate::store::{Bar, DataStore, PriceBuffer};
use finance_query::{Interval, Ticker};
use std::sync::Arc;

/// Trailing minute-bar history to seed. Yahoo serves ~29 days of 1m bars; the
/// price buffer holds 98,280 minutes (~252 sessions).
const WARMUP_DAYS: i64 = 29;
const PRICE_BUFFER_CAP: usize = 98_280;

/// Backfill price history for `symbol`, then mark it ready. On failure the symbol
/// stays in the universe but un-ready (not emitted); the next `portfolio_symbols.txt` edit
/// re-triggers warm-up.
pub async fn warm_up_symbol(store: &Arc<DataStore>, symbol: &str) {
    match seed_price_history(store, symbol).await {
        Ok(bars) => {
            store.mark_ready(symbol);
            store.bump_universe(); // it now emits — refresh stream + downstream
            tracing::info!(symbol = %symbol, bars, "universe: warm-up complete, now live");
        }
        Err(e) => {
            tracing::warn!(
                symbol = %symbol, err = %e,
                "universe: warm-up failed; symbol stays un-ready until next edit"
            );
        }
    }
}

async fn seed_price_history(store: &Arc<DataStore>, symbol: &str) -> anyhow::Result<usize> {
    let ticker = Ticker::new(symbol).await?;
    let now = chrono::Utc::now().timestamp();
    let start = now - WARMUP_DAYS * 86_400;
    let chart = ticker.chart_range(Interval::OneMinute, start, now).await?;

    let mut buf = PriceBuffer::new(PRICE_BUFFER_CAP);
    for c in &chart.candles {
        buf.push(Bar {
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
            volume: c.volume,
            timestamp: c.timestamp,
        });
    }
    let n = chart.candles.len();
    store.prices.insert(symbol.to_string(), buf);
    Ok(n)
}
