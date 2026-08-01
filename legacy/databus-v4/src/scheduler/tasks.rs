use crate::features::assembler::assemble_all;
use crate::output::emit_row;
use crate::poller::{
    analyst, insider, macro_fred, market, news_sentiment, options, risk, rss, screener, text_embed,
};
use crate::store::DataStore;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;

/// Every 1 minute: assemble features for all symbols and emit JSONL to stdout.
pub async fn run_one_minute(store: Arc<DataStore>) {
    let mut ticker = interval(Duration::from_secs(60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let rows = assemble_all(&store);
        let count = rows.len();
        for row in rows {
            emit_row(&row);
        }
        tracing::debug!(symbols = count, "emitted minute feature vectors");
    }
}

/// Every 5 minutes: poll RSS feeds, then score per-asset news sentiment.
pub async fn run_five_minute(store: Arc<DataStore>) {
    let mut ticker = interval(Duration::from_secs(5 * 60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        if let Err(e) = rss::poll_rss_feeds(&store).await {
            tracing::warn!(err = %e, "RSS poll failed");
        }
        if let Err(e) = news_sentiment::poll_news_sentiment(&store).await {
            tracing::warn!(err = %e, "news sentiment poll failed");
        }
        if let Err(e) = text_embed::poll_text_embeddings(&store).await {
            tracing::warn!(err = %e, "block8 text embedding poll failed");
        }
    }
}

/// Every 15 minutes: poll options chains.
pub async fn run_fifteen_minute(store: Arc<DataStore>) {
    let mut ticker = interval(Duration::from_secs(15 * 60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        if let Err(e) = options::poll_options_chains(&store).await {
            tracing::warn!(err = %e, "options poll failed");
        }
    }
}

/// Every 1 hour: poll world indices, crypto, market summary, and screener membership.
pub async fn run_hourly(store: Arc<DataStore>) {
    let mut ticker = interval(Duration::from_secs(60 * 60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        tokio::join!(
            log_err(market::poll_world_indices(&store), "world_indices"),
            log_err(market::poll_crypto_market(&store), "crypto"),
            log_err(market::poll_market_summary(&store), "market_summary"),
            log_err(screener::poll_screener_membership(&store), "screener"),
        );
    }
}

/// Every 6 hours: poll insider transactions.
pub async fn run_six_hour(store: Arc<DataStore>) {
    let mut ticker = interval(Duration::from_secs(6 * 60 * 60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        if let Err(e) = insider::poll_insider_tx(&store).await {
            tracing::warn!(err = %e, "insider_tx poll failed");
        }
    }
}

/// Every 24 hours: poll analyst ratings, yield curve, sector data, fear & greed, risk metrics.
pub async fn run_daily(store: Arc<DataStore>) {
    let mut ticker = interval(Duration::from_secs(24 * 60 * 60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        tokio::join!(
            log_err(analyst::poll_analyst_ratings(&store), "analyst"),
            log_err(macro_fred::poll_yield_curve(&store), "yield_curve"),
            log_err(market::poll_sector_industry(&store), "sector"),
            log_err(market::poll_fear_greed(&store), "fear_greed"),
            log_err(risk::poll_risk_analytics(&store), "risk"),
            log_err(macro_fred::poll_fred_macro(&store), "fred_macro"),
        );
    }
}

/// Every 1 hour: persist rolling statistics to disk.
pub async fn run_stats_persist(store: Arc<DataStore>, stats_path: std::path::PathBuf) {
    let mut ticker = interval(Duration::from_secs(60 * 60));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        if let Err(e) = store.rolling_stats.save(&stats_path) {
            tracing::warn!(err = %e, "failed to save rolling stats");
        }
    }
}

async fn log_err<F: std::future::Future<Output = anyhow::Result<()>>>(f: F, label: &str) {
    if let Err(e) = f.await {
        tracing::warn!(task = %label, err = %e, "poller task failed");
    }
}
