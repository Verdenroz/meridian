pub mod stream;
pub mod tasks;
pub mod warmup;
pub mod watch;

use crate::store::DataStore;
use std::path::PathBuf;
use std::sync::Arc;

/// Spawn all scheduler tasks. Returns immediately; tasks run indefinitely in the background.
pub fn spawn_all(store: Arc<DataStore>, stats_path: PathBuf, symbols_file: PathBuf) {
    // Real-time price stream (every 1s via WebSocket; re-subscribes on universe change)
    tokio::spawn(stream::run_price_stream(store.clone()));

    // Live universe control — hot-reload portfolio_symbols.txt
    tokio::spawn(watch::watch_symbols_file(store.clone(), symbols_file));

    // Feature assembly + JSONL emission (every 1m)
    tokio::spawn(tasks::run_one_minute(store.clone()));

    // RSS feeds (every 5m)
    tokio::spawn(tasks::run_five_minute(store.clone()));

    // Options chains (every 15m)
    tokio::spawn(tasks::run_fifteen_minute(store.clone()));

    // World indices + crypto + market summary (every 1h)
    tokio::spawn(tasks::run_hourly(store.clone()));

    // Insider transactions (every 6h)
    tokio::spawn(tasks::run_six_hour(store.clone()));

    // Analysts + FRED + sector + fear/greed + risk (every 24h)
    tokio::spawn(tasks::run_daily(store.clone()));

    // Rolling stats persistence (every 1h)
    tokio::spawn(tasks::run_stats_persist(store, stats_path));
}
