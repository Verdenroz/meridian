//! Meridian DataBus — Section 6.1 of meridian-spec.md
//!
//! A persistent Rust process that runs on the Raspberry Pi 5 and supplies the
//! Python inference pipeline with all model inputs.  It also handles historical
//! batch collection when invoked from `meridian data collect`.
//!
//! # Responsibilities
//!
//! ## Scheduler (live mode)
//! ```text
//! every  1s:  PriceStream tick → price_buffer (100 tickers)
//! every  1m:  assemble_features() → emit feature vectors to Python bridge
//! every  5m:  poll_rss_feeds()       — 31 sources via feeds::fetch_all()
//! every 15m:  poll_options_chains()  — ticker.options() × 100
//! every  1h:  poll_world_indices()   — finance::indices(Americas|Europe|AsiaPacific)
//! every  1h:  poll_crypto_market()   — crypto::coins() + crypto::coin()
//! every  6h:  poll_insider_tx()      — ticker.insider_transactions()
//! every 24h:  poll_analyst_ratings() — ticker.grading_history()
//! every 24h:  poll_yield_curve()     — fred::treasury_yields()
//! every 24h:  poll_sector_industry() — finance::sector(Sector::*) × 11
//! every 24h:  poll_fear_greed()      — finance::fear_and_greed()
//! every 24h:  poll_market_summary()  — finance::market_summary(Region::US)
//! on_release: poll_fred_macro()      — fred::series() × 12 series
//! quarterly:  poll_fundamentals()    — ticker.financial_data() + key_stats() + ...
//! quarterly:  poll_earnings_tx()     — finance::earnings_transcript()
//! quarterly:  poll_institution_own() — ticker.institution_ownership()
//! ```

use meridian::{args, historical, poller, scheduler, store};

use anyhow::Context;
use args::{Args, Mode};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    install_tracing();

    // Initialize FRED client if API key is available (required for macro series polls)
    match std::env::var("FRED_API_KEY") {
        Ok(key) if !key.is_empty() => {
            if let Err(e) = finance_query::fred::init(key) {
                tracing::warn!(err = %e, "FRED init failed (macro series will be unavailable)");
            } else {
                tracing::info!("FRED client initialized");
            }
        }
        _ => {
            tracing::warn!("FRED_API_KEY not set — macro series (FEDFUNDS, CPI, etc.) will be skipped");
        }
    }

    let args = Args::parse();

    match args.mode {
        Mode::Live(live) => live_mode(live).await,
        Mode::Historical(hist) => historical::batch::run(hist).await,
    }
}

async fn live_mode(args: args::LiveArgs) -> anyhow::Result<()> {
    // Load symbol list
    let symbols = load_symbols(&args.symbols_file)?;
    tracing::info!(count = symbols.len(), "symbols loaded");

    // Load (or initialize) rolling statistics from disk
    let rolling_stats = store::RollingStatsStore::load(&args.stats_path);

    // Build the shared DataStore
    let store = store::DataStore::new(symbols, rolling_stats);

    // Register SIGINT / SIGTERM handler to persist rolling stats on exit
    let store_for_signal = store.clone();
    let stats_path_for_signal = args.stats_path.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received — saving rolling stats");
        if let Err(e) = store_for_signal.rolling_stats.save(&stats_path_for_signal) {
            tracing::error!(err = %e, "failed to save rolling stats on shutdown");
        }
        std::process::exit(0);
    });

    // Warm up: run a first-pass poll of all long-cycle data before the scheduler starts
    tracing::info!("running initial data warm-up");
    tokio::join!(
        log_err(poller::market::poll_world_indices(&store), "world_indices"),
        log_err(poller::market::poll_crypto_market(&store), "crypto"),
        log_err(poller::market::poll_fear_greed(&store), "fear_greed"),
        log_err(poller::macro_fred::poll_yield_curve(&store), "yield_curve"),
        log_err(poller::rss::poll_rss_feeds(&store), "rss"),
    );

    // Spawn all scheduler tasks (they run indefinitely)
    scheduler::spawn_all(store, args.stats_path);

    // Block forever — the process runs until SIGINT/SIGTERM
    std::future::pending::<()>().await;
    Ok(())
}

fn load_symbols(path: &std::path::Path) -> anyhow::Result<Vec<String>> {
    if !path.exists() {
        // Fallback: a small default set for testing
        tracing::warn!(path = %path.display(), "symbols file not found, using defaults");
        return Ok(default_symbols());
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading symbols file {}", path.display()))?;
    let symbols: Vec<String> = content
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .take(100) // cap at 100
        .map(String::from)
        .collect();
    if symbols.is_empty() {
        anyhow::bail!("symbols file {} is empty", path.display());
    }
    Ok(symbols)
}

fn default_symbols() -> Vec<String> {
    [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "UNH",
        "XOM", "LLY", "V", "AVGO", "JNJ", "PG", "MA", "HD", "CVX", "MRK",
        "SPY", "QQQ", "IWM",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn install_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(fmt::layer().with_writer(std::io::stderr)) // logs to stderr; JSONL goes to stdout
        .init();
}

async fn log_err<F: std::future::Future<Output = anyhow::Result<()>>>(f: F, label: &str) {
    if let Err(e) = f.await {
        tracing::warn!(task = %label, err = %e, "warm-up task failed (will retry on schedule)");
    }
}
