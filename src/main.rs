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
//! every 15m:  poll_options_chains()  — ticker.options(date) × 100
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
//!
//! ## Event Detector (continuous)
//! Diffs each polled snapshot against the previous one to detect discrete events:
//! ```text
//! RSS diff              → NEWS_HIGH_IMPACT event
//! insider_tx diff       → INSIDER_BUY / INSIDER_SELL event
//! grading diff          → ANALYST_UPGRADE / ANALYST_DOWNGRADE event
//! sec_filings diff      → SEC_FILING_10K / 10Q / 8K / OTHER event
//! earnings actual/est   → EARNINGS_BEAT / EARNINGS_MISS event
//! FRED release          → MACRO_RELEASE / FOMC_DECISION event
//! options anomaly       → OPTIONS_UNUSUAL_ACTIVITY event
//! trending diff         → TRENDING_ENTRY event
//! recommendation diff   → RATING_CHANGE event
//! calendar diff         → DIVIDEND_ANNOUNCED / SPLIT_ANNOUNCED event
//! ```
//!
//! ## Feature Assembly (per-minute)
//! Produces two vectors every minute:
//! - Per-asset: 224 continuous + 64 text = 288 dims × 100 assets
//! - Market-wide: 96 dims
//!
//! ## Historical Mode (batch)
//! When invoked with `--symbol <SYM> --start <DATE> --end <DATE>`, collects
//! historical data and emits JSONL rows to stdout matching the v3.0 schema:
//! ```text
//! cargo run -- --symbol AAPL --start 2020-01-01 --end 2025-06-30
//! ```
//!
//! # Feature Blocks (per-asset, 224 continuous dims)
//!
//! | Block                 | Dims | Source                                    |
//! |-----------------------|------|-------------------------------------------|
//! | price_volume          |  10  | PriceStream / chart()                     |
//! | technical             |  42  | indicators::* (full suite)                |
//! | options               |  12  | ticker.options(date)   every 15m          |
//! | fundamental           |  30  | financial_data/key_stats/earnings/analyst |
//! | risk_analytics        |   8  | ticker.risk()          daily              |
//! | event_state           |  10  | decayed signals from event detector       |
//! | cross_asset_context   |   8  | relative strength / sector / trending     |
//! | text (MarketReader)   |  64  | feeds + ticker.news() + transcripts       |
//!
//! # Market-Wide Feature Blocks (96 dims)
//!
//! | Block                    | Dims | Source                                 |
//! |--------------------------|------|----------------------------------------|
//! | volatility_sentiment     |  12  | VIX, Fear&Greed, HY spread             |
//! | yield_curve              |  12  | fred::treasury_yields() — all 12 mats  |
//! | fred_macro               |  12  | fred::series() × 12 FRED series        |
//! | sector_industry          |  16  | finance::sector(Sector::*) × 11        |
//! | world_indices_cross_asset|  20  | finance::indices(*) + crypto + FX      |
//! | breadth_flow             |  12  | market_summary + screener + hours      |
//! | rss_news_aggregate       |  12  | feeds::fetch_all() → MarketReader      |
//!
//! # Rolling Statistics
//! All z-scored features maintain per-asset 252-day rolling mean and variance.
//! Normalization state is persisted to disk and restored on startup.
//!
//! # Staleness & Graceful Degradation
//! Every cached value carries a `last_updated` timestamp. If an API is
//! temporarily unavailable, the DataBus serves the last known value so the
//! model always receives a complete feature vector.

// TODO: implement DataBus (Section 6.1 of meridian-spec.md)

fn main() {
    todo!("DataBus — see module-level doc and Section 6.1 of meridian-spec.md")
}
