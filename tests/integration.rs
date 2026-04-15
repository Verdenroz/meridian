//! End-to-end integration tests for all finance-query API integrations.
//!
//! All tests hit live network endpoints and are marked `#[ignore]` so they
//! don't run in CI. Run manually with:
//!
//!   cargo test --test integration -- --ignored --nocapture
//!
//! Requires:
//!   FRED_API_KEY set in environment (or .env sourced)

use std::sync::Arc;

use meridian::{
    features::assembler,
    poller::{analyst, macro_fred, market, options, risk, rss},
    store::{DataStore, RollingStatsStore},
};

const TEST_SYMBOL: &str = "AAPL";
const TEST_SYMBOLS: &[&str] = &["AAPL", "MSFT", "SPY"];

fn make_store(symbols: &[&str]) -> Arc<DataStore> {
    let syms: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    let stats = RollingStatsStore::new(); // returns Arc<RollingStatsStore>
    DataStore::new(syms, stats)
}

// ── Per-ticker endpoints ──────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_ticker_quote() {
    let ticker = finance_query::Ticker::new(TEST_SYMBOL).await.unwrap();
    let quote = ticker.quote().await.unwrap();
    assert_eq!(quote.symbol, TEST_SYMBOL);
    let price = quote
        .regular_market_price
        .as_ref()
        .and_then(|fv| fv.raw)
        .expect("regular_market_price should be present");
    assert!(price > 0.0, "price should be positive, got {price}");
    println!("{TEST_SYMBOL} price: {price:.2}");
}

#[tokio::test]
#[ignore]
async fn test_ticker_indicators() {
    use finance_query::{Interval, TimeRange};
    let ticker = finance_query::Ticker::new(TEST_SYMBOL).await.unwrap();
    let summary = ticker
        .indicators(Interval::OneDay, TimeRange::OneYear)
        .await
        .unwrap();
    // Check that at least some indicators were computed
    assert!(
        summary.sma_10.is_some() || summary.ema_20.is_some() || summary.rsi_14.is_some(),
        "no indicators returned"
    );
    println!(
        "{TEST_SYMBOL} indicators: sma_10={:?} ema_20={:?} rsi_14={:?}",
        summary.sma_10, summary.ema_20, summary.rsi_14
    );
}

#[tokio::test]
#[ignore]
async fn test_ticker_options() {
    let ticker = finance_query::Ticker::new(TEST_SYMBOL).await.unwrap();
    let opts = ticker.options(None).await.unwrap();
    let calls = opts.calls();
    let puts = opts.puts();
    assert!(
        !calls.is_empty() || !puts.is_empty(),
        "no options contracts returned"
    );
    println!(
        "{TEST_SYMBOL} options: {} calls, {} puts",
        calls.len(),
        puts.len()
    );
}

#[tokio::test]
#[ignore]
async fn test_ticker_risk() {
    use finance_query::{Interval, TimeRange};
    let ticker = finance_query::Ticker::new(TEST_SYMBOL).await.unwrap();
    let risk = ticker
        .risk(Interval::OneDay, TimeRange::OneYear, Some("SPY"))
        .await
        .unwrap();
    assert!(risk.max_drawdown >= 0.0 && risk.max_drawdown <= 1.0);
    assert!(risk.var_95 >= 0.0);
    println!(
        "{TEST_SYMBOL} risk: sharpe={:?} beta={:?} maxDD={:.1}%",
        risk.sharpe.map(|s| format!("{s:.2}")),
        risk.beta.map(|b| format!("{b:.2}")),
        risk.max_drawdown * 100.0
    );
}

#[tokio::test]
#[ignore]
async fn test_ticker_news() {
    let ticker = finance_query::Ticker::new(TEST_SYMBOL).await.unwrap();
    let news = ticker.news().await.unwrap();
    assert!(!news.is_empty(), "no news articles returned");
    println!(
        "{TEST_SYMBOL} news: {} articles, latest: {:?}",
        news.len(),
        news[0].title
    );
}

#[tokio::test]
#[ignore]
async fn test_ticker_chart_range() {
    use chrono::{Duration, Utc};
    use finance_query::{Interval, Ticker};
    let ticker = Ticker::new(TEST_SYMBOL).await.unwrap();
    let end = Utc::now().timestamp();
    let start = (Utc::now() - Duration::days(5)).timestamp();
    let chart = ticker
        .chart_range(Interval::OneHour, start, end)
        .await
        .unwrap();
    assert!(
        !chart.candles.is_empty(),
        "no candles returned for 5d/1h chart"
    );
    println!("{TEST_SYMBOL} chart: {} 1h candles", chart.candles.len());
}

// ── Market-wide pollers ───────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_poller_market_world_indices() {
    let store = make_store(TEST_SYMBOLS);
    market::poll_world_indices(&store).await.unwrap();
    let wi = store.world_indices.read();
    // At least some index values should be non-zero during market hours
    println!(
        "World indices spot-check: dxy={:.2} gold={:.2}",
        wi.value.dxy, wi.value.gold
    );
}

#[tokio::test]
#[ignore]
async fn test_poller_market_fear_greed() {
    let store = make_store(TEST_SYMBOLS);
    market::poll_fear_greed(&store).await.unwrap();
    let fg = store.fear_greed.read();
    assert!(fg.value.value >= 0.0 && fg.value.value <= 100.0);
    println!(
        "Fear & Greed: {:.1} ({})",
        fg.value.value, fg.value.classification
    );
}

#[tokio::test]
#[ignore]
async fn test_poller_market_sector() {
    let store = make_store(TEST_SYMBOLS);
    market::poll_sector_industry(&store).await.unwrap();
    let sector = store.sector.read();
    // 11 GICS sectors; at least some should be non-zero
    let nonzero = sector.value.returns.iter().filter(|&&v| v != 0.0).count();
    println!("Sector perf: {nonzero}/11 non-zero returns");
}

#[tokio::test]
#[ignore]
async fn test_poller_macro_yield_curve() {
    let store = make_store(TEST_SYMBOLS);
    macro_fred::poll_yield_curve(&store).await.unwrap();
    let yc = store.yield_curve.read();
    // 3m, 2y, 10y, 30y should all be non-zero
    println!(
        "Yield curve: 3m={:.2}% 2y={:.2}% 10y={:.2}% 30y={:.2}%",
        yc.value.y3m * 100.0,
        yc.value.y2 * 100.0,
        yc.value.y10 * 100.0,
        yc.value.y30 * 100.0
    );
}

#[tokio::test]
#[ignore]
async fn test_poller_macro_fred_series() {
    dotenvy::dotenv().ok();
    let key = std::env::var("FRED_API_KEY").expect("FRED_API_KEY must be set");
    assert!(!key.is_empty(), "FRED_API_KEY is empty");
    finance_query::fred::init(key).expect("FRED init failed");
    let store = make_store(TEST_SYMBOLS);
    macro_fred::poll_fred_macro(&store).await.unwrap();
    let macro_data = store.fred_macro.read();
    let nonzero = macro_data
        .value
        .values
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    assert!(nonzero > 0, "all FRED series returned zero");
    println!("FRED macro: {nonzero}/12 series non-zero");
}

#[tokio::test]
#[ignore]
async fn test_poller_rss_feeds() {
    let store = make_store(TEST_SYMBOLS);
    rss::poll_rss_feeds(&store).await.unwrap();
    let rss = store.rss_feed.read();
    println!(
        "RSS: {} articles cached, sentiment_1h={:.3}",
        rss.value.recent_articles.len(),
        rss.value.sentiment_1h
    );
}

#[tokio::test]
#[ignore]
async fn test_poller_options() {
    let store = make_store(TEST_SYMBOLS);
    options::poll_options_chains(&store).await.unwrap();
    let count = store
        .options
        .iter()
        .filter(|e| !e.value().value.calls.is_empty() || !e.value().value.puts.is_empty())
        .count();
    assert!(count > 0, "no options data populated");
    println!("Options: {count}/{} symbols have data", TEST_SYMBOLS.len());
}

#[tokio::test]
#[ignore]
async fn test_poller_risk() {
    let store = make_store(TEST_SYMBOLS);
    risk::poll_risk_analytics(&store).await.unwrap();
    let count = store.risk.len();
    assert!(count > 0, "no risk data populated");
    println!("Risk: {count} symbols populated");
    for entry in store.risk.iter() {
        println!(
            "  {}: sharpe={:.2} maxDD={:.1}%",
            entry.key(),
            entry.value().value.sharpe,
            entry.value().value.max_drawdown * 100.0
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_poller_analyst() {
    let store = make_store(TEST_SYMBOLS);
    analyst::poll_analyst_ratings(&store).await.unwrap();
    let count = store.analyst.len();
    println!("Analyst: {count} symbols populated");
}

// ── Feature assembly ──────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_feature_assembly_shape() {
    use finance_query::{Interval, Ticker, TimeRange};
    use meridian::features::FeatureRow;
    use meridian::store::{Bar, PriceBuffer};

    let store = make_store(TEST_SYMBOLS);

    // Warm up price buffers from 1-day history
    for sym in TEST_SYMBOLS {
        let ticker = Ticker::new(*sym).await.unwrap();
        let chart = ticker
            .chart(Interval::OneMinute, TimeRange::OneDay)
            .await
            .unwrap();
        let mut buf = PriceBuffer::new(512);
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
        store.prices.insert(sym.to_string(), buf);
    }

    let rows: Vec<FeatureRow> = assembler::assemble_all(&store);
    assert_eq!(rows.len(), TEST_SYMBOLS.len());

    for row in &rows {
        // Every array dimension must match spec
        assert_eq!(
            row.price_volume.len(),
            10,
            "{} price_volume dim",
            row.symbol
        );
        assert_eq!(row.technical.len(), 42, "{} technical dim", row.symbol);
        assert_eq!(row.options.len(), 12, "{} options dim", row.symbol);
        assert_eq!(row.fundamental.len(), 30, "{} fundamental dim", row.symbol);
        assert_eq!(row.risk_analytics.len(), 8, "{} risk dim", row.symbol);
        assert_eq!(row.event_state.len(), 10, "{} event_state dim", row.symbol);
        assert_eq!(
            row.cross_asset_context.len(),
            8,
            "{} cross_asset dim",
            row.symbol
        );
        assert_eq!(row.text.len(), 64, "{} text dim", row.symbol);
        assert_eq!(row.market.len(), 96, "{} market dim", row.symbol);

        // No NaN / Inf in any block
        let all: Vec<f32> = row
            .price_volume
            .iter()
            .chain(row.technical.iter())
            .chain(row.options.iter())
            .chain(row.fundamental.iter())
            .chain(row.risk_analytics.iter())
            .chain(row.event_state.iter())
            .chain(row.cross_asset_context.iter())
            .chain(row.market.iter())
            .copied()
            .collect();
        let bad: Vec<(usize, f32)> = all
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_nan() || v.is_infinite())
            .map(|(i, v)| (i, *v))
            .collect();
        assert!(
            bad.is_empty(),
            "{} has NaN/Inf at indices {:?}",
            row.symbol,
            bad
        );

        println!(
            "{}: price={:.2} tech[0]={:.4} mkt[0]={:.4}",
            row.symbol,
            row.price_volume[3], // close return
            row.technical[0],
            row.market[0]
        );
    }
}
