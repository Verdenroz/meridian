use crate::events::detector::detect_fomc_from_articles;
use crate::store::{CachedValue, DataStore, RssArticle, RssFeedState};
use chrono::Utc;
use finance_query::feeds::{self, FeedSource};
use std::sync::Arc;

/// All RSS feed sources used by Meridian.
fn all_sources() -> Vec<FeedSource> {
    vec![
        FeedSource::FederalReserve,
        FeedSource::SecPressReleases,
        FeedSource::SecFilings("8-K".to_string()),
        FeedSource::Bea,
        FeedSource::Ecb,
        FeedSource::Cfpb,
        FeedSource::BankOfEngland,
        FeedSource::MarketWatch,
        FeedSource::Cnbc,
        FeedSource::Bloomberg,
        FeedSource::FinancialTimes,
        FeedSource::NytBusiness,
        FeedSource::GuardianBusiness,
        FeedSource::WsjMarkets,
        FeedSource::Fortune,
        FeedSource::BusinessWire,
        FeedSource::FinancialPost,
        FeedSource::FtLex,
        FeedSource::CoinDesk,
        FeedSource::CoinTelegraph,
        FeedSource::TechCrunch,
        FeedSource::HackerNews,
        FeedSource::VentureBeat,
        FeedSource::YCombinator,
        FeedSource::Investing,
        FeedSource::OilPrice,
        FeedSource::CalculatedRisk,
        FeedSource::RitholtzBigPicture,
        FeedSource::TheEconomist,
        FeedSource::Scmp,
        FeedSource::NikkeiAsia,
    ]
}

/// Poll all RSS sources, deduplicate by URL, update the shared feed state.
pub async fn poll_rss_feeds(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let entries = feeds::fetch_all(&all_sources()).await?;

    let now = Utc::now();
    let mut state = store.rss_feed.read().value.clone();

    // Deduplicate: keep only articles with URLs not already seen
    let new_articles: Vec<RssArticle> = entries
        .into_iter()
        .filter(|e| !state.seen_urls.contains(&e.url))
        .map(|e| RssArticle {
            url: e.url,
            title: e.title,
            source: e.source,
            fetched_at: now,
            sentiment: 0.0, // MarketReader stub
        })
        .collect();

    for a in &new_articles {
        state.seen_urls.insert(a.url.clone());
    }

    // Trim seen_urls to prevent unbounded growth (keep last 50,000)
    if state.seen_urls.len() > 50_000 {
        state.seen_urls.clear();
    }

    // Update aggregate sentiment signals (stubs until MarketReader is wired)
    update_aggregate_signals(&mut state, &new_articles, now);

    // Detect FOMC events
    let fomc_articles: Vec<RssArticle> = new_articles
        .iter()
        .filter(|a| a.source.to_lowercase().contains("federal"))
        .cloned()
        .collect();
    let fomc_events = detect_fomc_from_articles(&fomc_articles);
    for event in fomc_events {
        tracing::info!(kind = ?event.kind, desc = %event.description, "FOMC event detected");
        // Market-wide events broadcast to all assets
        for sym in &store.symbols {
            let mut es = store.event_state.entry(sym.clone()).or_default();
            *es = es.with_event(&event);
        }
    }

    // Append new articles to rolling buffer (keep last 1,000)
    state.recent_articles.extend(new_articles);
    if state.recent_articles.len() > 1_000 {
        let drain_to = state.recent_articles.len() - 1_000;
        state.recent_articles.drain(..drain_to);
    }

    // Write updated state
    let new_cached = CachedValue::new(state);
    *store.rss_feed.write() = new_cached;

    tracing::debug!("RSS poll complete");
    Ok(())
}

fn update_aggregate_signals(
    state: &mut RssFeedState,
    new_articles: &[RssArticle],
    now: chrono::DateTime<Utc>,
) {
    let cutoff_1h = now - chrono::Duration::hours(1);
    let cutoff_24h = now - chrono::Duration::hours(24);

    // Count articles in time windows from the rolling buffer + new
    let count_1h = state
        .recent_articles
        .iter()
        .filter(|a| a.fetched_at > cutoff_1h)
        .count()
        + new_articles.iter().filter(|a| a.fetched_at > cutoff_1h).count();

    let count_24h = state
        .recent_articles
        .iter()
        .filter(|a| a.fetched_at > cutoff_24h)
        .count()
        + new_articles.iter().filter(|a| a.fetched_at > cutoff_24h).count();

    state.volume_1h = count_1h as f64;
    state.volume_24h = count_24h as f64;

    // Source-specific counts
    let is_source = |a: &RssArticle, kw: &str| a.source.to_lowercase().contains(kw);

    state.fed_count_24h = state
        .recent_articles
        .iter()
        .filter(|a| a.fetched_at > cutoff_24h && is_source(a, "federal"))
        .count() as f64
        + new_articles.iter().filter(|a| is_source(a, "federal")).count() as f64;

    state.sec_count_24h = state
        .recent_articles
        .iter()
        .filter(|a| a.fetched_at > cutoff_24h && is_source(a, "sec"))
        .count() as f64
        + new_articles.iter().filter(|a| is_source(a, "sec")).count() as f64;

    // Sentiment stubs (0.0 until MarketReader) — topic diversity from distinct sources
    let sources_1h: std::collections::HashSet<&str> = state
        .recent_articles
        .iter()
        .filter(|a| a.fetched_at > cutoff_1h)
        .map(|a| a.source.as_str())
        .collect();
    state.topic_diversity = sources_1h.len() as f64;

    // EMA-based news surprise: current volume vs EWMA baseline
    let ema_alpha = 0.05;
    let baseline = if state.news_surprise > 0.0 {
        state.news_surprise * (1.0 - ema_alpha) + count_1h as f64 * ema_alpha
    } else {
        count_1h as f64
    };
    state.news_surprise = (count_1h as f64 - baseline).max(0.0);
}
