//! SEC 8-K live-feed context, polled from the finance-query API (the same
//! `/v2/feeds` surface the Python archiver uses; the crate's direct sec.gov
//! feed lacks the SEC-required User-Agent). The event study (§10 step 6)
//! found the vol signal in filing *occurrence*, and the score study found LLM
//! severity adds no ranking power — and the occurrence-flag strategy is held,
//! not shipped. So the feed is context on every alert record, never a trigger.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const USER_AGENT: &str = "meridian-bus/0.2";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedContext {
    /// 8-K entries published within the trailing hour (market-wide, all filers).
    pub eightk_count_1h: usize,
    /// Newest few 8-K titles, for the human reading an alert record.
    pub latest_titles: Vec<String>,
    pub fetched_at: DateTime<Utc>,
}

#[derive(Deserialize)]
struct FeedEntry {
    title: Option<String>,
    published: Option<String>,
}

pub async fn poll_8k(base_url: &str) -> Result<FeedContext> {
    let client = reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .user_agent(USER_AGENT)
        .build()?;
    let url = format!("{base_url}/v2/feeds?sources=sec-filings");
    let entries: Vec<FeedEntry> = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("parsing /v2/feeds response")?;

    let eightks: Vec<&FeedEntry> = entries
        .iter()
        .filter(|e| e.title.as_deref().is_some_and(|t| t.starts_with("8-K")))
        .collect();
    let cutoff = Utc::now() - Duration::hours(1);
    let recent = eightks
        .iter()
        .filter(|e| {
            e.published
                .as_deref()
                .and_then(|p| DateTime::parse_from_rfc3339(p).ok())
                .is_some_and(|t| t.with_timezone(&Utc) > cutoff)
        })
        .count();
    Ok(FeedContext {
        eightk_count_1h: recent,
        latest_titles: eightks
            .iter()
            .take(5)
            .filter_map(|e| e.title.clone())
            .collect(),
        fetched_at: Utc::now(),
    })
}
