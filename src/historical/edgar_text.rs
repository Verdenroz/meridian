//! EDGAR filing full-document text retrieval for Block 8 (Section 6.2).
//!
//! Yahoo provides no historical news archive, so the historical text timeline is
//! anchored on EDGAR filings, which ARE datable in both the historical and live
//! regimes (identical source → no train/inference gap). For each filing we fetch
//! the *primary document* from the SEC Archives, strip HTML/XBRL to plain text,
//! and hand the dated snippets to the embedder + FinBERT pipeline.
//!
//! SEC fair-access policy requires a descriptive `User-Agent` carrying a contact
//! address and caps clients at ~10 requests/second. We honour both: the UA comes
//! from `MERIDIAN_SEC_UA` (or a sane default) and requests are throttled. Document
//! text is cached on disk by accession number so re-collection is free.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;

use finance_query::EdgarSubmissions;

/// SEC asks automated clients to stay at/under 10 req/s. We use a conservative
/// minimum spacing between requests to stay well within that.
const MIN_REQUEST_SPACING: Duration = Duration::from_millis(150);
/// Cap stripped document text so embedding cost stays bounded; the embedder
/// server token-truncates anyway and the lead of a filing carries the signal.
const MAX_TEXT_CHARS: usize = 20_000;
/// Only fetch document text for forms that carry signal-bearing prose. Form 4
/// (insider) / 13F etc. are handled by their own structured feature blocks.
const TEXT_FORMS: &[&str] = &[
    "10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A", "6-K", "20-F", "40-F",
];

#[derive(Clone, Debug)]
pub struct EdgarFilingDoc {
    pub accession_number: String,
    pub form: String,
    /// Unix timestamp (UTC midnight) parsed from `filing_date` (YYYY-MM-DD).
    pub filing_ts: i64,
    pub primary_document: String,
}

#[derive(Clone)]
pub struct EdgarTextConfig {
    pub user_agent: String,
    pub cache_dir: PathBuf,
}

impl Default for EdgarTextConfig {
    fn default() -> Self {
        let user_agent = std::env::var("MERIDIAN_SEC_UA").unwrap_or_else(|_| {
            "meridian-research admin@meridian.local".to_string()
        });
        Self {
            user_agent,
            cache_dir: PathBuf::from("data/edgar_text"),
        }
    }
}

pub struct EdgarTextFetcher {
    cfg: EdgarTextConfig,
    http: reqwest::Client,
    last_request: Arc<Mutex<Option<Instant>>>,
}

impl EdgarTextFetcher {
    pub fn new(cfg: EdgarTextConfig) -> Result<Self> {
        std::fs::create_dir_all(&cfg.cache_dir)
            .with_context(|| format!("create edgar text cache dir {:?}", cfg.cache_dir))?;
        let http = reqwest::Client::builder()
            .user_agent(cfg.user_agent.clone())
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            cfg,
            http,
            last_request: Arc::new(Mutex::new(None)),
        })
    }

    /// Block until at least `MIN_REQUEST_SPACING` has elapsed since the previous
    /// SEC request, then record this request's start time.
    async fn throttle(&self) {
        let mut guard = self.last_request.lock().await;
        if let Some(prev) = *guard {
            let elapsed = prev.elapsed();
            if elapsed < MIN_REQUEST_SPACING {
                tokio::time::sleep(MIN_REQUEST_SPACING - elapsed).await;
            }
        }
        *guard = Some(Instant::now());
    }

    fn cache_path(&self, accession: &str) -> PathBuf {
        // Accession numbers are unique filing IDs; strip dashes for a flat key.
        let key = accession.replace('-', "");
        self.cfg.cache_dir.join(format!("{key}.txt"))
    }

    /// Fetch + strip the primary document text for one filing, hitting the disk
    /// cache first. Returns plain text (possibly empty).
    pub async fn fetch_document_text(&self, cik: &str, filing: &EdgarFilingDoc) -> Result<String> {
        let cache = self.cache_path(&filing.accession_number);
        if let Ok(cached) = std::fs::read_to_string(&cache) {
            return Ok(cached);
        }
        if filing.primary_document.is_empty() {
            return Ok(String::new());
        }

        let url = document_url(cik, &filing.accession_number, &filing.primary_document);
        self.throttle().await;
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        if !resp.status().is_success() {
            // Missing/blocked documents are non-fatal: cache empty so we don't retry.
            let _ = write_cache(&cache, "");
            return Ok(String::new());
        }
        let body = resp.text().await.with_context(|| format!("read body {url}"))?;
        let text = html_to_text(&body, MAX_TEXT_CHARS);
        write_cache(&cache, &text)?;
        Ok(text)
    }

    /// All text-bearing filings for an entity, newest-first, across the inline
    /// `recent` block and any paginated history files (needed to reach back to
    /// the start of the training window for active filers).
    pub async fn gather_filings(&self, subs: &EdgarSubmissions) -> Vec<EdgarFilingDoc> {
        let mut out: Vec<EdgarFilingDoc> = Vec::new();
        if let Some(filings) = subs.filings.as_ref() {
            if let Some(recent) = filings.recent.as_ref() {
                let docs: Vec<EdgarFilingDoc> = recent
                    .to_filings()
                    .into_iter()
                    .filter_map(|f| {
                        Some(EdgarFilingDoc {
                            filing_ts: parse_filing_date(&f.filing_date)?,
                            accession_number: f.accession_number,
                            form: f.form,
                            primary_document: f.primary_document,
                        })
                    })
                    .collect();
                collect_filings(&docs, &mut out);
            }
            // Older history is split into additional JSON files under
            // https://data.sec.gov/submissions/{name}.
            for file in &filings.files {
                if let Some(name) = file.name.as_ref() {
                    if let Some(docs) = self.fetch_history_file(name).await {
                        collect_filings(&docs, &mut out);
                    }
                }
            }
        }
        out.sort_by(|a, b| b.filing_ts.cmp(&a.filing_ts));
        out
    }

    async fn fetch_history_file(&self, name: &str) -> Option<Vec<EdgarFilingDoc>> {
        let url = format!("https://data.sec.gov/submissions/{name}");
        self.throttle().await;
        let resp = self.http.get(&url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json: serde_json::Value = resp.json().await.ok()?;
        Some(parse_history_file(&json))
    }
}

/// `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}`
fn document_url(cik: &str, accession: &str, primary_document: &str) -> String {
    let cik_num = cik.trim_start_matches('0');
    let acc_nodash = accession.replace('-', "");
    format!(
        "https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/{primary_document}"
    )
}

fn collect_filings(docs: &[EdgarFilingDoc], out: &mut Vec<EdgarFilingDoc>) {
    for d in docs {
        if is_text_form(&d.form) {
            out.push(d.clone());
        }
    }
}

fn is_text_form(form: &str) -> bool {
    let f = form.trim().to_ascii_uppercase();
    TEXT_FORMS.iter().any(|t| t.eq_ignore_ascii_case(&f))
}

/// Parse YYYY-MM-DD into a UTC-midnight unix timestamp.
pub fn parse_filing_date(s: &str) -> Option<i64> {
    let d = chrono::NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()?;
    let dt = d.and_hms_opt(0, 0, 0)?;
    Some(dt.and_utc().timestamp())
}

/// Parse the columnar `filing_date/form/accession/primaryDocument` arrays of a
/// paginated history file into filing docs.
fn parse_history_file(json: &serde_json::Value) -> Vec<EdgarFilingDoc> {
    let arr = |k: &str| -> Vec<String> {
        json.get(k)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|x| x.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let accession = arr("accessionNumber");
    let filing_date = arr("filingDate");
    let form = arr("form");
    let primary_document = arr("primaryDocument");
    let n = accession.len();
    (0..n)
        .filter_map(|i| {
            let ts = parse_filing_date(filing_date.get(i)?)?;
            Some(EdgarFilingDoc {
                accession_number: accession.get(i).cloned().unwrap_or_default(),
                form: form.get(i).cloned().unwrap_or_default(),
                filing_ts: ts,
                primary_document: primary_document.get(i).cloned().unwrap_or_default(),
            })
        })
        .collect()
}

fn write_cache(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Minimal, dependency-free HTML/XBRL → text: drop `<script>`/`<style>` blocks,
/// strip tags, decode a handful of common entities, collapse whitespace, and
/// truncate to `max_chars`. Good enough to embed; we do not need a DOM.
pub fn html_to_text(html: &str, max_chars: usize) -> String {
    let without_blocks = strip_block(&strip_block(html, "script"), "style");

    let mut out = String::with_capacity(without_blocks.len() / 2);
    let mut in_tag = false;
    for ch in without_blocks.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }

    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'");

    let collapsed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() > max_chars {
        // Truncate on a char boundary.
        let mut end = max_chars;
        while end > 0 && !collapsed.is_char_boundary(end) {
            end -= 1;
        }
        collapsed[..end].to_string()
    } else {
        collapsed
    }
}

/// Remove `<tag ...> ... </tag>` blocks (case-insensitive) entirely.
fn strip_block(html: &str, tag: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0usize;
    while let Some(rel) = lower[cursor..].find(&open) {
        let start = cursor + rel;
        out.push_str(&html[cursor..start]);
        match lower[start..].find(&close) {
            Some(crel) => cursor = start + crel + close.len(),
            None => {
                cursor = html.len();
                break;
            }
        }
    }
    out.push_str(&html[cursor..]);
    out
}

/// Build the per-accession text map used by the embedding stage, keyed by the
/// text content so it shares the embedder cache with other snippet sources.
pub type DocTextMap = HashMap<String, String>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_html_and_collapses() {
        let html = "<html><head><style>.x{color:red}</style></head>\
                    <body><p>Net&nbsp;income rose 10%.</p>\
                    <script>evil()</script><div>Outlook&amp;risks.</div></body></html>";
        let text = html_to_text(html, 1000);
        assert!(text.contains("Net income rose 10%."));
        assert!(text.contains("Outlook&risks."));
        assert!(!text.contains("evil"));
        assert!(!text.contains("color:red"));
    }

    #[test]
    fn truncates_on_char_boundary() {
        let html = "<p>".to_string() + &"é".repeat(100) + "</p>";
        let text = html_to_text(&html, 50);
        assert!(text.len() <= 50);
    }

    #[test]
    fn document_url_strips_leading_zeros_and_dashes() {
        let url = document_url("0000320193", "0000320193-23-000106", "aapl-20230930.htm");
        assert_eq!(
            url,
            "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
        );
    }

    #[test]
    fn parses_filing_date() {
        assert_eq!(parse_filing_date("2023-09-30"), Some(1696032000));
    }

    #[test]
    fn only_text_forms_kept() {
        assert!(is_text_form("10-K"));
        assert!(is_text_form("8-K/A"));
        assert!(!is_text_form("4"));
        assert!(!is_text_form("13F-HR"));
    }
}
