//! EDGAR filing full-document text retrieval for Block 8 (Section 6.2).
//!
//! Yahoo provides no historical news archive, so the historical text timeline is
//! anchored on EDGAR filings, which ARE datable in both the historical and live
//! regimes (identical source → no train/inference gap). For each filing we fetch
//! the *primary document* from the SEC Archives, strip HTML/XBRL to plain text,
//! extract the highest-signal sections (MD&A, Risk Factors, Business description),
//! and hand the dated snippets to the embedder + FinBERT pipeline.
//!
//! Section extraction targets the most signal-dense parts of each form type:
//!   10-K / 20-F / 40-F → Item 7 (MD&A), Item 7A (Market Risk), Item 1A (Risk
//!     Factors), Item 1 (Business)
//!   10-Q → Item 2 (MD&A), Item 1A (Risk Factors), Item 4 (Controls)
//!   8-K  → all Item sections (typically short event-driven disclosures)
//!   6-K  → first ~32K chars (foreign issuer reports vary in structure)
//!
//! The form type comes from finance_query::EdgarFiling.form, which is set by
//! the EDGAR submissions API. When section headers can't be found (unusual
//! formatting, non-standard filer), we fall back to the first MAX_TEXT_CHARS.
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
/// Maximum total characters retained after section extraction. Calibrated to
/// ~90% of harrier-oss's 32K context: 130K chars × ~4.4 chars/token (financial
/// text) ≈ 28.8K tokens, leaving 10% headroom for tokeniser variance.
/// Sections are extracted in priority order (MD&A → Risk Factors → Business);
/// there is no per-section cap — the most signal-dense section gets as much
/// of the global budget as it needs, and subsequent sections fill what remains.
const MAX_TEXT_CHARS: usize = 130_000;
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
            // SEC fair-access requires a real contact in the UA; a placeholder
            // like "@meridian.local" gets 403'd and Block 8 silently collapses
            // to edgar=0. Default to a routable contact; override via the env var.
            "Harvey Tseng harveytseng2@gmail.com".to_string()
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
    /// cache first. Extracts key sections (MD&A, Risk Factors, Business) based
    /// on the filing's form type from finance-query, falling back to the first
    /// MAX_TEXT_CHARS of plain text if section headers aren't found.
    /// Returns plain text (possibly empty).
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
        let full_text = html_to_text(&body, usize::MAX);
        let extracted = extract_key_sections(&full_text, &filing.form, MAX_TEXT_CHARS);
        write_cache(&cache, &extracted)?;
        Ok(extracted)
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

// ═══════════════════════════════════════════════════════════════════════════
// Section extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Section header in a stripped SEC filing text, with its character offset.
struct SectionBoundary {
    /// Character offset into the text where this section starts.
    offset: usize,
    /// The header text (e.g. "Item 7. Management s Discussion and Analysis").
    header: String,
}

/// Extract key sections from stripped SEC filing plain text.
///
/// Uses the `form` field from finance_query's `EdgarFiling` to select which
/// sections to target:
///   * 10-K / 20-F / 40-F → MD&A (Item 7), Risk Factors (1A), Business (1),
///     Market Risk (7A) — assembled in that **priority order** so the most
///     predictive section (MD&A) always gets the budget first.
///   * 10-Q → MD&A (Item 2), Risk Factors (1A), Controls (4), same priority idea.
///   * 8-K → every Item section in document order (short, event-driven).
///
/// For 10-K/10-Q each target item is taken from its **largest** matching body,
/// which skips the tiny table-of-contents stubs (whose body runs only to the
/// next TOC line) in favour of the real section body. Falls back to the first
/// `max_chars` if no usable section is found.
fn extract_key_sections(stripped_text: &str, form: &str, max_chars: usize) -> String {
    let boundaries = find_section_boundaries(stripped_text);

    if boundaries.is_empty() {
        // No section headers found — take the lead of the filing.
        return stripped_text.chars().take(max_chars).collect();
    }

    let form = form.trim().to_ascii_uppercase();
    let is_10k = form == "10-K" || form == "10-K/A" || form == "20-F" || form == "40-F";
    let is_10q = form == "10-Q" || form == "10-Q/A";
    let is_8k = form == "8-K" || form == "8-K/A";

    // Priority-ordered target item prefixes. MD&A leads so it is never starved
    // by an earlier, longer section (Risk Factors / Business) eating the budget.
    let priority: &[&str] = if is_10k {
        &["Item 7.", "Item 1A.", "Item 1.", "Item 7A."]
    } else if is_10q {
        &["Item 2.", "Item 1A.", "Item 4."]
    } else {
        &[]
    };

    let extracted = if is_8k {
        // 8-K: short, event-driven; keep every Item section in document order.
        assemble_document_order(stripped_text, &boundaries, max_chars, is_8k_section)
    } else if !priority.is_empty() {
        assemble_priority_order(stripped_text, &boundaries, max_chars, priority)
    } else {
        // 6-K or unknown: no standard section structure — take the lead.
        return stripped_text.chars().take(max_chars).collect();
    };

    if extracted.is_empty() {
        // No target sections matched — take the lead.
        stripped_text.chars().take(max_chars).collect()
    } else {
        extracted
    }
}

/// `[start, end)` char range of the section body at boundary `i` (its offset to
/// the next boundary, or end-of-text for the last one).
fn section_body_range(boundaries: &[SectionBoundary], i: usize, text_len: usize) -> (usize, usize) {
    let start = boundaries[i].offset;
    let end = boundaries
        .get(i + 1)
        .map(|b| b.offset)
        .unwrap_or(text_len);
    (start, end)
}

/// Append `section` to `dst` (with a "\n\n" separator), clamped on a char
/// boundary to the remaining `max_chars` budget. No-op once the budget is spent.
fn append_clamped(dst: &mut String, section: &str, max_chars: usize) {
    let sep = if dst.is_empty() { 0 } else { 2 };
    let remaining = max_chars.saturating_sub(dst.len() + sep);
    if remaining == 0 {
        return;
    }
    let piece = if section.len() > remaining {
        let mut end = remaining;
        while end > 0 && !section.is_char_boundary(end) {
            end -= 1;
        }
        &section[..end]
    } else {
        section
    };
    if piece.is_empty() {
        return;
    }
    if !dst.is_empty() {
        dst.push_str("\n\n");
    }
    dst.push_str(piece);
}

/// Assemble target sections in `priority` order. For each prefix, pick the
/// matching boundary with the **largest** body — the real section, not the
/// table-of-contents stub — and append it within the global budget.
fn assemble_priority_order(
    text: &str,
    boundaries: &[SectionBoundary],
    max_chars: usize,
    priority: &[&str],
) -> String {
    let mut extracted = String::with_capacity(max_chars.min(text.len()));
    for prefix in priority {
        let best = boundaries
            .iter()
            .enumerate()
            .filter(|(_, b)| b.header.starts_with(prefix))
            .map(|(i, _)| {
                let (s, e) = section_body_range(boundaries, i, text.len());
                (i, e.saturating_sub(s))
            })
            .max_by_key(|(_, len)| *len);
        let Some((i, _)) = best else { continue };
        let (s, e) = section_body_range(boundaries, i, text.len());
        append_clamped(&mut extracted, &text[s..e], max_chars);
        if extracted.len() >= max_chars {
            break;
        }
    }
    extracted
}

/// Assemble every section whose header satisfies `keep`, in document order.
fn assemble_document_order(
    text: &str,
    boundaries: &[SectionBoundary],
    max_chars: usize,
    keep: impl Fn(&str) -> bool,
) -> String {
    let mut extracted = String::new();
    for (i, b) in boundaries.iter().enumerate() {
        if !keep(&b.header) {
            continue;
        }
        let (s, e) = section_body_range(boundaries, i, text.len());
        append_clamped(&mut extracted, &text[s..e], max_chars);
        if extracted.len() >= max_chars {
            break;
        }
    }
    extracted
}

/// Find the next case-insensitive `"item "` at/after byte `from`, returning its
/// offset. ASCII-only: filings write headers as "Item"/"ITEM"/"item", and the
/// ASCII pattern can only align on char boundaries (UTF-8 continuation/lead
/// bytes are all ≥ 0x80, never matching ASCII), so the returned offset is safe.
fn find_item_ci(text: &str, from: usize) -> Option<usize> {
    const PAT: &[u8] = b"item ";
    let bytes = text.as_bytes();
    if bytes.len() < PAT.len() {
        return None;
    }
    (from..=bytes.len() - PAT.len()).find(|&i| {
        bytes[i..i + PAT.len()]
            .iter()
            .zip(PAT)
            .all(|(b, p)| b.eq_ignore_ascii_case(p))
    })
}

/// Scan collapsed plain text for "Item N." section headers, returning their
/// offsets. Works on the output of `html_to_text` which collapses all whitespace
/// into single spaces (no newlines remain), so we search the flat string. Header
/// matching is case-insensitive so all-caps filings ("ITEM 7. ...") are caught.
fn find_section_boundaries(text: &str) -> Vec<SectionBoundary> {
    let mut boundaries: Vec<SectionBoundary> = Vec::new();
    let mut search_from = 0usize;

    while let Some(abs_pos) = find_item_ci(text, search_from) {
        // Require whitespace (or start-of-text) before "Item " to avoid
        // matching prose like "discontinued items. Item 7 covers..." etc.
        if abs_pos > 0 && !text.as_bytes()[abs_pos - 1].is_ascii_whitespace() {
            search_from = abs_pos + 5;
            continue;
        }

        let candidate = &text[abs_pos..];
        if let Some(header) = try_parse_header(candidate) {
            boundaries.push(SectionBoundary {
                offset: abs_pos,
                header,
            });
        }
        search_from = abs_pos + 5; // skip past "Item "
    }

    boundaries
}

/// Try to parse "Item N. Description" from the start of `s`.
///
/// The `Item` prefix is matched case-insensitively ("ITEM"/"item" both occur in
/// filings) and the section number is upper-cased, so the emitted header is
/// always canonical — e.g. "item 7a. ..." → "Item 7A. ...". This lets the
/// downstream prefix checks ("Item 7A.") stay case-sensitive against a
/// normalized header. The description is the next ~10 words after the number.
fn try_parse_header(s: &str) -> Option<String> {
    if s.len() < 5 || !s.as_bytes()[..5].eq_ignore_ascii_case(b"item ") {
        return None;
    }
    let rest = &s[5..];

    // Expect a section number/letter: digits, optionally followed by letters
    let dot_pos = rest.find('.')?;
    if dot_pos == 0 {
        return None;
    }
    let num_part = &rest[..dot_pos];
    if !num_part.chars().all(|c| c.is_alphanumeric()) {
        return None;
    }
    // Canonicalize "7a" → "7A" so prefix checks match regardless of source case.
    let num_part = num_part.to_ascii_uppercase();

    // Take the next few words as the description
    let after_dot = rest[dot_pos + 1..].trim_start();
    let desc_words: Vec<&str> = after_dot
        .split_whitespace()
        .take(10)
        .collect();
    if desc_words.len() < 2 {
        return None;
    }
    let desc = desc_words.join(" ");

    Some(format!("Item {}. {}", num_part, desc))
}

/// True for any 8-K item section.
fn is_8k_section(header: &str) -> bool {
    let h = header;
    h.starts_with("Item 1.") || h.starts_with("Item 2.") || h.starts_with("Item 3.")
        || h.starts_with("Item 4.") || h.starts_with("Item 5.") || h.starts_with("Item 6.")
        || h.starts_with("Item 7.") || h.starts_with("Item 8.") || h.starts_with("Item 9.")
}

// ═══════════════════════════════════════════════════════════════════════════
// HTML stripping (unchanged)
// ═══════════════════════════════════════════════════════════════════════════

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
        .replace("&#160;", " ")
        .replace("&#xa0;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'");

    let collapsed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
    if max_chars == usize::MAX {
        collapsed
    } else if collapsed.len() > max_chars {
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

    // ── section extraction tests ──

    #[test]
    fn try_parse_header_detection() {
        assert_eq!(
            try_parse_header("Item 1. Business description here").as_deref(),
            Some("Item 1. Business description here")
        );
        assert_eq!(
            try_parse_header("Item 1A. Risk Factors and other things").as_deref(),
            Some("Item 1A. Risk Factors and other things")
        );
        assert_eq!(
            try_parse_header("Item 7. Management s Discussion and Analysis of Results").as_deref(),
            Some("Item 7. Management s Discussion and Analysis of Results")
        );
        assert!(try_parse_header("Item").is_none());           // no section number
        assert!(try_parse_header("Item . Business").is_none()); // no number before dot
        assert!(try_parse_header("Part I").is_none());         // not Item
        assert!(try_parse_header("X Item 1.").is_none());     // doesn't start with Item
    }

    #[test]
    fn finds_boundaries_in_flat_text() {
        // Simulates collapsed html_to_text output — single spaces between sections
        let text = "Leading text. Item 1. Business Apple Inc. designs. Item 1A. Risk Factors The company faces. Item 7. Management s Discussion Net sales increased. Item 7A. Quantitative Disclosures Interest rate. Item 8. Financial Statements skipped.";
        let boundaries = find_section_boundaries(text);
        assert!(boundaries.len() >= 4, "found {} boundaries", boundaries.len());
        assert!(boundaries.iter().any(|b| b.header.starts_with("Item 1.")));
        assert!(boundaries.iter().any(|b| b.header.starts_with("Item 1A.")));
        assert!(boundaries.iter().any(|b| b.header.starts_with("Item 7.")));
        assert!(boundaries.iter().any(|b| b.header.starts_with("Item 7A.")));
    }

    #[test]
    fn extracts_10k_sections() {
        let text = "\
Item 1. Business
BUSINESS CONTENT.
Item 1A. Risk Factors
RISK CONTENT.
Item 2. Properties
SKIP THIS.
Item 7. Management s Discussion
MDNA CONTENT.
Item 7A. Quantitative Disclosures
MARKET RISK CONTENT.
Item 8. Financial Statements
SKIP THIS TOO.
";
        let result = extract_key_sections(text, "10-K", 10_000);
        assert!(result.contains("BUSINESS CONTENT"));
        assert!(result.contains("RISK CONTENT"));
        assert!(result.contains("MDNA CONTENT"));
        assert!(result.contains("MARKET RISK CONTENT"));
        assert!(!result.contains("SKIP THIS"));
        assert!(!result.contains("SKIP THIS TOO"));
    }

    #[test]
    fn extracts_10q_sections() {
        let text = "\
Item 1. Financial Statements
FS CONTENT.
Item 2. Management s Discussion
QUARTERLY MDNA.
Item 3. Quantitative Disclosures
SKIP.
Item 4. Controls and Procedures
CONTROLS.
";
        let result = extract_key_sections(text, "10-Q", 10_000);
        assert!(result.contains("QUARTERLY MDNA"));
        assert!(result.contains("CONTROLS"));
        assert!(!result.contains("FS CONTENT")); // Item 1 Financial Statements ≠ Item 1. Business
        assert!(!result.contains("SKIP"));
    }

    #[test]
    fn extracts_8k_all_items() {
        let text = "\
Item 1.01 Entry into Material Definitive Agreement
AGREEMENT TEXT.
Item 2.02 Results of Operations
RESULTS TEXT.
Item 9.01 Financial Statements and Exhibits
EXHIBITS TEXT.
";
        let result = extract_key_sections(text, "8-K", 10_000);
        assert!(result.contains("AGREEMENT TEXT"));
        assert!(result.contains("RESULTS TEXT"));
        assert!(result.contains("EXHIBITS TEXT"));
    }

    #[test]
    fn falls_back_when_no_sections() {
        let text = "Plain text with no Item headers at all.";
        let result = extract_key_sections(text, "10-K", 100);
        assert!(result.contains("Plain text"));
    }

    #[test]
    fn global_cap_truncates_across_sections() {
        // Build two large sections that exceed MAX_TEXT_CHARS together.
        // MD&A gets priority (first in extraction order); Business gets whatever remains.
        let mut text = String::new();
        // Item 7 (MD&A) — huge, would eat most of the budget
        text.push_str("Item 7. MDNA\n");
        text.push_str(&"M".repeat(100_000));
        text.push_str("\n");
        // Item 1 (Business) — also large, but MD&A consumed the budget
        text.push_str("Item 1. Business\n");
        text.push_str(&"B".repeat(100_000));
        text.push_str("\n");
        // Item 7A (Market Risk) — small, should get nothing if budget exhausted
        text.push_str("Item 7A. Market Risk\nshort risk text\n");

        let result = extract_key_sections(&text, "10-K", MAX_TEXT_CHARS);
        // Should be capped at MAX_TEXT_CHARS
        assert!(result.len() <= MAX_TEXT_CHARS, "result {} > MAX {}", result.len(), MAX_TEXT_CHARS);
        // MD&A should dominate
        assert!(result.contains("Item 7. MDNA"));
        // Business may or may not be included depending on what MD&A consumed
        // Market Risk likely excluded since budget exhausted after MD&A + Business
    }

    #[test]
    fn matches_all_caps_item_headers() {
        // Filers that write headers in upper case must still be parsed; otherwise
        // the whole filing falls back to lead-text and MD&A is lost.
        let text = "\
ITEM 1. BUSINESS
BUSINESS CONTENT.
ITEM 1A. RISK FACTORS
RISK CONTENT.
ITEM 7. MANAGEMENT S DISCUSSION
MDNA CONTENT.
ITEM 8. FINANCIAL STATEMENTS
SKIP THIS.
";
        let result = extract_key_sections(text, "10-K", 10_000);
        assert!(result.contains("MDNA CONTENT"), "MD&A from ALL-CAPS header missing");
        assert!(result.contains("RISK CONTENT"));
        assert!(result.contains("BUSINESS CONTENT"));
        assert!(!result.contains("SKIP THIS"));
    }

    #[test]
    fn skips_table_of_contents_and_emits_in_priority_order() {
        // A realistic 10-K: a table of contents (short stubs with page numbers)
        // followed by the real section bodies, collapsed to single spaces like
        // html_to_text output. The TOC entries match the same "Item N." headers,
        // so naive document-order extraction would grab the stubs and could
        // starve MD&A.
        let toc = "TABLE OF CONTENTS Item 1. Business 4 Item 1A. Risk Factors 12 \
                   Item 7. Management s Discussion 35 Item 7A. Market Risk 60 ";
        let body = format!(
            "PART I Item 1. Business {} Item 1A. Risk Factors {} \
             Item 7. Management s Discussion and Analysis {} \
             Item 7A. Quantitative Disclosures {} Item 8. Financial Statements {}",
            "BUSINESSBODY ".repeat(40),
            "RISKBODY ".repeat(40),
            "MDNABODY ".repeat(40),
            "MARKETRISKBODY ".repeat(10),
            "SKIPSTATEMENTS ".repeat(40),
        );
        let text = format!("{toc}{body}");
        let result = extract_key_sections(&text, "10-K", 100_000);

        // Real section bodies captured — not just the TOC stubs / page numbers.
        assert!(result.contains("MDNABODY"), "real MD&A body missing");
        assert!(result.contains("RISKBODY"), "real Risk Factors body missing");
        assert!(result.contains("BUSINESSBODY"), "real Business body missing");
        // Item 8 (Financial Statements) is not a target section.
        assert!(!result.contains("SKIPSTATEMENTS"), "Item 8 should be excluded");
        // MD&A leads the output ahead of Business (priority, not document order).
        let mdna_at = result.find("MDNABODY").unwrap();
        let biz_at = result.find("BUSINESSBODY").unwrap();
        assert!(mdna_at < biz_at, "MD&A should be emitted before Business");
    }
}
