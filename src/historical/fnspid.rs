//! FNSPID historical news loader (Block 8 text augmentation).
//!
//! The live inference path builds Block 8 from EDGAR filings ∪ the realtime
//! `feeds` RSS stream. Yahoo/`feeds` expose no dated historical archive, so for
//! the train/val window we substitute the FNSPID corpus (`Zihan1004/FNSPID`,
//! ticker-tagged, dated) as the historical analogue of the live RSS headlines.
//! Both sources flow through the **same** FinBERT scorer + nomic embedder
//! downstream (see `batch.rs`), so only the text *source* differs across
//! train/inference — the representation pipeline is identical, which is the
//! train/inference-alignment requirement in DATA_PIPELINE_ISSUES.md (C1/H1).
//!
//! The Python `market_data.fnspid_ingest` step pre-filters the ~23 GB corpus to
//! the 100-asset universe and writes per-symbol JSONL to
//! `$MERIDIAN_DATA_ROOT/.fnspid_cache/{SYMBOL}.jsonl`, one `{"ts","text"}`
//! object per article (`ts` = unix seconds, `text` = the article headline, which
//! is exactly what the live `news_sentiment` poller scores). A missing or empty
//! cache file means there is genuinely no historical news for that symbol — an
//! honest absence (empty vec), never a fabricated fill.

use std::collections::HashSet;
use std::path::PathBuf;

/// Match the EDGAR primary-document text cap so no single news item can dominate
/// the trailing-window pool.
const MAX_NEWS_CHARS: usize = 20_000;
/// Defensive bound on how many articles we parse per symbol; the universe's
/// busiest names have a few thousand dated articles across the train window.
const MAX_ARTICLES: usize = 20_000;

#[derive(serde::Deserialize)]
struct FnspidRecord {
    ts: i64,
    text: String,
}

/// `$MERIDIAN_DATA_ROOT/.fnspid_cache` (default `./data/.fnspid_cache`), mirroring
/// the `.macro_cache.json` convention so the Python ingestion and the Rust
/// collector agree on a single location.
pub fn cache_dir() -> PathBuf {
    let root = std::env::var("MERIDIAN_DATA_ROOT").unwrap_or_else(|_| "./data".to_string());
    PathBuf::from(root).join(".fnspid_cache")
}

/// Truncate `s` to at most `max` bytes without splitting a UTF-8 char.
fn truncate_on_char_boundary(mut s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s
}

/// Load dated FNSPID news headlines for `symbol` as `(ts, text)` pairs, sorted by
/// `ts`, deduplicated (same ts + text), and capped. Missing/unreadable cache ⇒
/// empty vec (honest absence). Malformed lines are skipped, not fatal — the
/// EDGAR filing floor still carries Block 8 if the news cache is degraded.
pub fn load_news(symbol: &str) -> Vec<(i64, String)> {
    let path = cache_dir().join(format!("{symbol}.jsonl"));
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    parse_news(&content)
}

/// Parse the JSONL cache body into sorted, deduped, capped `(ts, text)` pairs.
/// Split out from `load_news` so the parsing logic is unit-testable without
/// touching the filesystem or the shared `MERIDIAN_DATA_ROOT` env var.
fn parse_news(content: &str) -> Vec<(i64, String)> {
    let mut out: Vec<(i64, String)> = Vec::new();
    let mut seen: HashSet<(i64, String)> = HashSet::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<FnspidRecord>(line) else {
            continue;
        };
        let text = rec.text.trim();
        if text.is_empty() {
            continue;
        }
        let text = truncate_on_char_boundary(text.to_string(), MAX_NEWS_CHARS);
        let key = (rec.ts, text.clone());
        if seen.insert(key) {
            out.push((rec.ts, text));
        }
        if out.len() >= MAX_ARTICLES {
            break;
        }
    }
    out.sort_by_key(|(ts, _)| *ts);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn with_cache<F: FnOnce(&str)>(symbol: &str, lines: &[&str], f: F) {
        let tmp = std::env::temp_dir().join(format!("fnspid_test_{}", std::process::id()));
        let dir = tmp.join(".fnspid_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{symbol}.jsonl"));
        let mut file = std::fs::File::create(&path).unwrap();
        for l in lines {
            writeln!(file, "{l}").unwrap();
        }
        // SAFETY: tests run single-threaded per process here; env is scoped.
        unsafe { std::env::set_var("MERIDIAN_DATA_ROOT", &tmp) };
        f(symbol);
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::remove_var("MERIDIAN_DATA_ROOT") };
    }

    #[test]
    fn missing_cache_is_empty() {
        let tmp = std::env::temp_dir().join("fnspid_missing_xyz");
        unsafe { std::env::set_var("MERIDIAN_DATA_ROOT", &tmp) };
        assert!(load_news("NOPE").is_empty());
        unsafe { std::env::remove_var("MERIDIAN_DATA_ROOT") };
    }

    #[test]
    fn parses_sorts_and_dedups() {
        let lines = [
            r#"{"ts": 200, "text": "later headline"}"#,
            r#"{"ts": 100, "text": "earlier headline"}"#,
            r#"{"ts": 100, "text": "earlier headline"}"#, // exact dup
            r#"{"ts": 150, "text": "   "}"#,              // blank → skip
            r#"not json"#,                                  // malformed → skip
        ];
        with_cache("AAPL", &lines, |sym| {
            let news = load_news(sym);
            assert_eq!(news.len(), 2);
            assert_eq!(news[0], (100, "earlier headline".to_string()));
            assert_eq!(news[1], (200, "later headline".to_string()));
        });
    }

    #[test]
    fn truncates_on_char_boundary() {
        let long = "é".repeat(MAX_NEWS_CHARS); // 2 bytes each → exceeds cap
        let line = format!(r#"{{"ts": 1, "text": "{long}"}}"#);
        with_cache("MSFT", &[line.as_str()], |sym| {
            let news = load_news(sym);
            assert_eq!(news.len(), 1);
            // valid UTF-8 and within the byte cap
            assert!(news[0].1.len() <= MAX_NEWS_CHARS);
            assert!(std::str::from_utf8(news[0].1.as_bytes()).is_ok());
        });
    }
}
