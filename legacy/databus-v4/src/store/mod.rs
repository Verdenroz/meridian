pub mod cached_value;
pub mod price_buffer;
pub mod rolling_stats;

pub use cached_value::CachedValue;
pub use price_buffer::{Bar, FormingBar, PriceBuffer};
pub use rolling_stats::RollingStatsStore;

use crate::events::EventState;
use dashmap::DashMap;
use finance_query::{OptionContract, Split};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::Arc;

// ── Our own extracted types (avoid naming private finance-query models) ────────

/// Extracted insider transaction record.
#[derive(Clone, Default)]
pub struct OurInsiderTx {
    pub filer_name: Option<String>,
    pub transaction_text: Option<String>,
    /// Unix timestamp of the transaction start date
    pub start_date_ts: Option<i64>,
    pub shares: Option<i64>,
    pub value: Option<i64>,
}

/// Extracted analyst grade change record.
#[derive(Clone, Default)]
pub struct OurGradeChange {
    pub firm: Option<String>,
    pub from_grade: Option<String>,
    pub to_grade: Option<String>,
    /// Action string: "up", "down", "init", "reit", etc.
    pub action: Option<String>,
    /// ISO date string "YYYY-MM-DD" (derived from epoch_grade_date)
    pub date: Option<String>,
}

/// One period's analyst recommendation counts.
#[derive(Clone, Default)]
pub struct OurRecoPeriod {
    pub strong_buy: i32,
    pub buy: i32,
    pub hold: i32,
    pub sell: i32,
    pub strong_sell: i32,
}

/// Extracted SEC filing record.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct OurSecFiling {
    pub filing_type: Option<String>,
    pub date: Option<String>,
}

/// Extracted calendar events.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct OurCalendar {
    /// Unix timestamp of next earnings date
    pub next_earnings_ts: Option<i64>,
    /// Unix timestamp of ex-dividend date
    pub ex_div_ts: Option<i64>,
}

// ── Per-asset cache types ─────────────────────────────────────────────────────

#[derive(Clone, Default)]
pub struct FundamentalData {
    pub trailing_pe: Option<f64>,
    pub forward_pe: Option<f64>,
    pub peg_ratio: Option<f64>,
    pub ev_to_ebitda: Option<f64>,
    pub dividend_yield: Option<f64>,
    pub profit_margins: Option<f64>,
    pub revenue_growth: Option<f64>,
    pub earnings_growth: Option<f64>,
    pub debt_to_equity: Option<f64>,
    pub free_cashflow: Option<f64>,
    pub market_cap: Option<f64>,
    pub return_on_equity: Option<f64>,
    pub current_ratio: Option<f64>,
    // Earnings intelligence
    pub last_earnings_surprise_pct: Option<f64>,
    pub earnings_surprise_streak: i32,
    pub next_q_est_revision_30d: Option<f64>,
    pub next_q_est_revision_90d: Option<f64>,
    pub rev_est_revision_30d: Option<f64>,
    pub days_since_earnings: Option<f64>,
    // Analyst
    pub analyst_consensus: Option<f64>,
    pub strong_buy_fraction: Option<f64>,
    pub rating_momentum_30d: Option<f64>,
    pub recent_grade_direction: f64,
    // Ownership
    pub net_insider_shares_90d: Option<f64>,
    pub insider_buy_count_90d: f64,
    pub net_insider_value_90d: Option<f64>,
    pub insider_ownership_pct: Option<f64>,
    pub institutional_pct: Option<f64>,
    pub institutional_delta_qtr: Option<f64>,
    // Identity
    pub sector_idx: f64,
    pub mktcap_log_rank: f64,
}

#[derive(Clone, Default)]
pub struct RiskData {
    pub var_95: f64,
    pub var_99: f64,
    pub parametric_var_95: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub beta: f64,
    pub max_drawdown: f64,
    pub max_dd_recovery: f64,
}

#[derive(Clone, Default)]
pub struct AnalystData {
    pub grading_history: Vec<OurGradeChange>,
    pub recommendation_period: Option<OurRecoPeriod>,
}

#[derive(Clone, Default)]
pub struct OptionsData {
    pub calls: Vec<OptionContract>,
    pub puts: Vec<OptionContract>,
}

// ── Market-wide cache types ───────────────────────────────────────────────────

#[derive(Clone, Default)]
pub struct WorldIndices {
    pub returns: std::collections::HashMap<String, f64>,
    pub dxy: f64,
    pub gold: f64,
    pub oil: f64,
    pub eurusd: f64,
    pub usdjpy: f64,
    pub vix_level: f64,
    pub vix_ret: f64,
}

#[derive(Clone, Default)]
pub struct CryptoMarket {
    pub btc_price: f64,
    pub btc_return_1d: f64,
    pub eth_price: f64,
    pub eth_return_1d: f64,
    pub total_mktcap: f64,
    pub btc_dominance: f64,
}

#[derive(Clone, Default)]
pub struct FearGreedData {
    pub value: f64,
    pub classification: u8,
}

#[derive(Clone, Default)]
pub struct SectorData {
    pub returns: [f64; 11],
    pub dispersion: f64,
    pub best_sector: usize,
    pub worst_sector: usize,
    pub rotation_velocity: f64,
    pub breadth_pct_above_sma200: f64,
}

#[derive(Clone, Default)]
pub struct YieldCurveData {
    pub y1m: f64,
    pub y3m: f64,
    pub y6m: f64,
    pub y1: f64,
    pub y2: f64,
    pub y3: f64,
    pub y5: f64,
    pub y7: f64,
    pub y10: f64,
    pub y20: f64,
    pub y30: f64,
    pub spread_2y10y: f64,
}

#[derive(Clone, Default)]
pub struct MacroData {
    pub values: [f64; 12],
    pub last_observation_dates: [String; 12],
}

#[derive(Clone, Default)]
pub struct RssFeedState {
    pub recent_articles: Vec<RssArticle>,
    pub seen_urls: std::collections::HashSet<String>,
    pub sentiment_1h: f64,
    pub sentiment_24h: f64,
    pub volume_1h: f64,
    pub volume_24h: f64,
    pub fed_count_24h: f64,
    pub sec_count_24h: f64,
    pub crypto_sentiment: f64,
    pub macro_sentiment: f64,
    pub tech_sentiment: f64,
    pub business_sentiment: f64,
    pub news_surprise: f64,
    pub topic_diversity: f64,
}

#[derive(Clone, Default)]
pub struct RssArticle {
    pub title: String,
    pub url: String,
    pub source: String,
    pub fetched_at: chrono::DateTime<chrono::Utc>,
    #[allow(dead_code)]
    pub sentiment: f64,
}

#[derive(Clone, Default)]
pub struct MarketSummaryData {
    pub advance_decline: f64,
    pub new_highs_lows: f64,
    pub spy_vol_5d: f64,
    pub qqq_vol_5d: f64,
    pub iwm_vol_5d: f64,
    pub cross_asset_corr: f64,
    pub trending_count: f64,
    pub trending_portfolio_overlap: f64,
    pub gainers_momentum: f64,
    pub actives_vol_ratio: f64,
    #[allow(dead_code)]
    pub market_hours_state: f64,
    #[allow(dead_code)]
    pub minutes_since_event: f64,
}

// ── DataStore ─────────────────────────────────────────────────────────────────

pub struct DataStore {
    // Per-asset (concurrent)
    pub prices: Arc<DashMap<String, PriceBuffer>>,
    pub forming: Arc<DashMap<String, FormingBar>>,
    pub options: Arc<DashMap<String, CachedValue<OptionsData>>>,
    pub fundamentals: Arc<DashMap<String, CachedValue<FundamentalData>>>,
    pub risk: Arc<DashMap<String, CachedValue<RiskData>>>,
    pub insider_tx: Arc<DashMap<String, CachedValue<Vec<OurInsiderTx>>>>,
    pub analyst: Arc<DashMap<String, CachedValue<AnalystData>>>,
    #[allow(dead_code)]
    pub sec_filings: Arc<DashMap<String, CachedValue<Vec<OurSecFiling>>>>,
    #[allow(dead_code)]
    pub splits: Arc<DashMap<String, CachedValue<Vec<Split>>>>,
    #[allow(dead_code)]
    pub calendar: Arc<DashMap<String, CachedValue<OurCalendar>>>,
    pub event_state: Arc<DashMap<String, EventState>>,

    // Market-wide singletons
    pub world_indices: Arc<RwLock<CachedValue<WorldIndices>>>,
    pub crypto: Arc<RwLock<CachedValue<CryptoMarket>>>,
    pub fear_greed: Arc<RwLock<CachedValue<FearGreedData>>>,
    pub sector: Arc<RwLock<CachedValue<SectorData>>>,
    pub yield_curve: Arc<RwLock<CachedValue<YieldCurveData>>>,
    pub fred_macro: Arc<RwLock<CachedValue<MacroData>>>,
    pub rss_feed: Arc<RwLock<CachedValue<RssFeedState>>>,
    pub market_summary: Arc<RwLock<CachedValue<MarketSummaryData>>>,

    pub rolling_stats: Arc<RollingStatsStore>,
    /// The tradable universe — user-controlled and mutable at runtime via the
    /// `portfolio_symbols.txt` hot-reload watcher (`scheduler::watch`). Insertion order is
    /// preserved; survivors keep their relative order across edits.
    pub symbols: Arc<RwLock<Vec<String>>>,
    /// Universe symbols that have warmed up enough to emit feature rows (a subset
    /// of `symbols`). Newly added symbols stay out until `scheduler::warmup`
    /// seeds their price history, so the assembler never emits cold-start garbage.
    pub ready: Arc<DashMap<String, ()>>,
    /// Monotonic generation counter bumped whenever the universe changes. The
    /// price stream subscribes to this and re-connects on every change. A `watch`
    /// channel (not `Notify`) so a change is never missed between stream events.
    universe_gen: tokio::sync::watch::Sender<u64>,

    // Snapshot-polled sets used by cross_asset::assemble (live mode)
    pub trending_symbols: Arc<RwLock<HashSet<String>>>,
    pub screener_membership: Arc<DashMap<String, f32>>,
    /// Per-symbol 24h decay-weighted news sentiment [-1.0, +1.0]. Updated every 5 min.
    pub news_sentiment: Arc<DashMap<String, f32>>,
    /// Per-symbol Block-8 text embedding (harrier-oss 1024-d, decay-pooled over the
    /// trailing news window then reduced to 64-d via the shared PCA basis — the
    /// SAME pipeline the historical collector uses). Updated every 5 min by the
    /// `text_embed` poller. Absent ⇒ assembler reads honest zeros.
    pub text_embedding: Arc<DashMap<String, [f32; 64]>>,
}

impl DataStore {
    /// Maximum universe size — the static cap matching the model's trained
    /// asset-axis dimension (`MeridianSpec.n_assets`). Adds beyond this are
    /// rejected. Growing it requires retraining + widening the portfolio vector.
    pub const MAX_ASSETS: usize = 100;

    pub fn new(symbols: Vec<String>, rolling_stats: Arc<RollingStatsStore>) -> Arc<Self> {
        let now = chrono::Utc::now();

        // The initial universe is treated as ready immediately (its rolling stats
        // are restored from disk / warm up live, same as before dynamic universe).
        let ready: DashMap<String, ()> = symbols.iter().map(|s| (s.clone(), ())).collect();

        Arc::new(Self {
            prices: Arc::new(DashMap::new()),
            forming: Arc::new(DashMap::new()),
            options: Arc::new(DashMap::new()),
            fundamentals: Arc::new(DashMap::new()),
            risk: Arc::new(DashMap::new()),
            insider_tx: Arc::new(DashMap::new()),
            analyst: Arc::new(DashMap::new()),
            sec_filings: Arc::new(DashMap::new()),
            splits: Arc::new(DashMap::new()),
            calendar: Arc::new(DashMap::new()),
            event_state: Arc::new(DashMap::new()),
            world_indices: Arc::new(RwLock::new(CachedValue {
                value: WorldIndices::default(),
                last_updated: now,
            })),
            crypto: Arc::new(RwLock::new(CachedValue {
                value: CryptoMarket::default(),
                last_updated: now,
            })),
            fear_greed: Arc::new(RwLock::new(CachedValue {
                value: FearGreedData::default(),
                last_updated: now,
            })),
            sector: Arc::new(RwLock::new(CachedValue {
                value: SectorData::default(),
                last_updated: now,
            })),
            yield_curve: Arc::new(RwLock::new(CachedValue {
                value: YieldCurveData::default(),
                last_updated: now,
            })),
            fred_macro: Arc::new(RwLock::new(CachedValue {
                value: MacroData::default(),
                last_updated: now,
            })),
            rss_feed: Arc::new(RwLock::new(CachedValue {
                value: RssFeedState::default(),
                last_updated: now,
            })),
            market_summary: Arc::new(RwLock::new(CachedValue {
                value: MarketSummaryData::default(),
                last_updated: now,
            })),
            rolling_stats,
            symbols: Arc::new(RwLock::new(symbols)),
            ready: Arc::new(ready),
            universe_gen: tokio::sync::watch::channel(0u64).0,
            trending_symbols: Arc::new(RwLock::new(HashSet::new())),
            screener_membership: Arc::new(DashMap::new()),
            news_sentiment: Arc::new(DashMap::new()),
            text_embedding: Arc::new(DashMap::new()),
        })
    }

    // ── Universe accessors (centralize lock access) ────────────────────────────

    /// Snapshot of the full universe (ready or not), in insertion order.
    pub fn symbols_snapshot(&self) -> Vec<String> {
        self.symbols.read().clone()
    }

    /// Number of symbols in the universe.
    pub fn symbols_len(&self) -> usize {
        self.symbols.read().len()
    }

    /// Whether `sym` is in the universe.
    pub fn symbols_contains(&self, sym: &str) -> bool {
        self.symbols.read().iter().any(|s| s == sym)
    }

    /// Subscribe to universe-change notifications (the price stream uses this to
    /// re-connect). The returned receiver yields on every [`bump_universe`].
    pub fn subscribe_universe(&self) -> tokio::sync::watch::Receiver<u64> {
        self.universe_gen.subscribe()
    }

    /// Signal that the universe changed (called by the watcher after applying a
    /// batch of adds/removes).
    pub fn bump_universe(&self) {
        self.universe_gen.send_modify(|g| *g += 1);
    }

    /// Whether `sym` has warmed up and may be emitted.
    pub fn is_ready(&self, sym: &str) -> bool {
        self.ready.contains_key(sym)
    }

    /// Mark `sym` as warmed up (called by `scheduler::warmup`).
    pub fn mark_ready(&self, sym: &str) {
        self.ready.insert(sym.to_string(), ());
    }

    /// Universe symbols that are warmed up, in universe order. The assembler emits
    /// exactly these — so the live active universe == this set.
    pub fn ready_snapshot(&self) -> Vec<String> {
        self.symbols
            .read()
            .iter()
            .filter(|s| self.ready.contains_key(*s))
            .cloned()
            .collect()
    }

    /// Add `sym` to the universe (not ready until warmed). Returns false if it is
    /// already present or the universe is at `MAX_ASSETS`.
    pub fn add_symbol(&self, sym: String) -> bool {
        let mut u = self.symbols.write();
        if u.iter().any(|s| *s == sym) {
            return false;
        }
        if u.len() >= Self::MAX_ASSETS {
            tracing::warn!(symbol = %sym, cap = Self::MAX_ASSETS, "universe at cap; ignoring add");
            return false;
        }
        u.push(sym);
        true
    }

    /// Remove `sym` from the universe and free all of its per-asset state.
    /// Returns false if it was not present.
    pub fn remove_symbol(&self, sym: &str) -> bool {
        let removed = {
            let mut u = self.symbols.write();
            match u.iter().position(|s| s == sym) {
                Some(i) => {
                    u.remove(i);
                    true
                }
                None => false,
            }
        };
        if removed {
            self.ready.remove(sym);
            self.prices.remove(sym);
            self.forming.remove(sym);
            self.options.remove(sym);
            self.fundamentals.remove(sym);
            self.risk.remove(sym);
            self.insider_tx.remove(sym);
            self.analyst.remove(sym);
            self.sec_filings.remove(sym);
            self.splits.remove(sym);
            self.calendar.remove(sym);
            self.event_state.remove(sym);
            self.news_sentiment.remove(sym);
            self.text_embedding.remove(sym);
            self.screener_membership.remove(sym);
        }
        removed
    }
}

/// Parse a `portfolio_symbols.txt` body into a capped, de-commented universe list.
///
/// Mirrors the Python loader (`line.split("#")[0].strip()`): strips inline and
/// full-line `#` comments, drops blanks, de-duplicates while preserving order,
/// and caps at [`DataStore::MAX_ASSETS`]. Shared by the binary loader and the
/// hot-reload watcher so they always agree on the universe.
pub fn parse_symbols(content: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    content
        .lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|s| !s.is_empty())
        .filter(|s| seen.insert(s.to_string()))
        .take(DataStore::MAX_ASSETS)
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod universe_tests {
    use super::*;

    fn store(syms: &[&str]) -> Arc<DataStore> {
        DataStore::new(
            syms.iter().map(|s| s.to_string()).collect(),
            RollingStatsStore::new(),
        )
    }

    #[test]
    fn parse_strips_comments_dedups_and_caps() {
        let body = "AAPL  # apple\n# full-line comment\n\nMSFT\nAAPL\n  NVDA  \n";
        assert_eq!(parse_symbols(body), vec!["AAPL", "MSFT", "NVDA"]);

        let many = (0..150).map(|i| format!("S{i}")).collect::<Vec<_>>().join("\n");
        assert_eq!(parse_symbols(&many).len(), DataStore::MAX_ASSETS);
    }

    #[test]
    fn initial_universe_is_ready() {
        let s = store(&["AAPL", "MSFT"]);
        assert_eq!(s.symbols_len(), 2);
        assert_eq!(s.ready_snapshot(), vec!["AAPL", "MSFT"]);
        assert!(s.is_ready("AAPL"));
    }

    #[test]
    fn add_is_unready_until_marked_then_remove_frees_state() {
        let s = store(&["AAPL"]);
        assert!(s.add_symbol("MSFT".into()));
        assert!(!s.add_symbol("MSFT".into()), "duplicate add rejected");
        // In the universe but not yet emitted (warming up).
        assert!(s.symbols_contains("MSFT"));
        assert!(!s.is_ready("MSFT"));
        assert_eq!(s.ready_snapshot(), vec!["AAPL"]);

        s.mark_ready("MSFT");
        assert_eq!(s.ready_snapshot(), vec!["AAPL", "MSFT"]);

        // Seed some per-asset state, then ensure removal frees it.
        s.prices.insert("MSFT".into(), PriceBuffer::new(8));
        assert!(s.remove_symbol("MSFT"));
        assert!(!s.symbols_contains("MSFT"));
        assert!(!s.is_ready("MSFT"));
        assert!(!s.prices.contains_key("MSFT"));
        assert!(!s.remove_symbol("MSFT"), "removing absent symbol is a no-op");
    }

    #[test]
    fn add_rejected_at_cap() {
        let syms: Vec<String> = (0..DataStore::MAX_ASSETS).map(|i| format!("S{i}")).collect();
        let s = DataStore::new(syms, RollingStatsStore::new());
        assert_eq!(s.symbols_len(), DataStore::MAX_ASSETS);
        assert!(!s.add_symbol("OVERFLOW".into()));
        assert_eq!(s.symbols_len(), DataStore::MAX_ASSETS);
    }

    #[test]
    fn bump_universe_notifies_subscriber() {
        let s = store(&["AAPL"]);
        let mut rx = s.subscribe_universe();
        assert_eq!(*rx.borrow_and_update(), 0);
        s.bump_universe();
        assert!(rx.has_changed().unwrap());
        assert_eq!(*rx.borrow_and_update(), 1);
    }
}
