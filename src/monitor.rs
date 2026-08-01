//! The monitor cycle: sweep quotes, normalize against baselines, evaluate the
//! break-glass rules, journal everything. Alert discipline: at most one alert
//! per (session, level), escalation always allowed, and alert rows are only
//! written during a REGULAR session — a Saturday `once` run reports the level
//! but records nothing. Alerts are advisory until execution (§10 step 10)
//! exists; "de-risk toward the stress allocation" is the standing suggestion.

use crate::bars::{self, TodayBars};
use crate::config::BusConfig;
use crate::feed::{self, FeedContext};
use crate::history::{Baseline, VolumeProfile, fetch_baseline};
use crate::journal;
use crate::quotes::{self, QuoteSnap};
use crate::severity::{Components, Level, evaluate};
use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use finance_query::Ticker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolReading {
    pub symbol: String,
    pub price: f64,
    pub prev_close: Option<f64>,
    pub intraday_return: Option<f64>,
    pub components: Components,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub fetched_at: DateTime<Utc>,
    pub session_date: NaiveDate,
    pub market_state: Option<String>,
    pub level: Level,
    pub drivers: Vec<String>,
    /// Market-level reading (the trigger symbol) first, legs after.
    pub readings: Vec<SymbolReading>,
    pub feed: Option<FeedContext>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRow {
    pub fetched_at: DateTime<Utc>,
    pub session_date: NaiveDate,
    pub level: Level,
    pub drivers: Vec<String>,
    pub market: SymbolReading,
    pub feed: Option<FeedContext>,
    pub suggested_action: String,
}

pub struct Monitor {
    cfg: BusConfig,
    baselines: HashMap<String, Baseline>,
    /// One authenticated upstream client per symbol, rebuilt lazily after a
    /// failure — per-request auth would mean thousands of logins per session.
    tickers: HashMap<String, Ticker>,
    profile: Option<VolumeProfile>,
    feed_cache: Option<FeedContext>,
    /// Highest level already alerted this session (restored from the journal
    /// on startup so a restart cannot double-alert).
    alerted: Level,
    alerted_date: Option<NaiveDate>,
}

impl Monitor {
    pub async fn bootstrap(cfg: BusConfig) -> Result<Self> {
        let mut m = Self {
            cfg,
            baselines: HashMap::new(),
            tickers: HashMap::new(),
            profile: None,
            feed_cache: None,
            alerted: Level::None,
            alerted_date: None,
        };
        for sym in m.cfg.symbols.clone() {
            let ticker = m.ticker(&sym).await?;
            let b = fetch_baseline(&sym, ticker).await?;
            tracing::info!(
                symbol = %sym,
                sigma_pct = format!("{:.2}", b.sigma * 100.0),
                prev_close = b.prev_close,
                "baseline built"
            );
            m.baselines.insert(sym, b);
        }
        let market = m.cfg.market_symbol().to_string();
        match VolumeProfile::fetch(&market, m.ticker(&market).await?).await {
            Ok(p) => m.profile = Some(p),
            Err(e) => {
                tracing::warn!(err = %e, "volume profile unavailable — vol_ratio disabled");
            }
        }
        m.restore_alert_state()?;
        Ok(m)
    }

    async fn ticker(&mut self, symbol: &str) -> Result<&Ticker> {
        if !self.tickers.contains_key(symbol) {
            let t = Ticker::new(symbol)
                .await
                .with_context(|| format!("building client for {symbol}"))?;
            self.tickers.insert(symbol.to_string(), t);
        }
        Ok(self.tickers.get(symbol).unwrap())
    }

    fn restore_alert_state(&mut self) -> Result<()> {
        let rows: Vec<AlertRow> = journal::read_jsonl(&self.cfg.alerts_path)?;
        if let Some(last) = rows.last() {
            self.alerted_date = Some(last.session_date);
            self.alerted = rows
                .iter()
                .filter(|r| r.session_date == last.session_date)
                .map(|r| r.level)
                .max()
                .unwrap_or(Level::None);
        }
        Ok(())
    }

    pub async fn refresh_feed(&mut self) {
        match feed::poll_8k(&self.cfg.feed_base_url).await {
            Ok(ctx) => self.feed_cache = Some(ctx),
            Err(e) => tracing::warn!(err = %e, "8-K feed poll failed"),
        }
    }

    /// Rebuild baselines when a new session's close has landed upstream.
    pub async fn refresh_baselines_if_stale(&mut self, session_date: NaiveDate) -> Result<()> {
        let stale = self
            .baselines
            .get(self.cfg.market_symbol())
            .is_none_or(|b| b.prev_date < session_date.pred_opt().unwrap_or(session_date));
        if !stale {
            return Ok(());
        }
        for sym in self.cfg.symbols.clone() {
            let refreshed = match self.ticker(&sym).await {
                Ok(t) => fetch_baseline(&sym, t).await,
                Err(e) => Err(e),
            };
            match refreshed {
                Ok(b) => {
                    self.baselines.insert(sym, b);
                }
                Err(e) => {
                    tracing::warn!(symbol = %sym, err = %e, "baseline refresh failed");
                    self.tickers.remove(&sym);
                }
            }
        }
        let market = self.cfg.market_symbol().to_string();
        if let Ok(t) = self.ticker(&market).await
            && let Ok(p) = VolumeProfile::fetch(&market, t).await
        {
            self.profile = Some(p);
        }
        Ok(())
    }

    pub async fn cycle(&mut self) -> Result<Snapshot> {
        let mut errors = Vec::new();

        let mut snaps: Vec<QuoteSnap> = Vec::new();
        for sym in self.cfg.symbols.clone() {
            let fetched = match self.ticker(&sym).await {
                Ok(t) => quotes::fetch_quote(&sym, t).await,
                Err(e) => Err(e),
            };
            match fetched {
                Ok(q) => snaps.push(q),
                Err(e) => {
                    errors.push(format!("{sym}: {e:#}"));
                    // Drop the client so the next cycle rebuilds auth from scratch.
                    self.tickers.remove(&sym);
                }
            }
        }
        let market = snaps
            .iter()
            .find(|q| q.symbol == self.cfg.market_symbol())
            .cloned()
            .context("market symbol quote missing — cannot evaluate severity")?;

        let today = match self.ticker(&market.symbol).await {
            Ok(t) => match bars::fetch_today(&market.symbol, t).await {
                Ok(t) => Some(t),
                Err(e) => {
                    errors.push(format!("today-bars: {e:#}"));
                    None
                }
            },
            Err(e) => {
                errors.push(format!("today-bars: {e:#}"));
                None
            }
        };

        let session_date = market
            .market_time
            .and_then(|ts| DateTime::from_timestamp(ts, 0))
            .map(|t| t.date_naive())
            .unwrap_or_else(|| Utc::now().date_naive());

        let readings: Vec<SymbolReading> = snaps
            .iter()
            .map(|q| self.reading(q, today.as_ref()))
            .collect();
        let market_reading = readings
            .iter()
            .find(|r| r.symbol == market.symbol)
            .cloned()
            .unwrap();

        let (level, drivers) = evaluate(&market_reading.components, &self.cfg.thresholds);

        let snapshot = Snapshot {
            fetched_at: Utc::now(),
            session_date,
            market_state: market.market_state.clone(),
            level,
            drivers: drivers.clone(),
            readings,
            feed: self.feed_cache.clone(),
            errors,
        };

        journal::append_jsonl(
            &journal::monitor_path(&self.cfg.monitor_dir, snapshot.session_date),
            &snapshot,
        )?;
        journal::write_latest(&self.cfg.latest_path, &snapshot)?;
        self.maybe_alert(&snapshot, &market_reading)?;
        Ok(snapshot)
    }

    fn reading(&self, q: &QuoteSnap, today: Option<&TodayBars>) -> SymbolReading {
        let b = self.baselines.get(&q.symbol);
        // The daily chart's own last close beats the quote's prev_close field
        // when both exist and disagree (quote fields go stale around the open).
        let prev_close = b.map(|b| b.prev_close).or(q.prev_close);

        let intraday_return =
            prev_close.and_then(|pc| (pc > 0.0 && q.price > 0.0).then(|| (q.price / pc).ln()));

        let mut c = Components::default();
        if let (Some(b), Some(r)) = (b, intraday_return) {
            c.ret_z = Some(r.abs() / b.sigma);
            if let Some(open) = q.open.filter(|o| *o > 0.0) {
                c.gap_z = Some((open / b.prev_close).ln().abs() / b.sigma);
            }
        }
        // Volume projection and realized vol: trigger symbol only — the legs'
        // baselines exist, but only the market-level reading can fire.
        if q.symbol == self.cfg.market_symbol()
            && let (Some(b), Some(t), Some(profile), Some(vol)) =
                (b, today, self.profile.as_ref(), q.day_volume)
        {
            let elapsed = ((t.last_ts - t.open_ts) / 300 + 1).max(0) as usize;
            // First half-hour projections whipsaw; stay blind until then.
            if elapsed >= 6 {
                let share = profile.expected_share(elapsed);
                c.vol_ratio = Some(vol as f64 / share / b.avg_volume_20d);
            }
            c.rvol_ratio = t.realized_vol_day.map(|rv| rv / b.sigma);
        }

        SymbolReading {
            symbol: q.symbol.clone(),
            price: q.price,
            prev_close,
            intraday_return,
            components: c,
        }
    }

    fn maybe_alert(&mut self, snap: &Snapshot, market: &SymbolReading) -> Result<()> {
        let in_session = snap.market_state.as_deref() == Some("REGULAR");
        if snap.level == Level::None || !in_session {
            return Ok(());
        }
        if self.alerted_date == Some(snap.session_date) && self.alerted >= snap.level {
            return Ok(());
        }
        let row = AlertRow {
            fetched_at: snap.fetched_at,
            session_date: snap.session_date,
            level: snap.level,
            drivers: snap.drivers.clone(),
            market: market.clone(),
            feed: snap.feed.clone(),
            suggested_action: "de_risk_toward_stress_allocation".to_string(),
        };
        journal::append_jsonl(&self.cfg.alerts_path, &row)?;
        tracing::warn!(
            level = %snap.level,
            drivers = ?snap.drivers,
            "BREAK-GLASS journal: alert recorded"
        );
        self.alerted_date = Some(snap.session_date);
        self.alerted = snap.level;
        Ok(())
    }
}

pub async fn run_once(cfg: BusConfig) -> Result<()> {
    let mut m = Monitor::bootstrap(cfg).await?;
    m.refresh_feed().await;
    let snap = m.cycle().await?;
    println!("{}", serde_json::to_string_pretty(&snap)?);
    Ok(())
}

pub async fn run_live(cfg: BusConfig) -> Result<()> {
    let mut m = Monitor::bootstrap(cfg.clone()).await?;
    let mut last_feed = std::time::Instant::now() - cfg.feed_cycle;
    loop {
        let started = std::time::Instant::now();
        if last_feed.elapsed() >= cfg.feed_cycle {
            m.refresh_feed().await;
            last_feed = std::time::Instant::now();
        }
        let state = match m.cycle().await {
            Ok(snap) => {
                if let Err(e) = m.refresh_baselines_if_stale(snap.session_date).await {
                    tracing::warn!(err = %e, "baseline refresh failed");
                }
                tracing::info!(
                    state = snap.market_state.as_deref().unwrap_or("?"),
                    level = %snap.level,
                    ret_z = snap.readings[0].components.ret_z.map(|z| format!("{z:.2}")),
                    "cycle complete"
                );
                snap.market_state
            }
            Err(e) => {
                tracing::error!(err = %e, "cycle failed");
                None
            }
        };
        let cadence = if state.as_deref() == Some("REGULAR") {
            cfg.cycle
        } else {
            cfg.idle_cycle
        };
        let sleep = cadence.saturating_sub(started.elapsed());
        tokio::select! {
            _ = tokio::time::sleep(sleep) => {}
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("shutdown signal — exiting cleanly");
                return Ok(());
            }
        }
    }
}

pub fn run_status(cfg: BusConfig) -> Result<()> {
    let alerts: Vec<AlertRow> = journal::read_jsonl(&cfg.alerts_path)?;
    println!("break-glass alerts: {} at {}", alerts.len(), cfg.alerts_path.display());
    for a in alerts.iter().rev().take(5) {
        println!(
            "  {} {} {}  [{}]",
            a.session_date,
            a.level,
            a.market.price,
            a.drivers.join("; ")
        );
    }
    match std::fs::read_to_string(&cfg.latest_path) {
        Ok(s) => {
            let snap: Snapshot = serde_json::from_str(&s)?;
            println!(
                "latest snapshot: {} state={} level={} ({} readings, {} errors)",
                snap.fetched_at,
                snap.market_state.as_deref().unwrap_or("?"),
                snap.level,
                snap.readings.len(),
                snap.errors.len()
            );
        }
        Err(_) => println!("no latest snapshot at {}", cfg.latest_path.display()),
    }
    // Journals are partitioned by session date, so report the newest one —
    // "today's" file does not exist on weekends and holidays.
    let newest = std::fs::read_dir(&cfg.monitor_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("monitor-")))
        .max();
    match newest {
        Some(path) => {
            let cycles = std::fs::read_to_string(&path)
                .map(|s| s.lines().count())
                .unwrap_or(0);
            println!("latest monitor journal: {cycles} cycles ({})", path.display());
        }
        None => println!("no monitor journals in {}", cfg.monitor_dir.display()),
    }
    Ok(())
}
