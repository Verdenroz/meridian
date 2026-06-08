use crate::args::{HistoricalArgs, HistoricalInterval};
use crate::features::{FeatureRow, classify_asset, sector_map::sector_etf_for};
use crate::historical::edgar_text::EdgarFilingDoc;
use crate::historical::embedder::{EMBED_DIM, Embedder, EmbedderConfig};
use crate::historical::pca::PcaModel;
use crate::historical::risk::RISK_WINDOW;
use crate::historical::text_pca;
use crate::output::emit_row;
use crate::store::{Bar, FundamentalData, PriceBuffer, RollingStatsStore};
use chrono::{Datelike, Timelike};
use finance_query::{Frequency, Interval, Screener, StatementType, Ticker, TimeRange, finance};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Trailing window for Block 8 text-event aggregation, in seconds.
const TEXT_WINDOW_SECS: i64 = 90 * 86_400;
/// Time-decay constant (days) used to weight events within the window.
const TEXT_DECAY_DAYS: f32 = 30.0;

/// Trailing window of `bars` ending at index `i` (inclusive), max length n.
fn trailing_window(bars: &[Bar], i: usize, n: usize) -> Vec<Bar> {
    let end = i + 1;
    let start = end.saturating_sub(n);
    bars[start..end].to_vec()
}

/// Trailing window of `bars` ending at the most recent bar with timestamp ≤
/// `ts`, max length n. Used to align SPY/sector benchmark windows to the
/// asset's per-bar timestamp (calendars don't always match).
fn trailing_window_at_or_before(bars: &[Bar], ts: i64, n: usize) -> Vec<Bar> {
    if bars.is_empty() {
        return Vec::new();
    }
    // partition_point returns the first index where bar.timestamp > ts;
    // the prefix [..idx] is the inclusive-of-ts window.
    let idx = bars.partition_point(|b| b.timestamp <= ts);
    let end = idx;
    let start = end.saturating_sub(n);
    bars[start..end].to_vec()
}

fn price_volume_return_horizon(interval: &HistoricalInterval) -> usize {
    match interval {
        HistoricalInterval::OneMinute => 390,
        HistoricalInterval::FiveMinutes => 78,
        HistoricalInterval::OneHour => 24,
        HistoricalInterval::OneDay => 5,
    }
}

fn trailing_log_return_at(bars: &[Bar], i: usize, horizon: usize) -> f32 {
    if i == 0 {
        return 0.0;
    }
    let end = i;
    let start = end.saturating_sub(horizon);
    if start == end {
        return 0.0;
    }
    let p0 = bars[start].close;
    let p1 = bars[end].close;
    if p0 > 0.0 && p1 > 0.0 {
        (p1 / p0).ln() as f32
    } else {
        0.0
    }
}

fn own_volume_zscore(asset_window: &[Bar], lookback: usize) -> f32 {
    if asset_window.len() < 3 {
        return 0.0;
    }
    let take = lookback.min(asset_window.len().saturating_sub(1));
    if take < 2 {
        return 0.0;
    }
    let hist = &asset_window[asset_window.len() - 1 - take..asset_window.len() - 1];
    let mean = hist.iter().map(|b| b.volume as f32).sum::<f32>() / hist.len() as f32;
    let var = hist
        .iter()
        .map(|b| {
            let x = b.volume as f32 - mean;
            x * x
        })
        .sum::<f32>()
        / hist.len() as f32;
    let std = var.sqrt().max(1e-6);
    let z = (asset_window.last().map(|b| b.volume as f32).unwrap_or(mean) - mean) / std;
    (0.5 + 0.25 * z).clamp(0.0, 1.0)
}

fn positive_return_ratio(asset_window: &[Bar], lookback: usize) -> f32 {
    if asset_window.len() < 3 {
        return 0.0;
    }
    let take = (lookback + 1).min(asset_window.len());
    let slice = &asset_window[asset_window.len() - take..];
    let mut pos = 0usize;
    let mut n = 0usize;
    for w in slice.windows(2) {
        if w[0].close > 0.0 && w[1].close > 0.0 {
            if (w[1].close / w[0].close).ln() > 0.0 {
                pos += 1;
            }
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { pos as f32 / n as f32 }
}

/// Historical batch collection mode.
///
/// Downloads chart data for a single symbol and emits one JSONL row per bar.
/// All feature blocks are computed from historical finance-query data:
///   - Price/Volume + Technical: from OHLCV chart bars
///   - Options: proxy computed from price volatility + snapshot metrics
///   - Fundamentals: reconstructed from snapshot APIs with event-timeline backfill
///   - Risk Analytics: computed from trailing OHLCV vs SPY benchmark
///   - Event State: dividend/split signals from chart events + SEC/grading timelines
///   - Cross-Asset: computed from asset + SPY + sector ETF chart series
///   - Text Embedding: pooled from finance-query news/transcript corpus
///   - Market-Wide: real historical series for VIX, yield curve, FRED macro,
///     sector ETF returns, major world indices, and crypto
///
/// Yahoo Finance availability per interval:
///   1m  — last 29 days  (chunking handled internally by chart_range)
///   5m  — last 58 days  (chunking handled internally by chart_range)
///   1h  — last ~2 years (chart_range with 730-day default window)
///   1d  — 10-year history (chart(TenYears) when no start given; Max downsamples to monthly)
pub async fn run(args: HistoricalArgs) -> anyhow::Result<()> {
    let symbol = &args.symbol;
    let ticker = Ticker::new(symbol).await?;

    let today = chrono::Utc::now().date_naive();
    let end_date = args.end.unwrap_or(today);
    let end_ts = end_date
        .and_hms_opt(23, 59, 59)
        .unwrap()
        .and_utc()
        .timestamp();

    let all_bars = fetch_historical_bars(&ticker, &args, end_ts).await?;
    let market_lookup = build_market_feature_lookup(&args, end_ts).await;

    let total = all_bars.len();
    if total == 0 {
        anyhow::bail!("no candle data returned for {symbol}");
    }

    // ── Hoist SPY benchmark (Block 5 risk + Block 7 cross-asset) ─────────────
    // Spec §6.2 Block 5 line 548 prescribes risk vs SPY; spec §6.2 Block 7
    // line 582-585 prescribes RS vs SPY and rolling correlation. Skip the
    // self-fetch when the symbol IS SPY.
    let spy_bars: Vec<Bar> = if symbol == "SPY" {
        all_bars.clone()
    } else {
        match Ticker::new("SPY").await {
            Ok(spy) => fetch_historical_bars(&spy, &args, end_ts)
                .await
                .unwrap_or_default(),
            Err(err) => {
                tracing::warn!(err = %err, "SPY benchmark fetch failed; risk + cross-asset will use empty SPY series");
                Vec::new()
            }
        }
    };
    // Sector ETF for Block 7 sector-relative-strength. We use the GICS-mapped
    // sector ETF from the same finance-query path; resolution happens in
    // `sector_etf_for`. Falls back to empty (RS_sector → 0.0) if absent.
    let sector_bars: Vec<Bar> = if let Some(sector_sym) = sector_etf_for(symbol) {
        if sector_sym == symbol {
            all_bars.clone()
        } else {
            match Ticker::new(sector_sym).await {
                Ok(t) => fetch_historical_bars(&t, &args, end_ts).await.unwrap_or_default(),
                Err(err) => {
                    tracing::warn!(symbol = %sector_sym, err = %err, "sector ETF fetch failed");
                    Vec::new()
                }
            }
        }
    } else {
        Vec::new()
    };

    // Fetch chart events (dividends, splits) — already cached from the chart request
    let dividends = ticker
        .dividends(TimeRange::TenYears)
        .await
        .unwrap_or_default();
    let splits = ticker.splits(TimeRange::TenYears).await.unwrap_or_default();

    // ── Fundamentals / events / options / text sources ───────────────────────
    let summary = ticker.summary_detail().await.ok().flatten();
    let financial = ticker.financial_data().await.ok().flatten();
    let key_stats = ticker.key_stats().await.ok().flatten();
    let profile = ticker.asset_profile().await.ok().flatten();
    let recommendation_trend = ticker.recommendation_trend().await.ok().flatten();
    let earnings_trend = ticker.earnings_trend().await.ok().flatten();
    let earnings_history = ticker.earnings_history().await.ok().flatten();
    let grading_history = ticker.grading_history().await.ok().flatten();
    let insider_transactions = ticker.insider_transactions().await.ok().flatten();
    let institution_ownership = ticker.institution_ownership().await.ok().flatten();
    let major_holders = ticker.major_holders().await.ok().flatten();
    let calendar_events = ticker.calendar_events().await.ok().flatten();
    let options_snapshot = ticker.options(None).await.ok();
    let income_quarterly_json = ticker
        .financials(StatementType::Income, Frequency::Quarterly)
        .await
        .ok()
        .and_then(|v| serde_json::to_value(v).ok());
    let balance_quarterly_json = ticker
        .financials(StatementType::Balance, Frequency::Quarterly)
        .await
        .ok()
        .and_then(|v| serde_json::to_value(v).ok());
    let cashflow_quarterly_json = ticker
        .financials(StatementType::CashFlow, Frequency::Quarterly)
        .await
        .ok()
        .and_then(|v| serde_json::to_value(v).ok());
    
    // Debug: log raw JSON keys from income statement
    if let Some(v) = &income_quarterly_json {
        if let Some(stmt) = json_at(v, &["statement"]) {
            if let Some(obj) = stmt.as_object() {
                if let Some(revenue) = obj.get("TotalRevenue").or_else(|| obj.get("totalRevenue")) {
                    if let Some(rev_obj) = revenue.as_object() {
                        let date_keys: Vec<_> = rev_obj.keys().take(5).cloned().collect();
                        tracing::info!(
                            symbol = %symbol,
                            revenue_dates = ?date_keys,
                            "raw quarterly revenue date keys"
                        );
                    }
                }
            }
        }
    }
    let _dividend_analytics = ticker.dividend_analytics(TimeRange::TenYears).await.ok();
    let recommendations = ticker.recommendations(5).await.ok();
    let edgar_submissions = ticker.edgar_submissions().await.ok();

    // Block 7 slots [4..7] are filled from finance-query snapshots where
    // Yahoo does not expose historical traces.
    let trending_symbols = finance::trending(Some(finance_query::Region::UnitedStates))
        .await
        .unwrap_or_default();
    let trending_set: HashSet<String> = trending_symbols.into_iter().map(|t| t.symbol).collect();
    let is_trending = if trending_set.contains(symbol) { 1.0 } else { 0.0 };

    let mut screener_membership_count = 0.0f32;
    for &scr in Screener::all() {
        if let Ok(res) = finance::screener(scr, 25).await
            && res.quotes.iter().any(|q| q.symbol == *symbol)
        {
            screener_membership_count += 1.0;
        }
    }
    let screener_membership = (screener_membership_count / Screener::all().len() as f32).clamp(0.0, 1.0);
    let recommendation_overlap = recommendations
        .as_ref()
        .map(|r| (r.count().min(5) as f32) / 5.0)
        .unwrap_or(0.0);

    let summary_json = summary.as_ref().and_then(|v| serde_json::to_value(v).ok());
    let financial_json = financial.as_ref().and_then(|v| serde_json::to_value(v).ok());
    let key_stats_json = key_stats.as_ref().and_then(|v| serde_json::to_value(v).ok());
    
    // Debug: log JSON keys
    if let Some(v) = &summary_json {
        if let Some(obj) = v.as_object() {
            let keys: Vec<_> = obj.keys().take(15).cloned().collect();
            tracing::info!(
                symbol = %symbol,
                keys = ?keys,
                "summary_detail JSON keys"
            );
        }
    }
    if let Some(v) = &key_stats_json {
        if let Some(obj) = v.as_object() {
            let keys: Vec<_> = obj.keys().take(15).cloned().collect();
            tracing::info!(
                symbol = %symbol,
                keys = ?keys,
                "key_stats JSON keys"
            );
        }
    }
    let profile_json = profile.as_ref().and_then(|v| serde_json::to_value(v).ok());
    let recommendation_trend_json = recommendation_trend
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let earnings_trend_json = earnings_trend
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let earnings_history_json = earnings_history
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let grading_history_json = grading_history
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let insider_transactions_json = insider_transactions
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let institution_ownership_json = institution_ownership
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let major_holders_json = major_holders
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    let calendar_events_json = calendar_events
        .as_ref()
        .and_then(|v| serde_json::to_value(v).ok());
    // Block 8 text floor: real EDGAR primary-document text, dated by filing_date.
    // Yahoo has no historical news archive, so filings (identical source in the
    // live and historical regimes) are the aligned backbone. FNSPID news is merged
    // in via the Python ingestion layer; live inference adds the `feeds` stream.
    // The previous static "{symbol} {sector} {industry}" identity event is gone —
    // it had collapsed Block 8 into a constant per-asset identity fingerprint.
    let (filing_docs, filing_texts) = fetch_filing_texts(&edgar_submissions).await;
    // Merge the dated FNSPID news corpus (historical analogue of the live `feeds`
    // RSS stream) onto the EDGAR filing-text floor. Both sources flow through the
    // SAME FinBERT scorer + harrier-oss embedder below, so only the text *source*
    // differs across train/inference — the representation pipeline is identical
    // (DATA_PIPELINE_ISSUES.md C1/H1). A missing news cache ⇒ filings-only, never
    // a fabricated fill.
    let edgar_count = filing_texts.len();
    let fnspid_news = crate::historical::fnspid::load_news(symbol);
    let fnspid_count = fnspid_news.len();
    let mut combined_texts = filing_texts;
    combined_texts.extend(fnspid_news);
    combined_texts.sort_by_key(|(ts, _)| *ts);
    tracing::info!(
        symbol = %symbol, edgar = edgar_count, fnspid = fnspid_count,
        "Block 8 text sources merged (EDGAR filings ∪ FNSPID news)"
    );
    let scored_texts = score_texts(&combined_texts).await;
    let text_events = build_text_events(&scored_texts);
    let sec_filing_events = build_sec_filing_timeline(&filing_docs);
    let grade_events = extract_grade_events(&grading_history_json);
    let insider_events = extract_insider_events(&insider_transactions_json);
    let earnings_events = extract_earnings_events(&earnings_history_json);
    let institution_events = extract_institution_events(&institution_ownership_json);
    let options_snapshot_metrics = extract_options_snapshot_metrics(options_snapshot.as_ref());
    let fundamental_static = build_fundamental_static(
        symbol,
        &summary_json,
        &financial_json,
        &key_stats_json,
        &profile_json,
        &recommendation_trend_json,
        &earnings_trend_json,
        &calendar_events_json,
        &major_holders_json,
    );
    
    tracing::info!(
        symbol = %symbol,
        trailing_pe = ?fundamental_static.trailing_pe,
        forward_pe = ?fundamental_static.forward_pe,
        peg_ratio = ?fundamental_static.peg_ratio,
        profit_margins = ?fundamental_static.profit_margins,
        earnings_growth = ?fundamental_static.earnings_growth,
        debt_to_equity = ?fundamental_static.debt_to_equity,
        "fundamental_static snapshot values"
    );
    let quarterly_fundamental_series = build_quarterly_fundamental_series(
        &income_quarterly_json,
        &balance_quarterly_json,
        &cashflow_quarterly_json,
        fundamental_static.market_cap,
        all_bars.first().map(|b| b.close),
    );
    
    // Log detailed quarterly data timestamps
    if !quarterly_fundamental_series.revenue.is_empty() {
        let mut dates_str = String::new();
        for ts in quarterly_fundamental_series.revenue.keys() {
            if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp(*ts, 0) {
                dates_str.push_str(&dt.format("%Y-%m-%d, ").to_string());
            } else {
                dates_str.push_str(&format!("ts={}, ", ts));
            }
        }
        tracing::info!(
            symbol = %symbol,
            dates = %dates_str,
            "quarterly revenue dates"
        );
    }
    
    tracing::info!(
        symbol = %symbol,
        rev = quarterly_fundamental_series.revenue.len(),
        ni = quarterly_fundamental_series.net_income.len(),
        ebitda = quarterly_fundamental_series.ebitda.len(),
        eps = quarterly_fundamental_series.eps_diluted.len(),
        debt = quarterly_fundamental_series.total_debt.len(),
        equity = quarterly_fundamental_series.total_equity.len(),
        cash = quarterly_fundamental_series.cash_and_equivalents.len(),
        ca = quarterly_fundamental_series.current_assets.len(),
        cl = quarterly_fundamental_series.current_liabilities.len(),
        ocf = quarterly_fundamental_series.operating_cash_flow.len(),
        capex = quarterly_fundamental_series.capital_expenditure.len(),
        shares = quarterly_fundamental_series.shares_outstanding.len(),
        "quarterly fundamental series coverage"
    );

    // ── Block 8: pre-compute embeddings for every unique snippet text ─────
    //
    // Single batched call to llama-swap covers the full symbol's event corpus.
    // Per-bar pooling later is a pure synchronous lookup against this map.
    let embedder = Embedder::new(EmbedderConfig::default())?;
    let mut unique_texts: HashSet<String> = HashSet::new();
    for ev in &text_events {
        unique_texts.insert(ev.text.clone());
    }
    let unique_vec: Vec<String> = unique_texts.into_iter().collect();
    tracing::info!(
        symbol = %symbol, snippets = unique_vec.len(), "embedding text events"
    );
    let embeddings = embedder.embed_many(&unique_vec).await?;
    let mut embed_map: HashMap<String, Vec<f32>> = HashMap::with_capacity(unique_vec.len());
    for (text, emb) in unique_vec.iter().zip(embeddings.into_iter()) {
        embed_map.insert(text.clone(), emb);
    }

    // Block 8 reduction basis: harrier-oss is not Matryoshka, so we project the
    // pooled 1024-d embedding down to the 64-d block via a fixed PCA basis. The
    // basis is fit ONCE over the whole universe by the `fit-text-pca` pre-pass
    // (`make collect` runs it before per-symbol collection); here we only LOAD
    // it, so every asset shares comparable axes. If it is missing we fall back to
    // zeros rather than fitting a per-symbol (incompatible) basis.
    let text_pca = text_pca::load_cached().unwrap_or_else(|| {
        tracing::warn!(
            "block8_text_pca basis not found — run the fit-text-pca pre-pass \
             (`make collect` runs it after `make fnspid`); Block 8 will be zeros this run"
        );
        PcaModel::zeros(EMBED_DIM, text_pca::TEXT_PCA_DIM)
    });

    // Pre-sort event timestamps for binary search
    let div_timestamps: Vec<i64> = dividends.iter().map(|d| d.timestamp).collect();
    let div_amounts: Vec<f64> = dividends.iter().map(|d| d.amount).collect();

    tracing::info!(
        symbol = %symbol, bars = total, dividends = dividends.len(),
        splits = splits.len(), interval = ?args.interval, "emitting JSONL rows"
    );

    let rolling_stats = RollingStatsStore::new();
    let seconds_per_day: f64 = 86_400.0;
    let div_decay_tau: f64 = 14.0; // 14-day decay for dividend proximity

    for i in 0..total {
        let buf = PriceBuffer::prefix(&all_bars, i);
        let bar = &all_bars[i];
        let bar_ts = bar.timestamp;

        let timestamp = chrono::DateTime::from_timestamp(bar_ts, 0).unwrap_or_default();

        // Compute dividend proximity: exp(-days_since / tau)
        // Find most recent dividend at or before this bar
        let exdiv_proximity = match div_timestamps.partition_point(|&t| t <= bar_ts) {
            0 => 0.0f32, // no dividend before this bar
            idx => {
                let days_since = (bar_ts - div_timestamps[idx - 1]) as f64 / seconds_per_day;
                let amount = div_amounts[idx - 1];
                // Scale by normalized dividend amount (typical ~0.2-2.0, log-compress)
                let mag = (1.0 + amount).ln().min(2.0);
                (mag * (-days_since / div_decay_tau).exp()) as f32
            }
        };

        // Build event_state (Block 6) — canonical layout shared with live
        // inference; ex-dividend proximity fills slot [9].
        let event_state = compute_event_state_at(
            bar_ts,
            exdiv_proximity,
            &insider_events,
            &grade_events,
            &sec_filing_events,
            &earnings_events,
            fundamental_static.next_earnings_ts,
            &text_events,
        );

        // ── Block 5: Risk Analytics ──────────────────────────────────────────
        // Take a trailing 252-day window of the asset and align SPY to the
        // same time range so beta is computed over equivalent observations.
        let asset_window: Vec<Bar> = trailing_window(&all_bars, i, RISK_WINDOW);
        let spy_window: Vec<Bar> = trailing_window_at_or_before(
            &spy_bars,
            bar_ts,
            RISK_WINDOW,
        );
        let risk = crate::historical::risk::compute_risk_from_bars(
            &asset_window,
            &spy_window,
        );

        // ── Block 7: Cross-Asset Context ─────────────────────────────────────
        // 60-day SPY correlation needs ≥ 60+1 bars; corr_change_20d needs an
        // extra 20-day offset on top, so window of 90 leaves headroom.
        let asset_xa: Vec<Bar> = trailing_window(&all_bars, i, 90);
        let spy_xa: Vec<Bar> = trailing_window_at_or_before(&spy_bars, bar_ts, 90);
        let sector_xa: Vec<Bar> =
            trailing_window_at_or_before(&sector_bars, bar_ts, 90);
        let div_amounts_so_far: Vec<f64> = div_amounts
            .iter()
            .zip(div_timestamps.iter())
            .filter(|&(_, t)| *t <= bar_ts)
            .map(|(amount, _)| *amount)
            .collect();
        let cross_asset_block = crate::historical::cross_asset::compute_cross_asset_at(
            &asset_xa,
            &spy_xa,
            &sector_xa,
            &div_amounts_so_far,
        );
        let mut cross_asset_block = cross_asset_block;
        let trend_proxy = own_volume_zscore(&asset_xa, 20);
        let screener_proxy = positive_return_ratio(&asset_xa, 20);
        let reco_proxy = (0.7 * cross_asset_block[2].abs() + 0.3 * cross_asset_block[3].abs())
            .clamp(0.0, 1.0);
        cross_asset_block[4] = (0.6 * trend_proxy + 0.4 * is_trending).clamp(0.0, 1.0);
        cross_asset_block[5] = (0.6 * screener_proxy + 0.4 * screener_membership).clamp(0.0, 1.0);
        cross_asset_block[6] = (0.6 * reco_proxy + 0.4 * recommendation_overlap).clamp(0.0, 1.0);

        // ── Block 3: Options proxy timeline (no hard-zero gaps) ───────────────
        let options_block = compute_options_proxy_at(
            &asset_xa,
            &options_snapshot_metrics,
            event_state[8], // earnings proximity
        );

        // ── Block 4: Fundamental timeline reconstruction ───────────────────────
        let fd = reconstruct_fundamentals_at(
            bar_ts,
            &fundamental_static,
            &earnings_events,
            &grade_events,
            &insider_events,
            &institution_events,
        );
        // TODO: quarterly_fundamental_series currently returns future dates (forward-looking data)
        // not historical time-series. Finance-query doesn't provide truly historical quarterly statements
        // archived at release date. Skipping for now and using snapshot-based fundamental data instead.
        // apply_quarterly_fundamentals_at(
        //     &mut fd,
        //     bar_ts,
        //     bar.close,
        //     &asset_window,
        //     &quarterly_fundamental_series,
        //     &div_timestamps,
        //     &div_amounts,
        // );
        let fundamental_block = encode_fundamental_block(&fd);

        // ── Block 8: Text embedding timeline (EDGAR filings ∪ FNSPID) ──────────
        // Sentiment (event_state[7]) is the FinBERT score carried on each text
        // event, pooled inside compute_event_state_at — not re-derived here.
        let text_block = compute_text_embedding_at(bar_ts, &text_events, &embed_map, &text_pca);

        let mut price_volume_block = crate::features::price_volume::assemble(
            Some(&buf),
            &rolling_stats,
            symbol,
            &args.interval,
        );
        price_volume_block[9] = trailing_log_return_at(
            &all_bars,
            i,
            price_volume_return_horizon(&args.interval),
        );
        
        let technical_block = crate::features::technical::assemble(Some(&buf), &rolling_stats, symbol, &args.interval);

        let row = FeatureRow {
            timestamp,
            symbol: symbol.clone(),
            asset_class: classify_asset(symbol),
            price_volume: price_volume_block,
            technical: technical_block,
            options: options_block,
            fundamental: fundamental_block,
            risk_analytics: crate::features::risk_block::assemble_from_struct(&risk),
            cross_asset_context: cross_asset_block,
            event_state,
            text: text_block,
            market: market_lookup.vector_at(bar_ts),
            ..FeatureRow::default()
        };

        emit_row(&row);

        if i > 0 && i % 10_000 == 0 {
            tracing::info!(symbol = %symbol, progress = i, total, "historical progress");
        }
    }

    tracing::info!(symbol = %symbol, rows = total, "historical collection complete");
    Ok(())
}

#[derive(Default)]
struct MarketFeatureLookup {
    returns: HashMap<&'static str, BTreeMap<i64, f32>>,
    closes: HashMap<&'static str, BTreeMap<i64, f64>>,
    /// Spec §6.3 sub-blocks: Yield Curve [12..24] + FRED Macro [24..36].
    /// Optional so a stale or partial fetch doesn't break collection.
    macro_lookup: Option<crate::historical::macro_lookup::MacroLookup>,
    macro_release_density: BTreeMap<i64, f32>,
    /// Time-varying market-news block (spec §6.3 lines 701–717). Keyed by
    /// FRED release timestamp; values are PCA(12) projections of trailing-90d
    /// time-decay-weighted-mean release embeddings. Forward-filled at
    /// `vector_at(ts)` via `range(..=ts).next_back()`.
    news_pca: BTreeMap<i64, [f32; 12]>,
    /// Universe-wide daily news-flow (intensity + breadth), z-scored. Overlays
    /// the first two of the 12 news dims so the block carries real DAILY
    /// variation instead of the ~monthly FRED-PCA proxy alone (M1). Empty when
    /// the FNSPID daily aggregate is absent — block then keeps pure FRED-PCA.
    news_flow: crate::historical::market_news_flow::NewsFlowLookup,
}

impl MarketFeatureLookup {
    fn vector_at(&self, ts: i64) -> [f32; 96] {
        let mut out = [0.0f32; 96];

        let spx = self.ret("spx", ts);
        let nasdaq = self.ret("nasdaq", ts);
        let russell = self.ret("russell", ts);
        let dow = self.ret("dow", ts);
        let vix_level = self.close("vix", ts) as f32;
        let vix_ret = self.ret("vix", ts);
        let dxy = self.ret("dxy", ts);
        let gold = self.ret("gold", ts);
        let oil = self.ret("oil", ts);
        let btc_close = self.close("btc", ts);
        let eth_close = self.close("eth", ts);
        let btc_ret = self.ret("btc", ts);
        let eth_ret = self.ret("eth", ts);
        let eurusd = self.ret("eurusd", ts);
        let usdjpy = self.ret("usdjpy", ts);

        // Volatility & sentiment block.
        out[0] = (vix_level / 100.0).clamp(0.0, 2.0);
        let regime_idx = if vix_level >= 35.0 {
            0usize // extreme fear
        } else if vix_level >= 28.0 {
            1usize // fear
        } else if vix_level >= 20.0 {
            2usize // neutral
        } else if vix_level >= 14.0 {
            3usize // greed
        } else {
            4usize // extreme greed
        };
        out[1 + regime_idx] = 1.0;
        out[6] = (vix_level / 20.0).clamp(0.0, 5.0);
        out[7] = vix_ret;
        out[8] = ((vix_level / (18.0 + 250.0 * vix_ret.abs() + 1.0)).clamp(0.0, 2.0)).min(2.0);
        out[9] = self
            .macro_release_density
            .range(..=ts)
            .next_back()
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        out[10] = dxy;
        out[11] = ((gold - oil) * 0.5).clamp(-2.0, 2.0);

        // ── Yield Curve [12..24] ────────────────────────────────────────
        // Spec §6.3 line 614-631. Forward-filled to the most recent FRED
        // observation at-or-before `ts`. All 12 yields stored as percent.
        if let Some(ml) = self.macro_lookup.as_ref() {
            let yc = ml.yield_curve_at(ts);
            out[12] = yc.y1m as f32;
            out[13] = yc.y3m as f32;
            out[14] = yc.y6m as f32;
            out[15] = yc.y1 as f32;
            out[16] = yc.y2 as f32;
            out[17] = yc.y3 as f32;
            out[18] = yc.y5 as f32;
            out[19] = yc.y7 as f32;
            out[20] = yc.y10 as f32;
            out[21] = yc.y20 as f32;
            out[22] = yc.y30 as f32;
            out[23] = yc.spread_2y10y as f32;

            // ── FRED Macro [24..36] ─────────────────────────────────────
            // Spec §6.3 line 633-648, ordered per `FRED_SERIES`.
            let macro_vals = ml.macro_values_at(ts);
            for (i, &v) in macro_vals.iter().enumerate() {
                out[24 + i] = v as f32;
            }
        }

        // Sector/industry block — real ETF chart returns.
        // Order: XLK, XLF, XLY, XLC, XLV, XLI, XLP, XLE, XLB, XLRE, XLU
        let sector_keys = ["xlk","xlf","xly","xlc","xlv","xli","xlp","xle","xlb","xlre","xlu"];
        let sector_rets: [f32; 11] = std::array::from_fn(|i| self.ret(sector_keys[i], ts));
        for (i, &r) in sector_rets.iter().enumerate() {
            out[36 + i] = r;
        }
        let dispersion = stddev(sector_rets);
        out[47] = dispersion;
        let (best_idx, worst_idx) = argmax_min(sector_rets);
        out[48] = best_idx as f32 / 10.0;
        out[49] = worst_idx as f32 / 10.0;
        out[50] = (sector_rets[0] - sector_rets[1]).abs();
        out[51] = sector_rets.iter().filter(|&&r| r > 0.0).count() as f32 / 11.0;

        // World indices & cross-asset block.
        let ftse  = self.ret("ftse",  ts);
        let dax   = self.ret("dax",   ts);
        let n225  = self.ret("n225",  ts);
        let hsi   = self.ret("hsi",   ts);
        // 000001.SS (SSEC) has gaps in Yahoo Finance history; fall back to a
        // dampened HSI return (both are Asian large-caps, corr ~0.7) when missing.
        let ssec_raw = self.ret("ssec", ts);
        let ssec = if ssec_raw == 0.0 { hsi * 0.7 } else { ssec_raw };
        out[52] = spx;
        out[53] = nasdaq;
        out[54] = russell;
        out[55] = dow;
        out[56] = ftse;
        out[57] = dax;
        out[58] = n225;
        out[59] = hsi;
        out[60] = ssec;
        out[61] = dxy;
        out[62] = eurusd;
        out[63] = usdjpy;
        out[64] = gold;
        out[65] = oil;
        out[66] = (btc_close as f32 / 100_000.0).clamp(0.0, 2.0);
        out[67] = btc_ret;
        out[68] = (eth_close as f32 / 10_000.0).clamp(0.0, 2.0);
        out[69] = eth_ret;
        let total_proxy = (btc_close.max(0.0) + eth_close.max(0.0)).max(1e-6);
        out[70] = ((total_proxy.ln() as f32) / 12.0).clamp(0.0, 2.0);
        out[71] = (btc_close / total_proxy) as f32;

        // Breadth/flow block (proxy)
        let breadth = [spx, nasdaq, russell, dow]
            .iter()
            .filter(|r| **r > 0.0)
            .count() as f32;
        out[72] = (breadth / 4.0) * 2.0 - 1.0; // [-1, 1]
        let max_r = [spx, nasdaq, russell, dow]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_r = [spx, nasdaq, russell, dow]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        out[73] = (max_r - min_r).clamp(0.0, 4.0);
        out[74] = spx.abs();
        out[75] = nasdaq.abs();
        out[76] = russell.abs();
        out[77] = ((spx * nasdaq) + (spx * russell) + (nasdaq * russell)) / 3.0;
        out[78] = breadth / 4.0;
        out[79] = (spx + nasdaq + russell + dow) / 4.0;
        out[80] = max_r.max(0.0);
        out[81] = if oil.abs() > 1e-6 {
            (spx.abs() / oil.abs()).min(5.0)
        } else {
            0.5
        };
        out[82] = market_hours_state(ts);
        out[83] = minutes_since_market_event(ts);
        // Spec §6.3 News block (12 dims). Forward-fill the most recent
        // FRED-release-derived PCA(12) projection at-or-before `ts`. The map
        // is built once per run by `market_news_pca::build_news_pca_lookup`
        // over the same FRED corpus that already populates [24..36].
        if let Some((_, v)) = self.news_pca.range(..=ts).next_back() {
            out[84..96].copy_from_slice(v);
        }
        // M1: overlay the first two news dims with real, daily-varying universe
        // news flow (FNSPID-derived intensity + breadth, z-scored). The FRED-PCA
        // projection only moves on ~monthly release dates, so on its own the
        // block was near-constant; these two dims give it genuine daily signal.
        // Absent aggregate ⇒ the FRED-PCA values stay in place untouched.
        if let Some([z_intensity, z_breadth]) = self.news_flow.value_at(ts) {
            out[84] = z_intensity;
            out[85] = z_breadth;
        }

        out
    }

    fn ret(&self, key: &'static str, ts: i64) -> f32 {
        self.returns
            .get(key)
            .and_then(|m| m.range(..=ts).next_back().map(|(_, v)| *v))
            .unwrap_or(0.0)
    }

    fn close(&self, key: &'static str, ts: i64) -> f64 {
        self.closes
            .get(key)
            .and_then(|m| m.range(..=ts).next_back().map(|(_, v)| *v))
            .unwrap_or(0.0)
    }
}

async fn build_market_feature_lookup(args: &HistoricalArgs, end_ts: i64) -> MarketFeatureLookup {
    // Core macro + commodity proxies (always fetched)
    const MARKET_PROXIES: [(&str, &str); 12] = [
        ("spx", "^GSPC"),
        ("nasdaq", "^IXIC"),
        ("russell", "^RUT"),
        ("dow", "^DJI"),
        ("vix", "^VIX"),
        ("dxy", "UUP"),
        // ETF proxies have deep, gap-free Yahoo Finance history vs. front-month
        // futures contracts (GC=F / CL=F) which can have roll-gap holes.
        ("gold", "GLD"),
        ("oil", "USO"),
        ("btc", "BTC-USD"),
        ("eth", "ETH-USD"),
        // Forex tickers for live-aligned EUR/USD and USD/JPY dims [62, 63]
        ("eurusd", "EURUSD=X"),
        ("usdjpy", "JPY=X"),
    ];
    // Sector ETFs — replace index-blend proxies with real sector return series.
    // Order matches poll_sector_industry: Tech, Financials, ConsDiscretionary,
    // CommServices, Healthcare, Industrials, ConsStaples, Energy, Materials,
    // RealEstate, Utilities.
    const SECTOR_ETFS: [(&str, &str); 11] = [
        ("xlk", "XLK"), ("xlf", "XLF"), ("xly", "XLY"), ("xlc", "XLC"),
        ("xlv", "XLV"), ("xli", "XLI"), ("xlp", "XLP"), ("xle", "XLE"),
        ("xlb", "XLB"), ("xlre", "XLRE"), ("xlu", "XLU"),
    ];
    // International index returns — best-effort; falls back to 0.0 on failure.
    const WORLD_INDICES: [(&str, &str); 5] = [
        ("ftse", "^FTSE"),
        ("dax", "^GDAXI"),
        ("n225", "^N225"),
        ("hsi", "^HSI"),
        ("ssec", "000001.SS"),
    ];

    let mut out = MarketFeatureLookup::default();

    // M1: universe-wide daily news-flow overlay for the market news block. Built
    // once by `make fnspid` (Python) into `.market_news_daily.json`; loading here
    // is a cheap single-file read shared by all per-symbol collectors.
    out.news_flow = crate::historical::market_news_flow::NewsFlowLookup::load();
    if !out.news_flow.is_empty() {
        tracing::info!("market_news_flow: daily universe news overlay loaded");
    }

    for &(alias, symbol) in MARKET_PROXIES
        .iter()
        .chain(SECTOR_ETFS.iter())
        .chain(WORLD_INDICES.iter())
    {
        match Ticker::new(symbol).await {
            Ok(ticker) => match fetch_historical_bars(&ticker, args, end_ts).await {
                Ok(bars) if bars.len() >= 2 => {
                    out.returns.insert(alias, build_return_series(&bars));
                    out.closes.insert(alias, build_close_series(&bars));
                }
                Ok(_) => {
                    tracing::warn!(proxy = %symbol, "market proxy returned insufficient bars");
                }
                Err(err) => {
                    tracing::warn!(proxy = %symbol, err = %err, "failed to fetch market proxy bars");
                }
            },
            Err(err) => {
                tracing::warn!(proxy = %symbol, err = %err, "failed to initialize market proxy ticker");
            }
        }
    }

    // Yield curve + FRED macro covering the training window. The training
    // shards span ~10 years; fetching one extra year on either side gives
    // padding for forward-fill at the boundaries.
    let end_year = chrono::DateTime::from_timestamp(end_ts, 0)
        .map(|d| d.format("%Y").to_string().parse::<i32>().unwrap_or(2026))
        .unwrap_or(2026);
    let start_year = end_year - 11;
    out.macro_lookup = Some(
        crate::historical::macro_lookup::MacroLookup::build(start_year, end_year).await,
    );
    if let Some(ml) = out.macro_lookup.as_ref() {
        out.macro_release_density = build_macro_release_density_lookup(&ml.per_series_raw);
    }

    // ── Market news block via FRED-release PCA(12) ──────────────────────
    //
    // finance-query has no dated archive of macro news; we substitute with
    // the FRED release calendar. Each observation we already pulled into
    // `macro_lookup.per_series_raw` becomes a deterministic text snippet
    // ("FEDFUNDS release 2024-03-15: value 5.33"). Embedded once via the
    // shared cache, projected to 12-d via NIPALS PCA, and pre-pooled per
    // release-date for O(log N) lookup at vector_at(ts).
    if let Some(ml) = out.macro_lookup.as_ref() {
        if !ml.per_series_raw.is_empty() {
            match crate::historical::embedder::Embedder::new(
                crate::historical::embedder::EmbedderConfig::default(),
            ) {
                Ok(embedder) => {
                    let releases = crate::historical::market_news_pca::build_release_corpus(
                        crate::historical::macro_lookup::FRED_SERIES,
                        crate::historical::macro_lookup::FRED_LABELS,
                        &ml.per_series_raw,
                    );
                    match crate::historical::market_news_pca::fit_or_load(
                        &embedder, &releases,
                    )
                    .await
                    {
                        Ok(pca) => match crate::historical::market_news_pca::build_news_pca_lookup(
                            &embedder, &pca, &releases,
                        )
                        .await
                        {
                            Ok(map) => {
                                tracing::info!(
                                    releases = releases.len(),
                                    pooled_dates = map.len(),
                                    "market_news_pca: lookup ready"
                                );
                                out.news_pca = map;
                            }
                            Err(e) => tracing::warn!(err = %e, "news_pca lookup build failed"),
                        },
                        Err(e) => tracing::warn!(err = %e, "news_pca fit/load failed"),
                    }
                }
                Err(e) => tracing::warn!(err = %e, "news_pca embedder init failed"),
            }
        }
    }

    out
}

fn build_macro_release_density_lookup(per_series_raw: &[BTreeMap<i64, f64>]) -> BTreeMap<i64, f32> {
    let mut events = Vec::<i64>::new();
    let mut unique_ts = std::collections::BTreeSet::<i64>::new();
    for s in per_series_raw {
        for &ts in s.keys() {
            events.push(ts);
            unique_ts.insert(ts);
        }
    }
    events.sort_unstable();

    let mut out = BTreeMap::new();
    if events.is_empty() {
        return out;
    }
    let window = 30 * 86_400;
    let mut left = 0usize;
    let mut right = 0usize;
    for ts in unique_ts {
        while right < events.len() && events[right] <= ts {
            right += 1;
        }
        while left < right && events[left] < ts - window {
            left += 1;
        }
        let count = (right - left) as f32;
        let density = (count.ln_1p() / 3.0).clamp(0.0, 1.0);
        out.insert(ts, density);
    }
    out
}

async fn fetch_historical_bars(
    ticker: &Ticker,
    args: &HistoricalArgs,
    end_ts: i64,
) -> anyhow::Result<Vec<Bar>> {
    let today = chrono::Utc::now().date_naive();
    let bars = match args.interval {
        HistoricalInterval::OneMinute => {
            let start_ts = args
                .start
                .unwrap_or(today - chrono::Duration::days(29))
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            ticker
                .chart_range(Interval::OneMinute, start_ts, end_ts)
                .await?
                .candles
                .iter()
                .map(bar_from_candle)
                .collect()
        }
        HistoricalInterval::FiveMinutes => {
            let start_ts = args
                .start
                .unwrap_or(today - chrono::Duration::days(58))
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            ticker
                .chart_range(Interval::FiveMinutes, start_ts, end_ts)
                .await?
                .candles
                .iter()
                .map(bar_from_candle)
                .collect()
        }
        HistoricalInterval::OneHour => {
            let start_ts = args
                .start
                .unwrap_or(today - chrono::Duration::days(730))
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            ticker
                .chart_range(Interval::OneHour, start_ts, end_ts)
                .await?
                .candles
                .iter()
                .map(bar_from_candle)
                .collect()
        }
        HistoricalInterval::OneDay => match args.start {
            Some(start) => {
                let start_ts = start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                ticker
                    .chart_range(Interval::OneDay, start_ts, end_ts)
                    .await?
                    .candles
                    .iter()
                    .map(bar_from_candle)
                    .collect()
            }
            None => ticker
                .chart(Interval::OneDay, TimeRange::TenYears)
                .await?
                .candles
                .iter()
                .filter(|c| c.timestamp <= end_ts)
                .map(bar_from_candle)
                .collect(),
        },
    };
    Ok(bars)
}

fn build_return_series(bars: &[Bar]) -> BTreeMap<i64, f32> {
    let mut out = BTreeMap::new();
    for w in bars.windows(2) {
        let prev = &w[0];
        let cur = &w[1];
        if prev.close > 0.0 && cur.close > 0.0 {
            let ret = (cur.close / prev.close).ln() as f32;
            out.insert(cur.timestamp, ret.clamp(-5.0, 5.0));
        }
    }
    out
}

fn build_close_series(bars: &[Bar]) -> BTreeMap<i64, f64> {
    let mut out = BTreeMap::new();
    for b in bars {
        out.insert(b.timestamp, b.close.max(0.0));
    }
    out
}

fn stddev<const N: usize>(vals: [f32; N]) -> f32 {
    if N == 0 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f32>() / N as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / N as f32;
    var.sqrt()
}

fn argmax_min<const N: usize>(vals: [f32; N]) -> (usize, usize) {
    let mut best = 0usize;
    let mut worst = 0usize;
    for i in 1..N {
        if vals[i] > vals[best] {
            best = i;
        }
        if vals[i] < vals[worst] {
            worst = i;
        }
    }
    (best, worst)
}

fn bar_from_candle(c: &finance_query::Candle) -> Bar {
    Bar {
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
        timestamp: c.timestamp,
    }
}

#[derive(Clone, Copy, Default)]
struct GradeEvent {
    ts: i64,
    direction: f32,
}

#[derive(Clone, Copy, Default)]
struct InsiderEvent {
    ts: i64,
    shares: f32,
    value: f32,
    is_buy: Option<bool>,
}

#[derive(Clone, Copy, Default)]
struct EarningsEvent {
    ts: i64,
    surprise_pct: f32,
}

#[derive(Clone, Copy, Default)]
struct InstitutionEvent {
    ts: i64,
    pct_held: f32,
    pct_change: f32,
}

#[derive(Clone, Copy, Default)]
struct SecFilingEvent {
    ts: i64,
    is_10k: bool,
    is_10q: bool,
    is_8k: bool,
}

#[derive(Clone, Default)]
struct TextEvent {
    ts: i64,
    text: String,
    sentiment: f32,
}

#[derive(Clone, Copy, Default)]
struct OptionSnapshotMetrics {
    atm_iv: f32,
    term_slope: f32,
    put_call_oi: f32,
    put_call_vol: f32,
    max_pain_moneyness: f32,
    open_interest_chg: f32,
}

#[derive(Clone, Default)]
struct FundamentalStatic {
    trailing_pe: Option<f64>,
    forward_pe: Option<f64>,
    peg_ratio: Option<f64>,
    ev_to_ebitda: Option<f64>,
    dividend_yield: Option<f64>,
    profit_margins: Option<f64>,
    revenue_growth: Option<f64>,
    earnings_growth: Option<f64>,
    debt_to_equity: Option<f64>,
    free_cashflow: Option<f64>,
    market_cap: Option<f64>,
    return_on_equity: Option<f64>,
    current_ratio: Option<f64>,
    next_q_est_revision_30d: Option<f64>,
    next_q_est_revision_90d: Option<f64>,
    rev_est_revision_30d: Option<f64>,
    analyst_consensus: Option<f64>,
    strong_buy_fraction: Option<f64>,
    sector_idx: f64,
    mktcap_log_rank: f64,
    next_earnings_ts: Option<i64>,
    symbol_hash_rank: f64,
}

#[derive(Clone, Default)]
struct QuarterlyFundamentalSeries {
    revenue: BTreeMap<i64, f64>,
    net_income: BTreeMap<i64, f64>,
    ebitda: BTreeMap<i64, f64>,
    eps_diluted: BTreeMap<i64, f64>,
    total_debt: BTreeMap<i64, f64>,
    total_equity: BTreeMap<i64, f64>,
    cash_and_equivalents: BTreeMap<i64, f64>,
    current_assets: BTreeMap<i64, f64>,
    current_liabilities: BTreeMap<i64, f64>,
    operating_cash_flow: BTreeMap<i64, f64>,
    capital_expenditure: BTreeMap<i64, f64>,
    shares_outstanding: BTreeMap<i64, f64>,
    market_cap_fallback: Option<f64>,
    shares_fallback: Option<f64>,
}

fn build_quarterly_fundamental_series(
    income_stmt: &Option<serde_json::Value>,
    balance_stmt: &Option<serde_json::Value>,
    cashflow_stmt: &Option<serde_json::Value>,
    market_cap_fallback: Option<f64>,
    start_close: Option<f64>,
) -> QuarterlyFundamentalSeries {
    QuarterlyFundamentalSeries {
        revenue: extract_statement_series(
            income_stmt,
            &[
                "TotalRevenue",
                "totalRevenue",
                "Revenue",
                "revenue",
                "Revenues",
                "OperatingRevenue",
            ],
        ),
        net_income: extract_statement_series(
            income_stmt,
            &[
                "NetIncome",
                "netIncome",
                "NetIncomeCommonStockholders",
                "NetIncomeIncludingNoncontrollingInterests",
                "NetIncomeFromContinuingOperations",
            ],
        ),
        ebitda: extract_statement_series(
            income_stmt,
            &["EBITDA", "ebitda", "NormalizedEBITDA", "Ebitda", "OperatingIncome"],
        ),
        eps_diluted: extract_statement_series(
            income_stmt,
            &[
                "DilutedEPS",
                "dilutedEPS",
                "BasicEPS",
                "basicEPS",
                "ReportedEPS",
                "epsDiluted",
            ],
        ),
        total_debt: extract_statement_series(
            balance_stmt,
            &["TotalDebt", "NetDebt", "LongTermDebtAndCapitalLeaseObligation"],
        ),
        total_equity: extract_statement_series(
            balance_stmt,
            &[
                "StockholdersEquity",
                "TotalEquityGrossMinorityInterest",
                "CommonStockEquity",
            ],
        ),
        cash_and_equivalents: extract_statement_series(
            balance_stmt,
            &[
                "CashAndCashEquivalents",
                "CashCashEquivalentsAndShortTermInvestments",
                "CashAndShortTermInvestments",
            ],
        ),
        current_assets: extract_statement_series(
            balance_stmt,
            &["CurrentAssets", "TotalCurrentAssets"],
        ),
        current_liabilities: extract_statement_series(
            balance_stmt,
            &["CurrentLiabilities", "TotalCurrentLiabilities"],
        ),
        operating_cash_flow: extract_statement_series(
            cashflow_stmt,
            &[
                "OperatingCashFlow",
                "CashFlowFromContinuingOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivities",
            ],
        ),
        capital_expenditure: extract_statement_series(
            cashflow_stmt,
            &[
                "CapitalExpenditure",
                "CapitalExpenditures",
                "PaymentsToAcquirePropertyPlantAndEquipment",
            ],
        ),
        shares_outstanding: extract_statement_series(
            balance_stmt,
            &[
                "OrdinarySharesNumber",
                "ShareIssued",
                "CommonStockSharesOutstanding",
                "sharesOutstanding",
            ],
        ),
        market_cap_fallback,
        shares_fallback: market_cap_fallback
            .zip(start_close)
            .and_then(|(mc, px)| (px > 0.0).then_some(mc / px)),
    }
}

fn extract_statement_series(
    stmt: &Option<serde_json::Value>,
    metric_keys: &[&str],
) -> BTreeMap<i64, f64> {
    let Some(statement) = stmt
        .as_ref()
        .and_then(|v| json_at(v, &["statement"]))
        .and_then(|v| v.as_object())
    else {
        return BTreeMap::new();
    };

    for key in metric_keys {
        if let Some(raw) = statement.get(*key) {
            let series = parse_statement_series_value(raw);
            if !series.is_empty() {
                return series;
            }
        }
    }
    BTreeMap::new()
}

fn parse_statement_series_value(v: &serde_json::Value) -> BTreeMap<i64, f64> {
    let mut out = BTreeMap::new();
    if let Some(obj) = v.as_object() {
        for (k, value) in obj {
            let Some(ts) = parse_financial_date_to_ts(k) else {
                continue;
            };
            if let Some(val) = json_f64_path(value, &["raw"])
                .or_else(|| value.as_f64())
            {
                if val.is_finite() {
                    out.insert(ts, val);
                }
            }
        }
        return out;
    }
    if let Some(arr) = v.as_array() {
        for row in arr {
            let date = json_at(row, &["asOfDate"])
                .and_then(|x| x.as_str())
                .or_else(|| json_at(row, &["date"]).and_then(|x| x.as_str()))
                .or_else(|| json_at(row, &["endDate"]).and_then(|x| x.as_str()));
            let Some(ts) = date.and_then(parse_financial_date_to_ts) else {
                continue;
            };
            let val = json_f64_path(row, &["reportedValue", "raw"])
                .or_else(|| json_f64_path(row, &["value", "raw"]))
                .or_else(|| json_f64_path(row, &["raw"]))
                .or_else(|| row.as_f64());
            if let Some(x) = val && x.is_finite() {
                out.insert(ts, x);
            }
        }
    }
    out
}

fn parse_financial_date_to_ts(s: &str) -> Option<i64> {
    parse_ymd_to_ts(s)
        .or_else(|| parse_mdy_financial(s))
        .or_else(|| parse_unix_ts_str(s))
        .or_else(|| parse_ymd_prefix(s))
        .or_else(|| {
        chrono::NaiveDate::parse_from_str(s, "%Y%m%d")
            .ok()
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| dt.and_utc().timestamp())
    })
}

fn parse_mdy_financial(s: &str) -> Option<i64> {
    let mut parts = s.split('/');
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    let y: i32 = parts.next()?.parse().ok()?;
    let date = chrono::NaiveDate::from_ymd_opt(y, m, d)?;
    Some(date.and_hms_opt(0, 0, 0)?.and_utc().timestamp())
}

fn parse_ymd_prefix(s: &str) -> Option<i64> {
    let prefix = s.get(..10)?;
    parse_ymd_to_ts(prefix)
}

fn parse_unix_ts_str(s: &str) -> Option<i64> {
    let raw: i64 = s.parse().ok()?;
    if raw > 1_000_000_000_000 {
        Some(raw / 1000)
    } else if raw > 1_000_000_000 {
        Some(raw)
    } else {
        None
    }
}

fn build_fundamental_static(
    symbol: &str,
    summary: &Option<serde_json::Value>,
    financial: &Option<serde_json::Value>,
    key_stats: &Option<serde_json::Value>,
    profile: &Option<serde_json::Value>,
    recommendation_trend: &Option<serde_json::Value>,
    earnings_trend: &Option<serde_json::Value>,
    calendar_events: &Option<serde_json::Value>,
    major_holders: &Option<serde_json::Value>,
) -> FundamentalStatic {
    // trailing_pe: Yahoo's summaryDetail/defaultKeyStatistics inconsistently
    // include this field (often absent for post-split, high-growth, and recent
    // IPO names). Compute from trailingEps / currentPrice as a final fallback
    // so the per-bar fundamentals block never has a structural None.
    let trailing_pe = summary
        .as_ref()
        .and_then(|v| json_f64_path(v, &["trailingPe", "raw"]).or_else(|| json_f64_path(v, &["trailingPe"])))
        .or_else(|| key_stats.as_ref().and_then(|v| json_f64_path(v, &["trailingPe", "raw"]).or_else(|| json_f64_path(v, &["trailingPe"]))))
        .or_else(|| {
            let eps = key_stats.as_ref()
                .and_then(|v| json_f64_path(v, &["trailingEps", "raw"]).or_else(|| json_f64_path(v, &["trailingEps"])));
            let price = financial.as_ref()
                .and_then(|v| json_f64_path(v, &["currentPrice", "raw"]).or_else(|| json_f64_path(v, &["currentPrice"])));
            match (eps, price) {
                (Some(e), Some(p)) if e > 0.0 => Some(p / e),
                _ => None,
            }
        });
    // forward_pe lives in defaultKeyStatistics, serialized "forwardPE" (uppercase —
    // explicit serde rename), NOT in summaryDetail. The old code read "forwardPe"
    // from `summary`, which is wrong module AND wrong casing, so forward P/E was
    // None for every symbol. Read key_stats first; fall back across spellings.
    let forward_pe = key_stats
        .as_ref()
        .and_then(|v| json_f64_path(v, &["forwardPE", "raw"]).or_else(|| json_f64_path(v, &["forwardPE"])))
        .or_else(|| key_stats.as_ref().and_then(|v| json_f64_path(v, &["forwardPe", "raw"]).or_else(|| json_f64_path(v, &["forwardPe"]))))
        .or_else(|| summary.as_ref().and_then(|v| json_f64_path(v, &["forwardPe", "raw"]).or_else(|| json_f64_path(v, &["forwardPe"]))));
    let market_cap = summary
        .as_ref()
        .and_then(|v| json_f64_path(v, &["marketCap", "raw"]).or_else(|| json_f64_path(v, &["marketCap"])));
    let dividend_yield = summary
        .as_ref()
        .and_then(|v| json_f64_path(v, &["dividendYield", "raw"]).or_else(|| json_f64_path(v, &["dividendYield"])));

    let peg_ratio = key_stats
        .as_ref()
        .and_then(|v| json_f64_path(v, &["pegRatio"]).or_else(|| json_f64_path(v, &["pegRatio", "raw"])));
    let ev_to_ebitda = key_stats
        .as_ref()
        .and_then(|v| json_f64_path(v, &["enterpriseToEbitda", "raw"]).or_else(|| json_f64_path(v, &["enterpriseToEbitda"])));

    let profit_margins = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["profitMargins", "raw"]).or_else(|| json_f64_path(v, &["profitMargins"])));
    let revenue_growth = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["revenueGrowth", "raw"]).or_else(|| json_f64_path(v, &["revenueGrowth"])));
    let earnings_growth = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["earningsGrowth", "raw"]).or_else(|| json_f64_path(v, &["earningsGrowth"])));
    let debt_to_equity = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["debtToEquity", "raw"]).or_else(|| json_f64_path(v, &["debtToEquity"])));
    let free_cashflow = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["freeCashflow", "raw"]).or_else(|| json_f64_path(v, &["freeCashflow"])));
    let return_on_equity = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["returnOnEquity", "raw"]).or_else(|| json_f64_path(v, &["returnOnEquity"])));
    let current_ratio = financial
        .as_ref()
        .and_then(|v| json_f64_path(v, &["currentRatio", "raw"]).or_else(|| json_f64_path(v, &["currentRatio"])));

    let rec0 = recommendation_trend
        .as_ref()
        .and_then(|v| json_at(v, &["trend"]))
        .and_then(|a| a.as_array())
        .and_then(|a| a.first());
    let analyst_consensus = rec0.and_then(|p| {
        let sb = json_i64_path(p, &["strongBuy"]).unwrap_or(0) as f64;
        let b = json_i64_path(p, &["buy"]).unwrap_or(0) as f64;
        let h = json_i64_path(p, &["hold"]).unwrap_or(0) as f64;
        let s = json_i64_path(p, &["sell"]).unwrap_or(0) as f64;
        let ss = json_i64_path(p, &["strongSell"]).unwrap_or(0) as f64;
        let total = (sb + b + h + s + ss).max(1.0);
        let avg = (1.0 * sb + 2.0 * b + 3.0 * h + 4.0 * s + 5.0 * ss) / total;
        Some(((3.0 - avg) / 2.0).clamp(-1.0, 1.0))
    });
    let strong_buy_fraction = rec0.map(|p| {
        let sb = json_i64_path(p, &["strongBuy"]).unwrap_or(0) as f64;
        let b = json_i64_path(p, &["buy"]).unwrap_or(0) as f64;
        let h = json_i64_path(p, &["hold"]).unwrap_or(0) as f64;
        let s = json_i64_path(p, &["sell"]).unwrap_or(0) as f64;
        let ss = json_i64_path(p, &["strongSell"]).unwrap_or(0) as f64;
        sb / (sb + b + h + s + ss).max(1.0)
    });

    let et0 = earnings_trend
        .as_ref()
        .and_then(|v| json_at(v, &["trend"]))
        .and_then(|a| a.as_array())
        .and_then(|a| a.first());
    let (next_q_est_revision_30d, next_q_est_revision_90d, rev_est_revision_30d) = et0
        .map(|p| {
            let up30 = json_i64_path(p, &["epsRevisions", "upLast30days", "raw"]).unwrap_or(0) as f64;
            let dn30 = json_i64_path(p, &["epsRevisions", "downLast30days", "raw"]).unwrap_or(0) as f64;
            let dn90 = json_i64_path(p, &["epsRevisions", "downLast90days", "raw"]).unwrap_or(0) as f64;
            let n = (up30 + dn30 + dn90).max(1.0);
            let est_30 = (up30 - dn30) / n;
            let est_90 = (up30 - dn90) / n;
            let rev_30 = json_f64_path(p, &["revenueEstimate", "growth", "raw"]).unwrap_or(0.0);
            (Some(est_30), Some(est_90), Some(rev_30))
        })
        .unwrap_or((None, None, None));

    let sector_idx = profile
        .as_ref()
        .and_then(|p| json_at(p, &["sector"]))
        .and_then(|s| s.as_str())
        .map(sector_to_idx)
        .unwrap_or(0.5);

    let mktcap_log_rank = market_cap
        .map(|m| ((m.max(1.0).ln() - 18.0) / 10.0).clamp(0.0, 1.0))
        .unwrap_or_else(|| simple_symbol_hash(symbol));

    let next_earnings_ts = calendar_events
        .as_ref()
        .and_then(|c| json_i64_path(c, &["earnings", "earningsDate", "0", "raw"]));

    let institutional_pct = major_holders
        .as_ref()
        .and_then(|m| json_f64_path(m, &["institutionsPercentHeld", "raw"]));
    let insider_pct = major_holders
        .as_ref()
        .and_then(|m| json_f64_path(m, &["insidersPercentHeld", "raw"]));

    let mut out = FundamentalStatic {
        trailing_pe,
        forward_pe,
        peg_ratio,
        ev_to_ebitda,
        dividend_yield,
        profit_margins,
        revenue_growth,
        earnings_growth,
        debt_to_equity,
        free_cashflow,
        market_cap,
        return_on_equity,
        current_ratio,
        next_q_est_revision_30d,
        next_q_est_revision_90d,
        rev_est_revision_30d,
        analyst_consensus,
        strong_buy_fraction,
        sector_idx,
        mktcap_log_rank,
        next_earnings_ts,
        symbol_hash_rank: simple_symbol_hash(symbol),
    };
    if out.strong_buy_fraction.is_none() {
        out.strong_buy_fraction = Some(0.2 + 0.6 * out.symbol_hash_rank);
    }
    if out.dividend_yield.is_none() {
        out.dividend_yield = Some((insider_pct.unwrap_or(0.03) * 0.25).clamp(0.0, 0.08));
    }
    if out.analyst_consensus.is_none() {
        out.analyst_consensus = Some(0.1 + (institutional_pct.unwrap_or(0.6) - 0.5) * 0.8);
    }
    if out.free_cashflow.is_none() {
        if let (Some(pm), Some(mc)) = (out.profit_margins, out.market_cap) {
            out.free_cashflow = Some(pm * mc * 0.65);
        }
    }
    out
}

fn extract_earnings_events(
    earnings_history: &Option<serde_json::Value>,
) -> Vec<EarningsEvent> {
    let mut out = Vec::new();
    if let Some(hist) = earnings_history
        .as_ref()
        .and_then(|h| json_at(h, &["history"]))
        .and_then(|a| a.as_array())
    {
        for e in hist {
            if let Some(ts) = json_i64_path(e, &["quarter", "raw"]) {
                out.push(EarningsEvent {
                    ts,
                    surprise_pct: json_f64_path(e, &["surprisePercent", "raw"]).unwrap_or(0.0) as f32,
                });
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    out
}

/// Earnings proximity `exp(-days/14)` to the nearest real or projected report
/// date (H2). Dates come from `earnings_history` + the next scheduled date; with
/// ≥2 real dates we project the company's own median (~quarterly) cadence across
/// the window — anchored on real report dates, not a generic cycle. 0.0 if none.
fn earnings_proximity(
    bar_ts: i64,
    events: &[EarningsEvent],
    next_earnings_ts: Option<i64>,
) -> f32 {
    let day = 86_400i64;
    let mut dates: Vec<i64> = events.iter().map(|e| e.ts).collect();
    if let Some(n) = next_earnings_ts {
        dates.push(n);
    }
    if dates.is_empty() {
        return 0.0;
    }
    dates.sort_unstable();
    let mut min_days = dates
        .iter()
        .map(|d| (d - bar_ts).abs() as f32 / day as f32)
        .fold(f32::INFINITY, f32::min);
    if dates.len() >= 2 {
        let mut gaps: Vec<i64> = dates.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0).collect();
        if !gaps.is_empty() {
            gaps.sort_unstable();
            let q = gaps[gaps.len() / 2].max(day); // median spacing ≈ one quarter
            let a0 = dates[0];
            let k = (((bar_ts - a0) as f64) / q as f64).round() as i64;
            let proj_days = ((a0 + k * q) - bar_ts).abs() as f32 / day as f32;
            min_days = min_days.min(proj_days);
        }
    }
    (-min_days.min(60.0) / 14.0).exp()
}

fn extract_grade_events(
    grading_history: &Option<serde_json::Value>,
) -> Vec<GradeEvent> {
    let mut out = Vec::new();
    if let Some(hist) = grading_history
        .as_ref()
        .and_then(|g| json_at(g, &["history"]))
        .and_then(|a| a.as_array())
    {
        for e in hist {
            if let Some(ts) = json_i64_path(e, &["epochGradeDate"]) {
                let action = json_at(e, &["action"])
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let dir = if action.contains("up") {
                    1.0
                } else if action.contains("down") {
                    -1.0
                } else {
                    0.0
                };
                out.push(GradeEvent { ts, direction: dir });
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    out
}

fn extract_insider_events(
    insider_transactions: &Option<serde_json::Value>,
) -> Vec<InsiderEvent> {
    let mut out = Vec::new();
    if let Some(txs) = insider_transactions
        .as_ref()
        .and_then(|i| json_at(i, &["transactions"]))
        .and_then(|a| a.as_array())
    {
        for tx in txs {
            if let Some(ts) = json_i64_path(tx, &["startDate", "raw"]) {
                let shares = json_f64_path(tx, &["shares", "raw"])
                    .or_else(|| json_i64_path(tx, &["shares", "raw"]).map(|v| v as f64))
                    .unwrap_or(0.0) as f32;
                let value = json_f64_path(tx, &["value", "raw"])
                    .or_else(|| json_i64_path(tx, &["value", "raw"]).map(|v| v as f64))
                    .unwrap_or(0.0) as f32;
                let is_buy = infer_insider_is_buy(tx, shares, value);
                out.push(InsiderEvent {
                    ts,
                    shares,
                    value,
                    is_buy,
                });
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    out
}

fn infer_insider_is_buy(tx: &serde_json::Value, shares: f32, value: f32) -> Option<bool> {
    let mut text = String::new();
    if let Some(s) = json_at(tx, &["transactionText"]).and_then(|v| v.as_str()) {
        text.push_str(s);
        text.push(' ');
    }
    if let Some(s) = json_at(tx, &["moneyText"]).and_then(|v| v.as_str()) {
        text.push_str(s);
    }
    let t = text.to_ascii_lowercase();
    if t.contains("buy") || t.contains("bought") || t.contains("purchase") || t.contains("acquir")
    {
        return Some(true);
    }
    if t.contains("sell") || t.contains("sold") || t.contains("sale") || t.contains("dispos") {
        return Some(false);
    }
    if shares < 0.0 || value < 0.0 {
        return Some(false);
    }
    if shares > 0.0 && value > 0.0 {
        return Some(true);
    }
    None
}

fn extract_institution_events(
    institution_ownership: &Option<serde_json::Value>,
) -> Vec<InstitutionEvent> {
    let mut out = Vec::new();
    if let Some(owners) = institution_ownership
        .as_ref()
        .and_then(|i| json_at(i, &["ownershipList"]))
        .and_then(|a| a.as_array())
    {
        for o in owners {
            if let Some(ts) = json_i64_path(o, &["reportDate", "raw"]) {
                out.push(InstitutionEvent {
                    ts,
                    pct_held: json_f64_path(o, &["pctHeld", "raw"]).unwrap_or(0.0) as f32,
                    pct_change: json_f64_path(o, &["pctChange", "raw"]).unwrap_or(0.0) as f32,
                });
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    out
}

/// SEC filing-recency timeline from the full-history doc list. Forms are
/// normalized with a prefix match so amendments (10-K/A, 10-Q/A, 8-K/A, …) count
/// as their base form.
fn build_sec_filing_timeline(docs: &[EdgarFilingDoc]) -> Vec<SecFilingEvent> {
    let mut out: Vec<SecFilingEvent> = docs
        .iter()
        .map(|d| {
            let form = d.form.trim().to_ascii_uppercase();
            SecFilingEvent {
                ts: d.filing_ts,
                is_10k: form.starts_with("10-K"),
                is_10q: form.starts_with("10-Q"),
                is_8k: form.starts_with("8-K"),
            }
        })
        .collect();
    out.sort_by_key(|e| e.ts);
    out
}

/// Fetch + strip EDGAR primary-document text for all text-bearing filings of an
/// entity, returning `(filing_ts, document_text)` pairs (UTC-midnight timestamps).
/// Network/parse failures degrade to an empty list, in which case Block 8 becomes
/// a genuine zero vector for the affected bars (honest absence, not a fill).
/// Returns (all text-form filings across full history, their `(ts, text)` pairs).
/// The doc list also drives the SEC filing-recency timeline (Block 6). Failures
/// degrade to empty (Block 8 → zero vector for affected bars; honest absence).
async fn fetch_filing_texts(
    edgar: &Option<finance_query::EdgarSubmissions>,
) -> (Vec<EdgarFilingDoc>, Vec<(i64, String)>) {
    use crate::historical::edgar_text::{EdgarTextConfig, EdgarTextFetcher};

    let Some(subs) = edgar.as_ref() else {
        return (Vec::new(), Vec::new());
    };
    let Some(cik) = subs.cik.as_ref() else {
        tracing::warn!("edgar submissions missing CIK — Block 8 text will be empty");
        return (Vec::new(), Vec::new());
    };
    let fetcher = match EdgarTextFetcher::new(EdgarTextConfig::default()) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(err = %e, "edgar_text fetcher init failed");
            return (Vec::new(), Vec::new());
        }
    };
    let docs = fetcher.gather_filings(subs).await;
    let mut texts = Vec::with_capacity(docs.len());
    for f in &docs {
        match fetcher.fetch_document_text(cik, f).await {
            Ok(text) if !text.trim().is_empty() => texts.push((f.filing_ts, text)),
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(accession = %f.accession_number, err = %e, "edgar doc fetch failed")
            }
        }
    }
    tracing::info!(cik = %cik, filings = docs.len(), texts = texts.len(), "edgar filings fetched");
    (docs, texts)
}

/// Score each Block-8 text item (EDGAR filing document ∪ FNSPID news headline)
/// with the shared FinBERT client — the *same* model the live `news_sentiment`
/// poller uses, so the Block-6 sentiment slot is aligned across train and
/// inference. Client-init or per-item scoring failures degrade to a neutral
/// `0.0` (honest "unknown", no lexical fallback); the text still feeds the
/// embedder, so Block 8 keeps its signal regardless.
async fn score_texts(texts: &[(i64, String)]) -> Vec<(i64, String, f32)> {
    use crate::sentiment::{SentimentClient, SentimentConfig};

    let client = match SentimentClient::new(SentimentConfig::default()) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(err = %e, "sentiment client init failed; filing sentiment → 0.0");
            None
        }
    };
    let mut out = Vec::with_capacity(texts.len());
    for (ts, text) in texts {
        let sentiment = match &client {
            Some(c) => c.score(text).await.unwrap_or_else(|e| {
                tracing::debug!(err = %e, "filing sentiment score failed → 0.0");
                0.0
            }),
            None => 0.0,
        };
        out.push((*ts, text.clone(), sentiment));
    }
    out
}

/// Build the dated Block-8 text-event stream from real EDGAR primary-document
/// text. Each `(filing_ts, text, sentiment)` becomes a decay-weighted snippet
/// that `compute_text_embedding_at` pools over the trailing window. There is no
/// static identity/metadata event — that previously collapsed Block 8 into a
/// constant per-asset fingerprint. FNSPID news (historical) is merged by the
/// Python ingestion layer; live inference adds the `feeds` stream through the
/// same embedder + FinBERT pipeline.
fn build_text_events(filing_texts: &[(i64, String, f32)]) -> Vec<TextEvent> {
    let mut out = Vec::with_capacity(filing_texts.len());
    for (ts, text, sentiment) in filing_texts {
        if text.trim().is_empty() {
            continue;
        }
        out.push(TextEvent {
            ts: *ts,
            sentiment: *sentiment,
            text: text.clone(),
        });
    }
    out.sort_by_key(|e| e.ts);
    out
}

fn extract_options_snapshot_metrics(options: Option<&finance_query::Options>) -> OptionSnapshotMetrics {
    let mut m = OptionSnapshotMetrics::default();
    let Some(options) = options else {
        return m;
    };
    let calls = options.calls();
    let puts = options.puts();
    let call_iv: Vec<f32> = calls
        .iter()
        .filter_map(|c| c.implied_volatility.map(|x| x as f32))
        .collect();
    let put_iv: Vec<f32> = puts
        .iter()
        .filter_map(|c| c.implied_volatility.map(|x| x as f32))
        .collect();
    let mean_call_iv = mean(&call_iv).unwrap_or(0.35);
    let mean_put_iv = mean(&put_iv).unwrap_or(0.35);
    m.atm_iv = 0.5 * (mean_call_iv + mean_put_iv);
    m.term_slope = (mean_call_iv - mean_put_iv).clamp(-1.0, 1.0);

    let call_oi: f32 = calls
        .iter()
        .map(|c| c.open_interest.unwrap_or(0) as f32)
        .sum();
    let put_oi: f32 = puts
        .iter()
        .map(|c| c.open_interest.unwrap_or(0) as f32)
        .sum();
    let call_vol: f32 = calls.iter().map(|c| c.volume.unwrap_or(0) as f32).sum();
    let put_vol: f32 = puts.iter().map(|c| c.volume.unwrap_or(0) as f32).sum();
    m.put_call_oi = (put_oi + 1.0) / (call_oi + 1.0);
    m.put_call_vol = (put_vol + 1.0) / (call_vol + 1.0);

    let strikes: Vec<f32> = calls
        .iter()
        .chain(puts.iter())
        .map(|c| c.strike as f32)
        .filter(|s| s.is_finite())
        .collect();
    if !strikes.is_empty() {
        let strike_mid = strikes.iter().copied().sum::<f32>() / strikes.len() as f32;
        let strike_var = strikes
            .iter()
            .map(|s| (s - strike_mid) * (s - strike_mid))
            .sum::<f32>()
            / strikes.len() as f32;
        m.max_pain_moneyness = (strike_var.sqrt() / (strike_mid.abs() + 1.0)).clamp(0.0, 1.0);
    }

    m.open_interest_chg = ((call_vol + put_vol) / (call_oi + put_oi + 1.0)).clamp(0.0, 5.0);
    m
}

fn compute_options_proxy_at(
    asset_window: &[Bar],
    snapshot: &OptionSnapshotMetrics,
    earnings_proximity: f32,
) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    if asset_window.len() < 6 {
        out[0] = snapshot.atm_iv.max(0.05);
        out[1] = 0.5;
        out[2] = snapshot.term_slope;
        out[3] = snapshot.put_call_vol.clamp(0.1, 10.0);
        out[4] = snapshot.put_call_oi.clamp(0.1, 10.0);
        out[5] = 0.2;
        out[6] = 0.3;
        out[7] = snapshot.max_pain_moneyness;
        out[8] = 0.5;
        out[9] = 0.1;
        out[10] = earnings_proximity.clamp(0.0, 1.0);
        out[11] = snapshot.open_interest_chg.clamp(0.0, 5.0) / 5.0;
        return out;
    }

    let rets: Vec<f32> = asset_window
        .windows(2)
        .filter_map(|w| {
            let p0 = w[0].close as f32;
            let p1 = w[1].close as f32;
            (p0 > 0.0 && p1 > 0.0).then_some((p1 / p0).ln())
        })
        .collect();
    let rv20 = annualized_vol(&rets.iter().rev().take(20).copied().collect::<Vec<_>>()).unwrap_or(0.2);
    let rv60 = annualized_vol(&rets.iter().rev().take(60).copied().collect::<Vec<_>>()).unwrap_or(rv20);
    let up: Vec<f32> = rets.iter().copied().filter(|r| *r > 0.0).collect();
    let dn: Vec<f32> = rets.iter().copied().filter(|r| *r < 0.0).map(f32::abs).collect();
    let up_vol = annualized_vol(&up).unwrap_or(rv20);
    let dn_vol = annualized_vol(&dn).unwrap_or(rv20);
    let mean_abs = rets.iter().map(|r| r.abs()).sum::<f32>() / rets.len() as f32;
    let last_abs = rets.last().copied().unwrap_or(0.0).abs();

    let (atm_iv, iv_known) = if snapshot.atm_iv > 0.01 {
        ((0.35 * snapshot.atm_iv + 0.65 * rv20).clamp(0.01, 3.0), true)
    } else {
        (rv20, false)
    };
    out[0] = atm_iv.clamp(0.01, 3.0);
    out[1] = (rv20 / (rv60 + 1e-6)).clamp(0.0, 3.0) / 3.0;
    out[2] = ((snapshot.term_slope + (rv20 - rv60)) * 0.5).clamp(-2.0, 2.0);
    out[3] = ((dn_vol + 1e-4) / (up_vol + 1e-4)).clamp(0.2, 5.0);
    out[4] = (0.5 * snapshot.put_call_oi + 0.5 * ((dn_vol + 1e-4) / (up_vol + 1e-4)))
        .clamp(0.2, 5.0);
    out[5] = snapshot.open_interest_chg.clamp(0.0, 5.0) / 5.0;
    out[6] = snapshot.put_call_vol.clamp(0.2, 5.0);
    out[7] = (0.5 * snapshot.max_pain_moneyness + 0.5 * (rv20 - rv60).abs().clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
    out[8] = (last_abs / (mean_abs + 1e-5)).clamp(0.0, 5.0) / 5.0;
    // IV premium: IV/RV spread. When no live options data, use the typical
    // vol-risk-premium of ~12% plus earnings proximity to avoid a structural zero.
    out[9] = if iv_known {
        ((atm_iv - rv20) / (rv20 + 1e-4)).clamp(-2.0, 2.0)
    } else {
        (0.12 + 0.3 * earnings_proximity).clamp(0.0, 2.0)
    };
    out[10] = earnings_proximity.clamp(0.0, 1.0);
    out[11] = ((out[0] * out[10]) + 0.25 * out[8]).clamp(0.0, 3.0) / 3.0;
    out
}

fn reconstruct_fundamentals_at(
    bar_ts: i64,
    base: &FundamentalStatic,
    earnings_events: &[EarningsEvent],
    grade_events: &[GradeEvent],
    insider_events: &[InsiderEvent],
    institution_events: &[InstitutionEvent],
) -> FundamentalData {
    let q_cycle = cycle_signal(bar_ts, 90.0);
    let y_cycle = cycle_signal(bar_ts, 252.0);
    let mut d = FundamentalData {
        trailing_pe: base.trailing_pe,
        forward_pe: base.forward_pe,
        peg_ratio: base.peg_ratio,
        ev_to_ebitda: base.ev_to_ebitda,
        dividend_yield: base.dividend_yield,
        profit_margins: base.profit_margins,
        revenue_growth: base.revenue_growth,
        earnings_growth: base.earnings_growth,
        debt_to_equity: base.debt_to_equity,
        free_cashflow: base.free_cashflow,
        market_cap: base.market_cap,
        return_on_equity: base.return_on_equity,
        current_ratio: base.current_ratio,
        next_q_est_revision_30d: base.next_q_est_revision_30d,
        next_q_est_revision_90d: base.next_q_est_revision_90d,
        rev_est_revision_30d: base.rev_est_revision_30d,
        analyst_consensus: base.analyst_consensus,
        strong_buy_fraction: base.strong_buy_fraction,
        sector_idx: base.sector_idx,
        mktcap_log_rank: base.mktcap_log_rank,
        ..FundamentalData::default()
    };

    let past_earnings: Vec<_> = earnings_events.iter().copied().filter(|e| e.ts <= bar_ts).collect();
    if let Some(last) = past_earnings.last() {
        d.last_earnings_surprise_pct = Some(last.surprise_pct as f64);
        d.days_since_earnings = Some(((bar_ts - last.ts).max(0) as f64) / 86_400.0);
        let mut streak = 0i32;
        for ev in past_earnings.iter().rev() {
            if ev.surprise_pct > 0.0 {
                streak += 1;
            } else if ev.surprise_pct < 0.0 {
                streak -= 1;
            } else {
                break;
            }
            if streak.abs() >= 5 {
                break;
            }
        }
        d.earnings_surprise_streak = streak;
    } else if let Some(next) = base.next_earnings_ts
        && (next - bar_ts).abs() <= 180 * 86_400
    {
        d.days_since_earnings = Some(((next - bar_ts).abs() as f64) / 86_400.0);
    } else {
        let cycle_days = ((bar_ts / 86_400) % 90) as f64;
        d.days_since_earnings = Some(cycle_days.min(90.0 - cycle_days));
    }
    if d.last_earnings_surprise_pct.is_none() {
        d.last_earnings_surprise_pct = Some(8.0 * q_cycle);
    }
    if d.earnings_surprise_streak == 0 {
        d.earnings_surprise_streak = (3.0 * q_cycle).round() as i32;
    }

    let g30: Vec<_> = grade_events
        .iter()
        .copied()
        .filter(|g| g.ts <= bar_ts && g.ts >= bar_ts - 30 * 86_400)
        .collect();
    if !g30.is_empty() {
        let momentum = g30.iter().map(|g| g.direction as f64).sum::<f64>() / g30.len() as f64;
        d.rating_momentum_30d = Some(momentum);
        d.recent_grade_direction = g30.last().map(|g| g.direction as f64).unwrap_or(0.0);
    } else {
        d.rating_momentum_30d = Some(0.0);
        d.recent_grade_direction = 0.0;
    }

    let i90: Vec<_> = insider_events
        .iter()
        .copied()
        .filter(|i| i.ts <= bar_ts && i.ts >= bar_ts - 90 * 86_400)
        .collect();
    let mut net_shares = 0.0f64;
    let mut net_value = 0.0f64;
    let mut buy_count = 0.0f64;
    if i90.is_empty() {
        // No insider transactions in window (pre-2022 coverage gap or illiquid symbol):
        // use small cycle-based proxies so dims [22]/[24] are never structural zeros.
        let mc_scale = base.market_cap.unwrap_or(1e9);
        net_shares = 5_000.0 * q_cycle;
        net_value = 50_000.0 * mc_scale.sqrt() / 31_623.0 * q_cycle; // √mc normalised
    } else {
        for tx in &i90 {
            let sign = match tx.is_buy {
                Some(true) => 1.0,
                Some(false) => -1.0,
                None => 0.0,
            };
            net_shares += sign * tx.shares as f64;
            net_value += sign * tx.value as f64;
            if tx.is_buy == Some(true) {
                buy_count += 1.0;
            }
        }
    }
    d.net_insider_shares_90d = Some(net_shares);
    d.net_insider_value_90d = Some(net_value);
    d.insider_buy_count_90d = buy_count.max(0.0);
    if d.insider_buy_count_90d <= 0.0 {
        d.insider_buy_count_90d = (1.0 + 2.0 * q_cycle.max(0.0)).clamp(0.0, 3.0);
    }

    let inst_past: Vec<_> = institution_events
        .iter()
        .copied()
        .filter(|e| e.ts <= bar_ts)
        .collect();
    if let Some(last) = inst_past.last() {
        d.institutional_pct = Some(last.pct_held as f64);
        d.institutional_delta_qtr = Some(last.pct_change as f64);
    } else {
        let pct = 0.5 + 0.2 * y_cycle + 0.2 * (base.symbol_hash_rank - 0.5);
        d.institutional_pct = Some(pct.clamp(0.05, 0.95));
        d.institutional_delta_qtr = Some((0.06 * y_cycle).clamp(-0.2, 0.2));
    }

    let grade_momentum = d.rating_momentum_30d.unwrap_or(0.0);
    let rev30_proxy = (0.18 * q_cycle + 0.12 * d.recent_grade_direction).clamp(-0.3, 0.3);
    let rev90_proxy = (0.10 * q_cycle + 0.10 * y_cycle + 0.10 * grade_momentum).clamp(-0.3, 0.3);
    let rev_est_proxy = (0.12 * q_cycle + 0.10 * grade_momentum).clamp(-0.3, 0.3);
    d.next_q_est_revision_30d =
        Some(0.5 * d.next_q_est_revision_30d.unwrap_or(rev30_proxy) + 0.5 * rev30_proxy);
    d.next_q_est_revision_90d =
        Some(0.5 * d.next_q_est_revision_90d.unwrap_or(rev90_proxy) + 0.5 * rev90_proxy);
    d.rev_est_revision_30d =
        Some(0.5 * d.rev_est_revision_30d.unwrap_or(rev_est_proxy) + 0.5 * rev_est_proxy);

    let consensus_proxy = (0.6 * grade_momentum + 0.25 * q_cycle).clamp(-1.0, 1.0);
    d.analyst_consensus = Some(
        0.5 * d.analyst_consensus.unwrap_or(consensus_proxy) + 0.5 * consensus_proxy,
    );
    let strong_buy_proxy = (0.5 + 0.25 * consensus_proxy + 0.15 * y_cycle).clamp(0.0, 1.0);
    d.strong_buy_fraction = Some(
        0.5 * d.strong_buy_fraction.unwrap_or(strong_buy_proxy) + 0.5 * strong_buy_proxy,
    );

    d.insider_ownership_pct =
        Some((0.02 + 0.15 * (1.0 - d.institutional_pct.unwrap_or(0.5)) + 0.03 * q_cycle).clamp(0.0, 0.4));
    d
}

fn latest_series_value(series: &BTreeMap<i64, f64>, ts: i64) -> Option<f64> {
    series.range(..=ts).next_back().map(|(_, v)| *v)
}

fn latest_n_series_values(series: &BTreeMap<i64, f64>, ts: i64, n: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for (_, v) in series.range(..=ts).rev().take(n) {
        out.push(*v);
    }
    out.reverse();
    out
}

fn sum_last_n_series(series: &BTreeMap<i64, f64>, ts: i64, n: usize) -> Option<f64> {
    let vals = latest_n_series_values(series, ts, n);
    (vals.len() == n).then_some(vals.iter().sum::<f64>())
}

fn yoy_growth(series: &BTreeMap<i64, f64>, ts: i64) -> Option<f64> {
    let vals = latest_n_series_values(series, ts, 5);
    if vals.len() < 5 {
        return None;
    }
    let current = *vals.last().unwrap_or(&0.0);
    let year_ago = vals[0];
    if year_ago.abs() < 1e-9 {
        None
    } else {
        Some(((current / year_ago) - 1.0).clamp(-1.5, 1.5))
    }
}

fn trailing_dividend_yield_at(ts: i64, close: f64, div_ts: &[i64], div_amounts: &[f64]) -> Option<f64> {
    if close <= 0.0 {
        return None;
    }
    let one_year = 365 * 86_400;
    let t0 = ts - one_year;
    let mut sum = 0.0f64;
    for (&dts, &amt) in div_ts.iter().zip(div_amounts.iter()) {
        if dts > t0 && dts <= ts {
            sum += amt.max(0.0);
        }
    }
    Some((sum / close).clamp(0.0, 0.25))
}

fn apply_quarterly_fundamentals_at(
    d: &mut FundamentalData,
    bar_ts: i64,
    close: f64,
    _window: &[Bar],
    q: &QuarterlyFundamentalSeries,
    div_timestamps: &[i64],
    div_amounts: &[f64],
) {
    // Debug: check if we have quarterly data
    let has_revenue = !q.revenue.is_empty();
    if has_revenue {
        if let Some(sample_val) = q.revenue.values().next() {
            tracing::debug!(
                bar_ts = bar_ts,
                close = close,
                rev_count = q.revenue.len(),
                sample_rev = sample_val,
                "apply_quarterly_fundamentals_at: starting with revenue data"
            );
        }
    }
    
    let shares = latest_series_value(&q.shares_outstanding, bar_ts)
        .filter(|x| *x > 0.0)
        .or(q.shares_fallback)
        .or_else(|| d.market_cap.and_then(|m| (close > 0.0).then_some(m / close)));
    let market_cap = shares
        .filter(|s| *s > 0.0 && close > 0.0)
        .map(|s| s * close)
        .or(q.market_cap_fallback);
    if let Some(mc) = market_cap
        && mc.is_finite() && mc > 1.0
    {
        d.market_cap = Some(mc);
        d.mktcap_log_rank = ((mc.ln() - 18.0) / 10.0).clamp(0.0, 1.0);
    }

    let revenue_ttm = sum_last_n_series(&q.revenue, bar_ts, 4);
    let net_income_ttm = sum_last_n_series(&q.net_income, bar_ts, 4);
    let ebitda_ttm = sum_last_n_series(&q.ebitda, bar_ts, 4);
    let cfo_ttm = sum_last_n_series(&q.operating_cash_flow, bar_ts, 4);
    let capex_ttm = sum_last_n_series(&q.capital_expenditure, bar_ts, 4);

    if has_revenue && revenue_ttm.is_none() {
        tracing::debug!(
            bar_ts = bar_ts,
            rev_count = q.revenue.len(),
            "no revenue_ttm: sum_last_n_series returned None despite data"
        );
    }

    let eps_ttm = sum_last_n_series(&q.eps_diluted, bar_ts, 4).or_else(|| {
        net_income_ttm.zip(shares).and_then(|(ni, sh)| (sh.abs() > 1e-9).then_some(ni / sh))
    });
    if let Some(eps) = eps_ttm
        && eps.is_finite() && eps > 1e-6 && close > 0.0
    {
        let pe = (close / eps).clamp(0.1, 500.0);
        d.trailing_pe = Some(pe);
        tracing::debug!(eps = eps, pe = pe, "computed trailing_pe");
    }

    let earnings_growth = yoy_growth(&q.eps_diluted, bar_ts)
        .or_else(|| yoy_growth(&q.net_income, bar_ts));
    if let Some(g) = earnings_growth
        && g.is_finite()
    {
        d.earnings_growth = Some(g);
        if let Some(pe) = d.trailing_pe {
            let denom = (g.abs() * 100.0).max(1.0);
            d.peg_ratio = Some((pe / denom).clamp(0.0, 10.0));
            let fwd_eps = eps_ttm.unwrap_or(0.0) * (1.0 + g).max(0.1);
            if fwd_eps > 1e-6 && close > 0.0 {
                d.forward_pe = Some((close / fwd_eps).clamp(0.1, 500.0));
            }
        }
    }

    if let Some(g) = yoy_growth(&q.revenue, bar_ts)
        && g.is_finite()
    {
        d.revenue_growth = Some(g);
    }

    if let Some((ni, rev)) = net_income_ttm.zip(revenue_ttm)
        && rev.abs() > 1e-9
    {
        d.profit_margins = Some((ni / rev).clamp(-1.0, 1.0));
    }

    let total_debt = latest_series_value(&q.total_debt, bar_ts);
    let total_equity = latest_series_value(&q.total_equity, bar_ts);
    if let Some((debt, eq)) = total_debt.zip(total_equity)
        && eq.abs() > 1e-9
    {
        d.debt_to_equity = Some((100.0 * debt / eq.abs()).clamp(0.0, 500.0));
    }

    if let Some((ocf, capex, mc)) = cfo_ttm.zip(capex_ttm).zip(market_cap).map(|((a, b), c)| (a, b, c))
        && mc.abs() > 1e-9
    {
        let fcf = ocf + capex;
        d.free_cashflow = Some(fcf);
    }

    if let Some((ni, eq_vals)) = net_income_ttm.zip(Some(latest_n_series_values(&q.total_equity, bar_ts, 4)))
        && !eq_vals.is_empty()
    {
        let avg_eq = eq_vals.iter().sum::<f64>() / eq_vals.len() as f64;
        if avg_eq.abs() > 1e-9 {
            d.return_on_equity = Some((ni / avg_eq).clamp(-2.0, 2.0));
        }
    }

    if let Some((ca, cl)) = latest_series_value(&q.current_assets, bar_ts)
        .zip(latest_series_value(&q.current_liabilities, bar_ts))
        && cl.abs() > 1e-9
    {
        d.current_ratio = Some((ca / cl).clamp(0.0, 10.0));
    }

    if let Some(yld) = trailing_dividend_yield_at(bar_ts, close, div_timestamps, div_amounts) {
        d.dividend_yield = Some(yld);
    }

    if let Some((mc, debt, cash, ebitda)) = market_cap
        .zip(total_debt)
        .zip(latest_series_value(&q.cash_and_equivalents, bar_ts))
        .zip(ebitda_ttm)
        .map(|(((a, b), c), d)| (a, b, c, d))
        && ebitda.abs() > 1e-9
    {
        let ev = (mc + debt - cash).max(1.0);
        d.ev_to_ebitda = Some((ev / ebitda.abs()).clamp(0.0, 200.0));
    }
}

fn encode_fundamental_block(d: &FundamentalData) -> [f32; 30] {
    let p = |v: Option<f64>, scale: f64| -> f32 {
        let x = v.unwrap_or(0.0) / scale;
        (1.0 / (1.0 + (-x).exp())) as f32
    };
    let z = |v: Option<f64>, scale: f64| -> f32 { (v.unwrap_or(0.0) / scale).tanh() as f32 };
    let cl = |v: f32, lo: f32, hi: f32| v.clamp(lo, hi);
    [
        p(d.trailing_pe, 25.0),
        p(d.forward_pe, 20.0),
        p(d.peg_ratio, 2.0),
        p(d.ev_to_ebitda, 12.0),
        cl(d.dividend_yield.unwrap_or(0.02) as f32, 0.0, 0.2),
        cl(d.profit_margins.unwrap_or(0.1) as f32, -1.0, 1.0),
        cl(d.revenue_growth.unwrap_or(0.05) as f32, -1.0, 1.0),
        cl(d.earnings_growth.unwrap_or(0.05) as f32, -1.0, 1.0),
        p(d.debt_to_equity, 100.0),
        z(
            d.free_cashflow
                .zip(d.market_cap)
                .map(|(f, m)| if m > 0.0 { f / m } else { 0.0 }),
            0.1,
        ),
        cl(d.return_on_equity.unwrap_or(0.1) as f32, -1.0, 1.0),
        p(d.current_ratio, 1.5),
        z(d.last_earnings_surprise_pct, 15.0),
        cl(d.earnings_surprise_streak as f32 / 5.0, -1.0, 1.0),
        z(d.next_q_est_revision_30d, 0.3),
        z(d.next_q_est_revision_90d, 0.3),
        z(d.rev_est_revision_30d, 0.3),
        d.days_since_earnings
            .map(|x| (1.0 / (1.0 + x as f32 / 30.0)).clamp(0.0, 1.0))
            .unwrap_or(0.0),
        cl(d.analyst_consensus.unwrap_or(0.0) as f32, -1.0, 1.0),
        cl(d.strong_buy_fraction.unwrap_or(0.2) as f32, 0.0, 1.0),
        z(d.rating_momentum_30d, 1.0),
        cl(d.recent_grade_direction as f32, -1.0, 1.0),
        z(d.net_insider_shares_90d, 1_000_000.0),
        p(Some(d.insider_buy_count_90d), 4.0),
        z(d.net_insider_value_90d, 10_000_000.0),
        cl(d.insider_ownership_pct.unwrap_or(0.05) as f32, 0.0, 1.0),
        cl(d.institutional_pct.unwrap_or(0.6) as f32, 0.0, 1.0),
        z(d.institutional_delta_qtr, 0.1),
        cl(d.sector_idx as f32, 0.0, 1.0),
        cl(d.mktcap_log_rank as f32, 0.0, 1.0),
    ]
}

/// Historical Block-6 (event_state) vector — slot-for-slot identical to the live
/// `EventState::to_block6` (`src/events/state.rs`) so the same model dimension
/// means the same thing in training shards and at inference time. Each event
/// dimension is `magnitude · exp(-Δt / τ)` of the most recent qualifying event,
/// reusing the live per-kind magnitudes and decay constants (`EventKind`):
///   [0] insider buy recency      mag = ln|value|, τ = 1d
///   [1] insider buy magnitude    (live computes this identically to [0])
///   [2] analyst upgrade recency   mag = 1,        τ = 30d
///   [3] analyst downgrade recency mag = 1,        τ = 30d
///   [4] SEC filing recency        8-K (1.5, τ=7d) ∨ other (0.5, τ=30d)
///   [5] 10-K/10-Q recency         mag = 1,        τ = 30d
///   [6] news/event intensity      log-count of dated text events, decayed
///   [7] news sentiment            decay-weighted mean FinBERT score over window
///   [8] earnings proximity        (external, exp(-days/14))
///   [9] ex-dividend proximity     (external)
fn compute_event_state_at(
    bar_ts: i64,
    exdiv_proximity: f32,
    insider_events: &[InsiderEvent],
    grade_events: &[GradeEvent],
    sec_filing_events: &[SecFilingEvent],
    earnings_events: &[EarningsEvent],
    next_earnings_ts: Option<i64>,
    text_events: &[TextEvent],
) -> [f32; 10] {
    let mut out = [0.0f32; 10];
    let day = 86_400i64;

    // Most-recent qualifying event → `mag · exp(-Δdays / τ_days)`, else 0.
    let recency = |hit: Option<(i64, f32)>, tau_days: f32| -> f32 {
        match hit {
            Some((ts, mag)) if ts <= bar_ts => {
                let days = ((bar_ts - ts) as f32 / day as f32).max(0.0);
                mag * (-days / tau_days).exp()
            }
            _ => 0.0,
        }
    };
    let latest = |pred: &dyn Fn(&&InsiderEvent) -> bool, mag: &dyn Fn(&InsiderEvent) -> f32| {
        insider_events
            .iter()
            .rev()
            .find(|e| e.ts <= bar_ts && pred(e))
            .map(|e| (e.ts, mag(e)))
    };

    // [0]/[1] insider buy: live magnitude is ln|value|; the two slots are equal.
    let buy = recency(
        latest(&|e| e.is_buy == Some(true), &|e| (e.value.abs() as f64).ln().max(0.0) as f32),
        1.0,
    );
    out[0] = buy;
    out[1] = buy;

    // [2]/[3] analyst up/down-grade recency (magnitude 1.0, τ = 30d).
    let last_up = grade_events
        .iter()
        .rev()
        .find(|g| g.ts <= bar_ts && g.direction > 0.0)
        .map(|g| (g.ts, 1.0));
    let last_down = grade_events
        .iter()
        .rev()
        .find(|g| g.ts <= bar_ts && g.direction < 0.0)
        .map(|g| (g.ts, 1.0));
    out[2] = recency(last_up, 30.0);
    out[3] = recency(last_down, 30.0);

    // [4] SEC filing recency: 8-K (high-freq) vs other; [5] 10-K/10-Q.
    let last_sec = |pred: &dyn Fn(&SecFilingEvent) -> bool, mag: f32| {
        sec_filing_events
            .iter()
            .rev()
            .find(|f| f.ts <= bar_ts && pred(f))
            .map(|f| (f.ts, mag))
    };
    let recency_8k = recency(last_sec(&|f| f.is_8k, 1.5), 7.0);
    let recency_other = recency(last_sec(&|f| !f.is_8k && !f.is_10k && !f.is_10q, 0.5), 30.0);
    out[4] = recency_8k.max(recency_other);
    let recency_10k = recency(last_sec(&|f| f.is_10k, 1.0), 30.0);
    let recency_10q = recency(last_sec(&|f| f.is_10q, 1.0), 30.0);
    out[5] = recency_10k.max(recency_10q);

    // [6]/[7] text-derived intensity + sentiment over a trailing window. Live
    // draws these from realtime news; historically the dated EDGAR-filing (∪
    // FNSPID) stream fills the same slots through the same FinBERT scorer, so the
    // slot meaning is shared even though the source cadence differs (the values
    // are z-scored downstream, normalizing the cadence gap).
    const TEXT_TAU_DAYS: f32 = 7.0;
    let window = 30 * day;
    let mut intensity = 0.0f32;
    let mut sent_sum = 0.0f32;
    let mut sent_w = 0.0f32;
    for e in text_events.iter().filter(|e| e.ts <= bar_ts && e.ts >= bar_ts - window) {
        let days = ((bar_ts - e.ts) as f32 / day as f32).max(0.0);
        let w = (-days / TEXT_TAU_DAYS).exp();
        intensity += w;
        sent_sum += w * e.sentiment;
        sent_w += w;
    }
    out[6] = intensity.ln_1p();
    out[7] = if sent_w > 0.0 {
        (sent_sum / sent_w).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // [8] earnings proximity (nearest real or real-anchored quarterly projection).
    out[8] = earnings_proximity(bar_ts, earnings_events, next_earnings_ts).clamp(0.0, 1.0);
    // [9] ex-dividend proximity (computed by the caller from the dividend series).
    out[9] = exdiv_proximity.clamp(0.0, 4.0);
    out
}

/// Block 8 text vector.
///
/// Pools pre-computed 1024-dim harrier-oss embeddings of every TextEvent in the
/// trailing 90-day window with `weight = exp(-Δdays / 30)`, L2-normalizes the
/// pooled vector, then reduces it to the 64-dim Block-8 width via the fixed
/// `text_pca` basis (harrier-oss is not a Matryoshka model, so the 1024-d vector
/// cannot simply be truncated) and re-normalizes onto the unit sphere.
/// (Block 6 news sentiment is the FinBERT score on each event, pooled separately
/// in `compute_event_state_at` — not derived from this embedding.)
fn compute_text_embedding_at(
    bar_ts: i64,
    events: &[TextEvent],
    embed_map: &HashMap<String, Vec<f32>>,
    pca: &PcaModel,
) -> [f32; text_pca::TEXT_PCA_DIM] {
    let mut acc = vec![0.0f32; EMBED_DIM];
    let mut wsum = 0.0f32;
    let cutoff = bar_ts - TEXT_WINDOW_SECS;
    for ev in events
        .iter()
        .filter(|e| e.ts <= bar_ts && e.ts >= cutoff)
    {
        let Some(emb) = embed_map.get(&ev.text) else {
            continue;
        };
        if emb.len() != EMBED_DIM {
            continue;
        }
        let age_days = ((bar_ts - ev.ts).max(0) as f32) / 86_400.0;
        let w = (-age_days / TEXT_DECAY_DAYS).exp();
        if w < 1e-4 {
            continue;
        }
        for (a, &e) in acc.iter_mut().zip(emb.iter()) {
            *a += w * e;
        }
        wsum += w;
    }
    let mut out = [0.0f32; text_pca::TEXT_PCA_DIM];
    if wsum <= 0.0 {
        return out;
    }
    for a in &mut acc {
        *a /= wsum;
    }
    l2_normalize_inplace(&mut acc);

    // Reduce 1024 → 64 through the learned PCA basis, then re-normalize.
    pca.project_into(&acc, &mut out);
    let n: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in &mut out {
            *x /= n;
        }
    }
    out
}


fn l2_normalize_inplace(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v {
            *x /= n;
        }
    }
}

fn annualized_vol(xs: &[f32]) -> Option<f32> {
    if xs.len() < 2 {
        return None;
    }
    let m = xs.iter().sum::<f32>() / xs.len() as f32;
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (xs.len() as f32 - 1.0);
    Some(var.max(0.0).sqrt() * (252.0f32).sqrt())
}

fn mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then_some(xs.iter().sum::<f32>() / xs.len() as f32)
}

fn cycle_signal(ts: i64, period_days: f64) -> f64 {
    let phase = (ts as f64 / 86_400.0) / period_days;
    (std::f64::consts::TAU * phase).sin()
}

fn json_at<'a>(v: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut cur = v;
    for p in path {
        if let Ok(i) = p.parse::<usize>() {
            cur = cur.get(i)?;
        } else {
            cur = cur.get(*p)?;
        }
    }
    Some(cur)
}

fn json_f64_path(v: &serde_json::Value, path: &[&str]) -> Option<f64> {
    json_at(v, path).and_then(|x| x.as_f64())
}

fn json_i64_path(v: &serde_json::Value, path: &[&str]) -> Option<i64> {
    json_at(v, path).and_then(|x| x.as_i64())
}

fn market_hours_state(ts: i64) -> f32 {
    let Some(dt_utc) = chrono::DateTime::from_timestamp(ts, 0) else {
        return 0.0;
    };
    let weekday = dt_utc.weekday().num_days_from_monday();
    if weekday >= 5 {
        return 0.0;
    }
    let h = dt_utc.hour() as i32;
    let m = dt_utc.minute() as i32;
    let minute_of_day = h * 60 + m;
    let open = 14 * 60 + 30; // Approximate NYSE open in UTC
    let close = 21 * 60; // Approximate NYSE close in UTC
    if minute_of_day < open - 120 {
        0.1
    } else if minute_of_day < open {
        0.35
    } else if minute_of_day <= close {
        1.0
    } else if minute_of_day <= close + 120 {
        0.4
    } else {
        0.15
    }
}

fn minutes_since_market_event(ts: i64) -> f32 {
    let Some(dt_utc) = chrono::DateTime::from_timestamp(ts, 0) else {
        return 0.5;
    };
    let h = dt_utc.hour() as i32;
    let m = dt_utc.minute() as i32;
    let minute_of_day = h * 60 + m;
    let open = 14 * 60 + 30;
    let close = 21 * 60;
    let d_open = (minute_of_day - open).abs() as f32 / 390.0;
    let d_close = (minute_of_day - close).abs() as f32 / 390.0;
    // Use inverse-distance rather than `1 - distance` so daily bars at midnight
    // (far from any market event) produce a meaningful non-zero value (~0.3)
    // instead of clamping to 0.
    let min_dist = d_open.min(d_close);
    1.0 / (1.0 + min_dist)
}

fn parse_ymd_to_ts(s: &str) -> Option<i64> {
    let d = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
    Some(d.and_hms_opt(0, 0, 0)?.and_utc().timestamp())
}

fn sector_to_idx(sector: &str) -> f64 {
    let s = sector.to_ascii_lowercase();
    let idx = if s.contains("technology") {
        0.05
    } else if s.contains("communication") {
        0.15
    } else if s.contains("consumer cyclical") {
        0.25
    } else if s.contains("consumer defensive") {
        0.35
    } else if s.contains("industrials") {
        0.45
    } else if s.contains("financial") {
        0.55
    } else if s.contains("health") {
        0.65
    } else if s.contains("energy") {
        0.75
    } else if s.contains("materials") {
        0.85
    } else {
        0.95
    };
    idx
}

fn simple_symbol_hash(symbol: &str) -> f64 {
    let mut h = 2166136261u32;
    for b in symbol.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    (h as f64 / u32::MAX as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        EarningsEvent, MarketFeatureLookup, argmax_min, build_close_series, build_return_series,
        earnings_proximity, stddev,
    };
    use crate::store::Bar;
    use serde_json::json;
    use std::collections::BTreeMap;

    const DAY: i64 = 86_400;

    #[test]
    fn earnings_proximity_zero_without_dates() {
        assert_eq!(earnings_proximity(0, &[], None), 0.0);
    }

    #[test]
    fn earnings_proximity_peaks_on_report_date() {
        let ev = [EarningsEvent { ts: 1000 * DAY, surprise_pct: 0.0 }];
        let on = earnings_proximity(1000 * DAY, &ev, None);
        let off = earnings_proximity(1014 * DAY, &ev, None); // +14d → exp(-1)
        assert!((on - 1.0).abs() < 1e-6);
        assert!(off < on && off > 0.0);
    }

    #[test]
    fn earnings_proximity_projects_quarterly_into_gap() {
        // Two real dates one quarter apart; a bar far before them should still
        // see a nearby projected report date (real-anchored cadence, not zero).
        let ev = [
            EarningsEvent { ts: 1000 * DAY, surprise_pct: 0.0 },
            EarningsEvent { ts: 1091 * DAY, surprise_pct: 0.0 },
        ];
        let far_past = earnings_proximity(727 * DAY, &ev, None); // ~3 quarters before, on grid
        assert!(far_past > 0.9, "projected on-grid bar should be high, got {far_past}");
    }

    // Real-symbol verification: fetches a live symbol from finance-query, runs it
    // through the production `build_fundamental_static` path, and asserts the
    // fundamental fields populate as expected — guarding the forward_pe wrong-module
    // regression and any future field/casing drift. Network-gated:
    //   cargo test --lib historical::batch::tests::fundamentals_collected_for_real_symbol -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "hits finance-query network"]
    async fn fundamentals_collected_for_real_symbol() {
        use finance_query::Ticker;
        let sym = "NVDA"; // mega-cap with a full fundamentals surface
        let ticker = Ticker::new(sym).await.expect("Ticker::new");

        macro_rules! j {
            ($e:expr) => {
                $e.await.ok().flatten().as_ref().and_then(|x| serde_json::to_value(x).ok())
            };
        }
        let summary_json = j!(ticker.summary_detail());
        let financial_json = j!(ticker.financial_data());
        let key_stats_json = j!(ticker.key_stats());
        let profile_json = j!(ticker.asset_profile());
        let rec_json = j!(ticker.recommendation_trend());
        let et_json = j!(ticker.earnings_trend());
        let cal_json = j!(ticker.calendar_events());
        let mh_json = j!(ticker.major_holders());

        let f = super::build_fundamental_static(
            sym,
            &summary_json,
            &financial_json,
            &key_stats_json,
            &profile_json,
            &rec_json,
            &et_json,
            &cal_json,
            &mh_json,
        );

        eprintln!("FundamentalStatic[{sym}]:");
        eprintln!("  trailing_pe={:?} forward_pe={:?} peg_ratio={:?} ev_to_ebitda={:?}",
            f.trailing_pe, f.forward_pe, f.peg_ratio, f.ev_to_ebitda);
        eprintln!("  dividend_yield={:?} profit_margins={:?} revenue_growth={:?} earnings_growth={:?}",
            f.dividend_yield, f.profit_margins, f.revenue_growth, f.earnings_growth);
        eprintln!("  debt_to_equity={:?} free_cashflow={:?} market_cap={:?} return_on_equity={:?} current_ratio={:?}",
            f.debt_to_equity, f.free_cashflow, f.market_cap, f.return_on_equity, f.current_ratio);
        eprintln!("  analyst_consensus={:?} strong_buy_fraction={:?} sector_idx={} mktcap_log_rank={} next_earnings_ts={:?}",
            f.analyst_consensus, f.strong_buy_fraction, f.sector_idx, f.mktcap_log_rank, f.next_earnings_ts);

        // The regression that motivated this test: forward_pe was always None.
        let fpe = f.forward_pe.expect("forward_pe must be populated (regression: was read from wrong module)");
        assert!(fpe > 0.0 && fpe < 1000.0, "forward_pe out of sane range: {fpe}");
        // Other core fundamentals that must come through for a mega-cap.
        assert!(f.market_cap.is_some_and(|m| m > 1e11), "market_cap missing/implausible: {:?}", f.market_cap);
        assert!(f.profit_margins.is_some(), "profit_margins missing");
        assert!(f.debt_to_equity.is_some(), "debt_to_equity missing");
        assert!(f.earnings_growth.is_some(), "earnings_growth missing");
        assert!(f.ev_to_ebitda.is_some(), "ev_to_ebitda missing");
        assert!(f.dividend_yield.is_some(), "dividend_yield missing");
    }

    #[test]
    fn return_series_uses_log_returns() {
        let bars = vec![
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 100.0,
                volume: 1,
                timestamp: 10,
            },
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 110.0,
                volume: 1,
                timestamp: 20,
            },
        ];
        let series = build_return_series(&bars);
        let r = series.get(&20).copied().unwrap_or_default();
        assert!(r > 0.09 && r < 0.11);
    }

    #[test]
    fn close_series_aligns_timestamps() {
        let bars = vec![
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 42.0,
                volume: 1,
                timestamp: 11,
            },
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 43.0,
                volume: 1,
                timestamp: 12,
            },
        ];
        let series = build_close_series(&bars);
        assert_eq!(series.get(&11).copied().unwrap_or_default(), 42.0);
        assert_eq!(series.get(&12).copied().unwrap_or_default(), 43.0);
    }

    #[test]
    fn stddev_and_extrema_helpers_are_stable() {
        let d = stddev([0.0, 1.0, 2.0, 3.0]);
        assert!(d > 1.11 && d < 1.13);
        assert_eq!(argmax_min([0.1, -2.0, 0.3, 0.2]), (2, 1));
    }

    #[test]
    fn market_vector_populates_proxy_slots() {
        let mut lookup = MarketFeatureLookup::default();
        for key in [
            "spx", "nasdaq", "russell", "dow", "dxy", "gold", "oil", "btc", "eth", "vix",
        ] {
            let mut series = BTreeMap::new();
            series.insert(100, 0.01);
            lookup.returns.insert(key, series);
        }

        let mut vix_close = BTreeMap::new();
        vix_close.insert(100, 20.0);
        lookup.closes.insert("vix", vix_close);

        let mut btc_close = BTreeMap::new();
        btc_close.insert(100, 90_000.0);
        lookup.closes.insert("btc", btc_close);

        let mut eth_close = BTreeMap::new();
        eth_close.insert(100, 3_000.0);
        lookup.closes.insert("eth", eth_close);

        let out = lookup.vector_at(100);
        assert!(out[52] > 0.0); // SPX proxy return
        assert!(out[61] > 0.0); // DXY proxy return
        assert!(out[66] > 0.0); // BTC level proxy
        assert!(out[68] > 0.0); // ETH level proxy
        assert!(out[72] > 0.0); // breadth proxy
    }

    #[test]
    fn extract_insider_events_marks_ambiguous_direction_unknown() {
        let payload = Some(json!({
            "transactions": [{
                "startDate": { "raw": 1_700_000_000 },
                "shares": { "raw": 1200 },
                "value": null,
                "transactionText": "",
                "moneyText": ""
            }]
        }));
        let events = super::extract_insider_events(&payload);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].is_buy, None);
    }

    /// Post-H5 contract: every `event_state` slot derives from a *real* dated
    /// event or is honestly zero — no synthetic/cyclical proxy fills in. A real
    /// analyst downgrade drives its recency slot, while an unlabeled insider
    /// transaction (no buy/sell direction) and the absence of any SEC filing
    /// leave their slots at exactly 0 rather than fabricating signal.
    #[test]
    fn event_state_uses_real_events_only_no_proxy_fill() {
        let bar_ts = 1_800_000_000i64;
        let insider_events = vec![super::InsiderEvent {
            ts: bar_ts - 2 * 86_400,
            shares: 2_000.0,
            value: 0.0,
            is_buy: None, // unlabeled → must NOT count as a buy
        }];
        let grade_events = vec![super::GradeEvent {
            ts: bar_ts - 10 * 86_400,
            direction: -1.0, // real downgrade 10d ago
        }];
        let out = super::compute_event_state_at(
            bar_ts,
            0.0,
            &insider_events,
            &grade_events,
            &[], // no SEC filings
            &[],
            None,
            &[],
        );
        // [3] analyst-downgrade recency: real event, exp(-10/30) ≈ 0.72.
        assert!(out[3] > 0.0, "real downgrade must drive its recency slot");
        // [0]/[1] insider-buy: unlabeled transaction → genuine zero (no proxy).
        assert_eq!(out[0], 0.0, "unlabeled insider txn must not fabricate a buy");
        assert_eq!(out[1], 0.0);
        // [4]/[5] SEC filing recency: no filings → genuine zero (no proxy).
        assert_eq!(out[4], 0.0, "absent SEC filings must yield zero, not a proxy");
        assert_eq!(out[5], 0.0);
    }
}
