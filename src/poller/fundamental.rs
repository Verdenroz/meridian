use crate::store::{CachedValue, DataStore, FundamentalData};
use finance_query::Ticker;
use std::sync::Arc;

/// Poll fundamentals for all symbols (runs quarterly; triggered manually or at startup).
pub async fn poll_fundamentals(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let symbols = store.symbols.clone();

    for chunk in symbols.chunks(3) {
        let tasks: Vec<_> = chunk
            .iter()
            .map(|sym| {
                let sym = sym.clone();
                let store = store.clone();
                tokio::spawn(async move { poll_one_fundamental(&sym, &store).await })
            })
            .collect();
        for t in tasks { let _ = t.await; }
        // Small pause to avoid rate limiting on quarterly batch
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    tracing::info!("Fundamentals updated for {} symbols", store.symbols.len());
    Ok(())
}

async fn poll_one_fundamental(symbol: &str, store: &Arc<DataStore>) {
    let ticker = match Ticker::new(symbol).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "Ticker::new failed in fundamental poll");
            return;
        }
    };

    let mut d = FundamentalData::default();

    // ── Financial fundamentals ─────────────────────────────────────────────
    if let Ok(Some(fd)) = ticker.financial_data().await {
        d.profit_margins = fd.profit_margins.as_ref().and_then(|v| v.raw);
        d.revenue_growth = fd.revenue_growth.as_ref().and_then(|v| v.raw);
        d.earnings_growth = fd.earnings_growth.as_ref().and_then(|v| v.raw);
        d.debt_to_equity = fd.debt_to_equity.as_ref().and_then(|v| v.raw);
        d.free_cashflow = fd.free_cashflow.as_ref().and_then(|v| v.raw).map(|v| v as f64);
        d.return_on_equity = fd.return_on_equity.as_ref().and_then(|v| v.raw);
        d.current_ratio = fd.current_ratio.as_ref().and_then(|v| v.raw);
    }

    if let Ok(Some(ks)) = ticker.key_stats().await {
        d.forward_pe = ks.forward_pe.as_ref().and_then(|v| v.raw);
        d.ev_to_ebitda = ks.enterprise_to_ebitda.as_ref().and_then(|v| v.raw);
    }

    if let Ok(Some(sd)) = ticker.summary_detail().await {
        d.trailing_pe = sd.trailing_pe.as_ref().and_then(|v| v.raw);
        d.market_cap = sd.market_cap.as_ref().and_then(|v| v.raw).map(|v| v as f64);
        d.dividend_yield = sd.dividend_yield.as_ref().and_then(|v| v.raw);
    }

    // ── Earnings intelligence ──────────────────────────────────────────────
    if let Ok(Some(hist)) = ticker.earnings_history().await {
        if let Some(latest) = hist.history.last() {
            let actual = latest.eps_actual.as_ref().and_then(|v| v.raw).unwrap_or(0.0);
            let est = latest.eps_estimate.as_ref().and_then(|v| v.raw).unwrap_or(actual);
            d.last_earnings_surprise_pct = Some(
                if est.abs() > 1e-8 { (actual - est) / est.abs() } else { 0.0 }
            );
            let streak: i32 = hist
                .history
                .iter()
                .rev()
                .take(5)
                .map(|e| {
                    let a = e.eps_actual.as_ref().and_then(|v| v.raw).unwrap_or(0.0);
                    let e2 = e.eps_estimate.as_ref().and_then(|v| v.raw).unwrap_or(a);
                    if (a - e2).abs() < 1e-8 { 0 } else if a > e2 { 1 } else { -1 }
                })
                .scan(0i32, |acc, x| {
                    if *acc == 0 || (*acc > 0 && x > 0) || (*acc < 0 && x < 0) {
                        *acc += x;
                        Some(*acc)
                    } else {
                        None
                    }
                })
                .last()
                .unwrap_or(0);
            d.earnings_surprise_streak = streak;
        }
    }

    if let Ok(Some(et)) = ticker.earnings_trend().await {
        if let Some(trend) = et.trend.first() {
            d.next_q_est_revision_30d = trend.eps_revisions.as_ref().and_then(|r| {
                let up = r.up_last30days.as_ref().and_then(|v| v.raw)?;
                let dn = r.down_last30days.as_ref().and_then(|v| v.raw)?;
                Some((up - dn) as f64)
            });
            d.rev_est_revision_30d = trend
                .revenue_estimate
                .as_ref()
                .and_then(|r| r.growth.as_ref().and_then(|v| v.raw));
        }
    }

    // ── Analyst & ratings ─────────────────────────────────────────────────
    if let Ok(Some(rt)) = ticker.recommendation_trend().await {
        if let Some(period) = rt.trend.first() {
            let sb = period.strong_buy.unwrap_or(0) as f64;
            let b = period.buy.unwrap_or(0) as f64;
            let h = period.hold.unwrap_or(0) as f64;
            let s = period.sell.unwrap_or(0) as f64;
            let ss = period.strong_sell.unwrap_or(0) as f64;
            let total = (sb + b + h + s + ss).max(1.0);
            d.analyst_consensus = Some((sb * 2.0 + b - s - ss * 2.0) / total);
            d.strong_buy_fraction = Some(sb / total);
        }
    }

    if let Ok(Some(gh)) = ticker.grading_history().await {
        let cutoff_ts = (chrono::Utc::now() - chrono::Duration::days(30)).timestamp();
        let recent: Vec<_> = gh
            .history
            .iter()
            .filter(|g| g.epoch_grade_date.map(|ts| ts > cutoff_ts).unwrap_or(false))
            .collect();
        let upgrades = recent
            .iter()
            .filter(|g| g.action.as_deref() == Some("up"))
            .count() as f64;
        let downgrades = recent
            .iter()
            .filter(|g| g.action.as_deref() == Some("down"))
            .count() as f64;
        d.rating_momentum_30d = Some(upgrades - downgrades);

        if let Some(latest) = gh.history.first() {
            let action = latest.action.as_deref().unwrap_or("");
            d.recent_grade_direction = if action == "up" { 1.0 } else if action == "down" { -1.0 } else { 0.0 };
        }
    }

    // ── Ownership & insider ───────────────────────────────────────────────
    if let Ok(Some(it)) = ticker.insider_transactions().await {
        let cutoff_ts = (chrono::Utc::now() - chrono::Duration::days(90)).timestamp();
        let recent: Vec<_> = it
            .transactions
            .iter()
            .filter(|t| {
                t.start_date
                    .as_ref()
                    .and_then(|v| v.raw)
                    .map(|ts| ts > cutoff_ts)
                    .unwrap_or(false)
            })
            .collect();
        let buys: Vec<_> = recent
            .iter()
            .filter(|t| {
                t.transaction_text
                    .as_deref()
                    .map(|s| s.to_lowercase().contains("purchase"))
                    .unwrap_or(false)
            })
            .collect();
        let sells: Vec<_> = recent
            .iter()
            .filter(|t| {
                !t.transaction_text
                    .as_deref()
                    .map(|s| s.to_lowercase().contains("purchase"))
                    .unwrap_or(false)
            })
            .collect();
        d.net_insider_shares_90d = Some(
            buys.iter()
                .filter_map(|t| t.shares.as_ref().and_then(|v| v.raw))
                .map(|s| s as f64)
                .sum::<f64>()
                - sells
                    .iter()
                    .filter_map(|t| t.shares.as_ref().and_then(|v| v.raw))
                    .map(|s| s as f64)
                    .sum::<f64>(),
        );
        d.insider_buy_count_90d = buys.len() as f64 + 1.0;
        d.net_insider_value_90d = Some(
            buys.iter()
                .filter_map(|t| t.value.as_ref().and_then(|v| v.raw))
                .map(|s| s as f64)
                .sum::<f64>()
                - sells
                    .iter()
                    .filter_map(|t| t.value.as_ref().and_then(|v| v.raw))
                    .map(|s| s as f64)
                    .sum::<f64>(),
        );
    }

    if let Ok(Some(mh)) = ticker.major_holders().await {
        d.insider_ownership_pct = mh.insiders_percent_held.as_ref().and_then(|v| v.raw);
        d.institutional_pct = mh.institutions_percent_held.as_ref().and_then(|v| v.raw);
    }

    if let Ok(Some(io)) = ticker.institution_ownership().await {
        let top10_current: f64 = io
            .ownership_list
            .iter()
            .take(10)
            .filter_map(|h| h.position.as_ref().and_then(|v| v.raw))
            .map(|s| s as f64)
            .sum();
        let prev = store
            .fundamentals
            .get(symbol)
            .and_then(|f| f.value.institutional_delta_qtr)
            .unwrap_or(0.0);
        d.institutional_delta_qtr = Some(top10_current - prev.abs() * 1e6);
    }

    if let Ok(Some(cal)) = ticker.calendar_events().await {
        if let Some(ts) = cal.next_earnings_date() {
            let today_ts = chrono::Utc::now().timestamp();
            d.days_since_earnings = Some(((today_ts - ts).abs() / 86400) as f64);
        }
    }

    d.sector_idx = 0.0;
    d.mktcap_log_rank = d
        .market_cap
        .map(|m| if m > 0.0 { (m.ln() / 30.0).clamp(0.0, 1.0) } else { 0.0 })
        .unwrap_or(0.5);

    store.fundamentals.insert(symbol.to_string(), CachedValue::new(d));
}
