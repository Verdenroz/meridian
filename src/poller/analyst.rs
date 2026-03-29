use crate::events::detector::{detect_analyst_events, detect_rating_change};
use crate::store::{AnalystData, CachedValue, DataStore, OurGradeChange, OurRecoPeriod};
use finance_query::Ticker;
use std::sync::Arc;

pub async fn poll_analyst_ratings(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let symbols = store.symbols.clone();

    for chunk in symbols.chunks(5) {
        let tasks: Vec<_> = chunk
            .iter()
            .map(|sym| {
                let sym = sym.clone();
                let store = store.clone();
                tokio::spawn(async move { poll_one_analyst(&sym, &store).await })
            })
            .collect();
        for t in tasks { let _ = t.await; }
    }

    tracing::debug!("Analyst ratings updated");
    Ok(())
}

async fn poll_one_analyst(symbol: &str, store: &Arc<DataStore>) {
    let ticker = match Ticker::new(symbol).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "Ticker::new failed in analyst poll");
            return;
        }
    };

    let grading = match ticker.grading_history().await {
        Ok(Some(h)) => h.history,
        Ok(None) => vec![],
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "grading_history failed");
            return;
        }
    };

    // Convert finance-query GradeChange to our own type (avoids naming private type)
    let our_grading: Vec<OurGradeChange> = grading
        .iter()
        .map(|g| OurGradeChange {
            firm: g.firm.clone(),
            from_grade: g.from_grade.clone(),
            to_grade: g.to_grade.clone(),
            action: g.action.clone(),
            date: g.epoch_grade_date.map(|ts| {
                chrono::DateTime::from_timestamp(ts, 0)
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default()
            }),
        })
        .collect();

    let rec_period: Option<OurRecoPeriod> = ticker
        .recommendation_trend()
        .await
        .ok()
        .flatten()
        .and_then(|rt| rt.trend.into_iter().next())
        .map(|p| OurRecoPeriod {
            strong_buy: p.strong_buy.unwrap_or(0),
            buy: p.buy.unwrap_or(0),
            hold: p.hold.unwrap_or(0),
            sell: p.sell.unwrap_or(0),
            strong_sell: p.strong_sell.unwrap_or(0),
        });

    // Detect analyst upgrade/downgrade events
    let old_grading = store
        .analyst
        .get(symbol)
        .map(|c| c.value.grading_history.clone())
        .unwrap_or_default();

    let events = detect_analyst_events(symbol, &old_grading, &our_grading);
    for event in events {
        tracing::info!(symbol = %symbol, kind = ?event.kind, "analyst event");
        let mut es = store.event_state.entry(symbol.to_string()).or_default();
        *es = es.with_event(&event);
    }

    // Detect consensus rating change
    let old_consensus = compute_consensus(
        store.analyst.get(symbol).as_ref().and_then(|c| c.value.recommendation_period.as_ref()),
    );
    let new_consensus = compute_consensus(rec_period.as_ref());
    if let Some(event) = detect_rating_change(symbol, old_consensus, new_consensus) {
        tracing::info!(symbol = %symbol, "RatingChange event");
        let mut es = store.event_state.entry(symbol.to_string()).or_default();
        *es = es.with_event(&event);
    }

    let data = AnalystData { grading_history: our_grading, recommendation_period: rec_period };
    store.analyst.insert(symbol.to_string(), CachedValue::new(data));
}

fn compute_consensus(period: Option<&OurRecoPeriod>) -> f64 {
    let Some(p) = period else { return 0.0 };
    let sb = p.strong_buy as f64;
    let b = p.buy as f64;
    let s = p.sell as f64;
    let ss = p.strong_sell as f64;
    let total = (sb + b + p.hold as f64 + s + ss).max(1.0);
    (sb * 2.0 + b - s - ss * 2.0) / total
}
