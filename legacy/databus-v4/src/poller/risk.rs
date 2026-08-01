use crate::store::{CachedValue, DataStore, RiskData};
use finance_query::{Interval, Ticker, TimeRange};
use std::sync::Arc;

pub async fn poll_risk_analytics(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let symbols = store.symbols_snapshot();

    for chunk in symbols.chunks(5) {
        let tasks: Vec<_> = chunk
            .iter()
            .map(|sym| {
                let sym = sym.clone();
                let store = store.clone();
                tokio::spawn(async move { poll_one_risk(&sym, &store).await })
            })
            .collect();
        for t in tasks {
            let _ = t.await;
        }
    }

    tracing::debug!("Risk analytics updated");
    Ok(())
}

async fn poll_one_risk(symbol: &str, store: &Arc<DataStore>) {
    let ticker = match Ticker::new(symbol).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "Ticker::new failed in risk poll");
            return;
        }
    };

    match ticker
        .risk(Interval::OneDay, TimeRange::OneYear, None)
        .await
    {
        Ok(metrics) => {
            let data = RiskData {
                var_95: metrics.var_95,
                var_99: metrics.var_99,
                parametric_var_95: metrics.parametric_var_95,
                sharpe: metrics.sharpe.unwrap_or(0.0),
                sortino: metrics.sortino.unwrap_or(0.0),
                beta: metrics.beta.unwrap_or(1.0),
                max_drawdown: metrics.max_drawdown.abs(),
                max_dd_recovery: metrics.max_drawdown_recovery_periods.unwrap_or(1) as f64,
            };
            store
                .risk
                .insert(symbol.to_string(), CachedValue::new(data));
        }
        Err(e) => tracing::warn!(symbol = %symbol, err = %e, "risk poll failed"),
    }
}
