use crate::events::detector::detect_insider_events;
use crate::store::{CachedValue, DataStore, OurInsiderTx};
use finance_query::Ticker;
use std::sync::Arc;

pub async fn poll_insider_tx(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let symbols = store.symbols.clone();

    for chunk in symbols.chunks(5) {
        let tasks: Vec<_> = chunk
            .iter()
            .map(|sym| {
                let sym = sym.clone();
                let store = store.clone();
                tokio::spawn(async move { poll_one_insider(&sym, &store).await })
            })
            .collect();
        for t in tasks { let _ = t.await; }
    }

    tracing::debug!("Insider transactions updated");
    Ok(())
}

async fn poll_one_insider(symbol: &str, store: &Arc<DataStore>) {
    let ticker = match Ticker::new(symbol).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "Ticker::new failed in insider poll");
            return;
        }
    };

    // Returns Result<Option<InsiderTransactions>> — private type, use type inference
    let new_txs: Vec<OurInsiderTx> = match ticker.insider_transactions().await {
        Ok(Some(it)) => it.transactions.iter().map(|t| OurInsiderTx {
            filer_name: t.filer_name.clone(),
            transaction_text: t.transaction_text.clone(),
            start_date_ts: t.start_date.as_ref().and_then(|v| v.raw),
            shares: t.shares.as_ref().and_then(|v| v.raw),
            value: t.value.as_ref().and_then(|v| v.raw),
        }).collect(),
        Ok(None) => vec![],
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "insider_transactions poll failed");
            return;
        }
    };

    let old_txs = store
        .insider_tx
        .get(symbol)
        .map(|c| c.value.clone())
        .unwrap_or_default();

    let events = detect_insider_events(symbol, &old_txs, &new_txs);
    for event in events {
        tracing::info!(symbol = %symbol, kind = ?event.kind, "insider event");
        let mut es = store.event_state.entry(symbol.to_string()).or_default();
        *es = es.with_event(&event);
    }

    store.insider_tx.insert(symbol.to_string(), CachedValue::new(new_txs));
}
