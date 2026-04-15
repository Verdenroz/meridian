use crate::events::detector::detect_options_unusual;
use crate::store::{CachedValue, DataStore, OptionsData};
use finance_query::Ticker;
use std::sync::Arc;

pub async fn poll_options_chains(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let symbols = store.symbols.clone();
    let concurrency = 10usize;

    for chunk in symbols.chunks(concurrency) {
        let tasks: Vec<_> = chunk
            .iter()
            .map(|sym| {
                let sym = sym.clone();
                let store = store.clone();
                tokio::spawn(async move {
                    poll_one_options(&sym, &store).await;
                })
            })
            .collect();

        for task in tasks {
            let _ = task.await;
        }
    }

    tracing::debug!("Options chains updated for {} symbols", store.symbols.len());
    Ok(())
}

async fn poll_one_options(symbol: &str, store: &Arc<DataStore>) {
    let ticker = match Ticker::new(symbol).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(symbol = %symbol, err = %e, "Ticker::new failed in options poll");
            return;
        }
    };

    // options() takes Option<i64> date; None = nearest expiry
    match ticker.options(None).await {
        Ok(opts) => {
            let calls = opts.calls().into_iter().collect::<Vec<_>>();
            let puts = opts.puts().into_iter().collect::<Vec<_>>();

            let call_vol: f64 = calls
                .iter()
                .filter_map(|c| c.volume)
                .map(|v| v as f64)
                .sum();
            let put_vol: f64 = puts.iter().filter_map(|c| c.volume).map(|v| v as f64).sum();
            let call_oi: f64 = calls
                .iter()
                .filter_map(|c| c.open_interest)
                .map(|v| v as f64)
                .sum();
            let put_oi: f64 = puts
                .iter()
                .filter_map(|c| c.open_interest)
                .map(|v| v as f64)
                .sum();
            let total_vol = call_vol + put_vol;
            let total_oi = call_oi + put_oi;

            let vol_mean = store.rolling_stats.get(symbol, "opt_total_vol").mean;
            let vol_std = store.rolling_stats.get(symbol, "opt_total_vol").std_dev();
            store
                .rolling_stats
                .update(symbol, "opt_total_vol", total_vol);

            if let Some(event) =
                detect_options_unusual(symbol, total_vol, total_oi, vol_mean, vol_std)
            {
                let mut es = store.event_state.entry(symbol.to_string()).or_default();
                *es = es.with_event(&event);
                tracing::info!(symbol = %symbol, "OptionsUnusualActivity event");
            }

            let data = OptionsData { calls, puts };
            store
                .options
                .insert(symbol.to_string(), CachedValue::new(data));
        }
        Err(e) => {
            tracing::debug!(symbol = %symbol, err = %e, "options poll skipped (no options or error)");
        }
    }
}
