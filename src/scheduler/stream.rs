use crate::store::DataStore;
use finance_query::streaming::PriceStream;
use std::sync::Arc;
use tokio_stream::StreamExt;

/// Drive the `PriceStream`, updating `store.prices` with completed minute bars.
///
/// On each tick:
///   1. Update the `FormingBar` for the symbol with the new price.
///   2. If the current tick's minute has advanced past the forming bar's minute,
///      commit the completed bar to `PriceBuffer`.
///
/// Reconnects with exponential backoff on stream failure.
pub async fn run_price_stream(store: Arc<DataStore>) {
    let mut universe_rx = store.subscribe_universe();
    let mut backoff_secs = 1u64;

    loop {
        // Re-read the universe on every (re)connect so live add/remove takes
        // effect. Snapshot is owned so the read lock isn't held across awaits.
        let symbol_strings = store.symbols_snapshot();
        let symbols: Vec<&str> = symbol_strings.iter().map(|s| s.as_str()).collect();

        if symbols.is_empty() {
            // Empty universe — wait for the watcher to populate it before connecting.
            let _ = universe_rx.changed().await;
            continue;
        }

        match PriceStream::subscribe(&symbols).await {
            Ok(mut stream) => {
                backoff_secs = 1; // reset on successful connect
                tracing::info!(count = symbols.len(), "PriceStream connected");

                loop {
                    tokio::select! {
                        // Universe changed → drop this subscription and reconnect
                        // with the new symbol set.
                        _ = universe_rx.changed() => {
                            tracing::info!("universe changed — re-subscribing PriceStream");
                            break;
                        }
                        maybe_update = stream.next() => {
                            let Some(update) = maybe_update else {
                                tracing::warn!("PriceStream ended, reconnecting in {backoff_secs}s");
                                break;
                            };
                            let ts = update.time;
                            let price = update.price as f64;
                            let day_volume = Some(update.day_volume);
                            let sym = &update.id;

                            // Update forming bar
                            let prev_forming = store
                                .forming
                                .get(sym)
                                .map(|f| f.clone())
                                .unwrap_or_default();
                            let new_forming = prev_forming.tick(price, day_volume, ts);

                            // Minute boundary: commit the completed bar.
                            if prev_forming.open.is_some()
                                && new_forming.minute != prev_forming.minute
                            {
                                if let Some(bar) = prev_forming.to_bar() {
                                    let mut price_buf = store
                                        .prices
                                        .entry(sym.clone())
                                        .or_insert_with(|| crate::store::PriceBuffer::new(98_280));
                                    price_buf.push(bar);
                                }
                            }

                            store.forming.insert(sym.clone(), new_forming);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!(err = %e, "PriceStream connect failed, retry in {backoff_secs}s");
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(60);
    }
}
