use crate::store::{DataStore, FormingBar};
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
    let symbols: Vec<&str> = store.symbols.iter().map(|s| s.as_str()).collect();
    let symbol_strings: Vec<String> = store.symbols.clone();

    let mut backoff_secs = 1u64;

    loop {
        match PriceStream::subscribe(&symbols).await {
            Ok(mut stream) => {
                backoff_secs = 1; // reset on successful connect
                tracing::info!(count = symbols.len(), "PriceStream connected");

                while let Some(update) = stream.next().await {
                    let ts = update.time;
                    let price = update.price as f64;
                    let day_volume = Some(update.day_volume);
                    let sym = &update.id;

                    // Update forming bar
                    let prev_forming = store.forming.get(sym).map(|f| f.clone()).unwrap_or_default();
                    let new_forming = prev_forming.tick(price, day_volume, ts);

                    // Detect minute boundary: if the minute changed, commit the previous bar
                    if prev_forming.open.is_some() && new_forming.minute != prev_forming.minute {
                        if let Some(bar) = prev_forming.to_bar() {
                            let mut price_buf = store.prices.entry(sym.clone()).or_insert_with(|| {
                                crate::store::PriceBuffer::new(98_280)
                            });
                            price_buf.push(bar);
                        }
                    }

                    store.forming.insert(sym.clone(), new_forming);
                }

                tracing::warn!("PriceStream ended, reconnecting in {backoff_secs}s");
            }
            Err(e) => {
                tracing::error!(err = %e, "PriceStream connect failed, retry in {backoff_secs}s");
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(60);
    }
}
