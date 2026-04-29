use crate::store::DataStore;
use finance_query::{Screener, finance};
use std::collections::HashMap;
use std::sync::Arc;

/// Poll all predefined screeners once and update `store.screener_membership`.
///
/// For each symbol in the portfolio, records the fraction of screeners it appears
/// in (0.0–1.0). Run hourly — screener membership changes slowly.
pub async fn poll_screener_membership(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let all_screeners = Screener::all();
    let n = all_screeners.len() as f32;
    if n == 0.0 {
        return Ok(());
    }

    let mut counts: HashMap<String, f32> = HashMap::new();

    for &screener in all_screeners {
        match finance::screener(screener, 50).await {
            Ok(result) => {
                for quote in &result.quotes {
                    *counts.entry(quote.symbol.clone()).or_insert(0.0) += 1.0;
                }
            }
            Err(e) => {
                tracing::warn!(screener = ?screener, err = %e, "screener poll failed");
            }
        }
    }

    for (sym, count) in counts {
        store.screener_membership.insert(sym, (count / n).clamp(0.0, 1.0));
    }

    tracing::debug!("Screener membership updated");
    Ok(())
}
