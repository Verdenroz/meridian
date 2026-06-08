//! Live universe control — hot-reload `portfolio_symbols.txt`.
//!
//! Polls the symbols file for modifications (mtime). On change it diffs the
//! parsed list against the live universe, removes dropped symbols (freeing their
//! per-asset state), adds new ones (warmed up in the background before they
//! emit), and bumps the universe generation so the price stream re-subscribes.
//! Edits take effect within `POLL_SECS` — no restart.

use crate::store::{DataStore, parse_symbols};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

const POLL_SECS: u64 = 3;

pub async fn watch_symbols_file(store: Arc<DataStore>, path: PathBuf) {
    let mut last_mtime = file_mtime(&path);
    tracing::info!(path = %path.display(), "watching symbols file for live universe edits");

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(POLL_SECS)).await;

        let mtime = file_mtime(&path);
        if mtime == last_mtime {
            continue;
        }
        last_mtime = mtime;

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), err = %e, "portfolio_symbols.txt reread failed");
                continue;
            }
        };

        let desired = parse_symbols(&content);
        if desired.is_empty() {
            tracing::warn!("portfolio_symbols.txt parsed empty — ignoring (keeping current universe)");
            continue;
        }

        let current = store.symbols_snapshot();
        let desired_set: HashSet<&str> = desired.iter().map(String::as_str).collect();
        let current_set: HashSet<&str> = current.iter().map(String::as_str).collect();

        let to_remove: Vec<String> = current
            .iter()
            .filter(|s| !desired_set.contains(s.as_str()))
            .cloned()
            .collect();
        let to_add: Vec<String> = desired
            .iter()
            .filter(|s| !current_set.contains(s.as_str()))
            .cloned()
            .collect();

        if to_remove.is_empty() && to_add.is_empty() {
            continue;
        }

        let mut changed = false;
        for sym in &to_remove {
            if store.remove_symbol(sym) {
                tracing::info!(symbol = %sym, "universe: removed");
                changed = true;
            }
        }
        for sym in to_add {
            if store.add_symbol(sym.clone()) {
                tracing::info!(symbol = %sym, "universe: added (warming up)");
                changed = true;
                let store2 = store.clone();
                tokio::spawn(async move {
                    super::warmup::warm_up_symbol(&store2, &sym).await;
                });
            }
        }

        if changed {
            // Re-subscribe the price stream now (removed symbols stop streaming);
            // warm-up bumps again when each added symbol becomes ready.
            store.bump_universe();
        }
    }
}

fn file_mtime(path: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}
