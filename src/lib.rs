//! Meridian DataBus — the live market-hours poller (spec §3.5).
//!
//! Runs on the appliance during market hours. Sweeps quotes and intraday bars
//! for the ETF core, polls the SEC 8-K feed for event context, and evaluates
//! the break-glass severity rules (§4.5) every cycle. Outputs are append-only
//! JSONL journals plus a `latest.json` snapshot for Python consumers.
//!
//! Break-glass alerts are advisory records — execution (§10 step 10) does not
//! exist yet, so an alert is a notification and a git-tracked row, not a trade.

pub mod bars;
pub mod config;
pub mod feed;
pub mod history;
pub mod journal;
pub mod monitor;
pub mod quotes;
pub mod severity;
