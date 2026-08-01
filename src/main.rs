//! meridian-bus — live market-hours poller and break-glass monitor (§3.5, §4.5).
//!
//! ```text
//! meridian-bus live               poll loop: quotes 60s in-session, 5m idle
//! meridian-bus once               bootstrap + single cycle, print snapshot
//! meridian-bus status             journal + alert summary
//! ```

use clap::{Parser, Subcommand};
use meridian::config::BusConfig;
use meridian::monitor;

#[derive(Parser)]
#[command(name = "meridian-bus", about = "Meridian live poller / break-glass monitor")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// Comma-separated monitored symbols; the first is the trigger symbol.
    #[arg(long, global = true)]
    symbols: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the persistent market-hours poll loop.
    Live,
    /// Bootstrap, run one cycle, print the snapshot, exit.
    Once,
    /// Summarize the alert record and today's monitor journal.
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let mut cfg = BusConfig::default();
    if let Some(s) = &cli.symbols {
        let symbols: Vec<String> = s
            .split(',')
            .map(|x| x.trim().to_uppercase())
            .filter(|x| !x.is_empty())
            .collect();
        anyhow::ensure!(!symbols.is_empty(), "--symbols parsed to an empty list");
        cfg.symbols = symbols;
    }

    match cli.cmd {
        Cmd::Live => monitor::run_live(cfg).await,
        Cmd::Once => monitor::run_once(cfg).await,
        Cmd::Status => monitor::run_status(cfg),
    }
}
