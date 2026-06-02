use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "meridian", about = "Meridian DataBus — Section 6.1")]
pub struct Args {
    #[command(subcommand)]
    pub mode: Mode,
}

#[derive(Subcommand)]
pub enum Mode {
    /// Run the live DataBus scheduler, emitting JSONL every minute
    Live(LiveArgs),
    /// Batch-collect historical data for a single symbol
    Historical(HistoricalArgs),
    /// Fit the global Block-8 text-embedding PCA basis over the whole universe.
    ///
    /// Run once before per-symbol historical collection so every asset shares
    /// comparable Block-8 axes (harrier-oss is not Matryoshka, so its 1024-d
    /// output is PCA-reduced to 64 dims, never truncated). `make collect` runs
    /// this automatically after the FNSPID news caches are built.
    FitTextPca(FitTextPcaArgs),
}

#[derive(clap::Args)]
pub struct FitTextPcaArgs {
    /// Path to newline-separated list of symbols (max 100) — the universe whose
    /// FNSPID news corpus the PCA basis is fit over.
    #[arg(long, default_value = "symbols.txt")]
    pub symbols_file: PathBuf,
}

#[derive(clap::Args)]
pub struct LiveArgs {
    /// Path to newline-separated list of symbols to track (max 100)
    #[arg(long, default_value = "symbols.txt")]
    pub symbols_file: PathBuf,

    /// Path to persist/restore rolling normalization statistics
    #[arg(long, default_value = "data/rolling_stats.bin")]
    pub stats_path: PathBuf,
}

#[derive(clap::Args)]
pub struct HistoricalArgs {
    /// Ticker symbol (e.g. AAPL)
    #[arg(long)]
    pub symbol: String,

    /// Data resolution.
    ///
    /// Yahoo Finance availability:
    ///   1m  — last 29 days only (7-day chunks)
    ///   5m  — last 58 days only (7-day chunks)
    ///   1h  — last ~2 years (single request)
    ///   1d  — full history since IPO (single request, range=max)
    #[arg(long, default_value = "1d")]
    pub interval: HistoricalInterval,

    /// Start date (YYYY-MM-DD). Omit to use maximum available for the interval.
    #[arg(long)]
    pub start: Option<NaiveDate>,

    /// End date (YYYY-MM-DD). Omit to default to today.
    #[arg(long)]
    pub end: Option<NaiveDate>,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum HistoricalInterval {
    #[value(name = "1m")]
    OneMinute,
    #[value(name = "5m")]
    FiveMinutes,
    #[value(name = "1h")]
    OneHour,
    #[value(name = "1d")]
    OneDay,
}
