use crate::args::{HistoricalArgs, HistoricalInterval};
use crate::features::{classify_asset, FeatureRow};
use crate::output::emit_row;
use crate::store::{Bar, PriceBuffer, RollingStatsStore};
use finance_query::{Interval, TimeRange, Ticker};

/// Historical batch collection mode.
///
/// Downloads chart data for a single symbol and emits JSONL rows.
/// Fundamental, options, and risk blocks are filled with zeros in this mode.
/// Event state block is populated with dividend and split signals from chart events.
///
/// Yahoo Finance availability per interval:
///   1m  — last 29 days  (chunking handled internally by chart_range)
///   5m  — last 58 days  (chunking handled internally by chart_range)
///   1h  — last ~2 years (chart_range with 730-day default window)
///   1d  — 10-year history (chart(TenYears) when no start given; Max downsamples to monthly)
pub async fn run(args: HistoricalArgs) -> anyhow::Result<()> {
    let symbol = &args.symbol;
    let ticker = Ticker::new(symbol).await?;

    let today = chrono::Utc::now().date_naive();
    let end_date = args.end.unwrap_or(today);
    let end_ts = end_date.and_hms_opt(23, 59, 59).unwrap().and_utc().timestamp();

    let all_bars: Vec<Bar> = match args.interval {
        HistoricalInterval::OneMinute => {
            let start_ts = args.start.unwrap_or(today - chrono::Duration::days(29))
                .and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            ticker.chart_range(Interval::OneMinute, start_ts, end_ts).await?
                .candles.iter().map(bar_from_candle).collect()
        }
        HistoricalInterval::FiveMinutes => {
            let start_ts = args.start.unwrap_or(today - chrono::Duration::days(58))
                .and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            ticker.chart_range(Interval::FiveMinutes, start_ts, end_ts).await?
                .candles.iter().map(bar_from_candle).collect()
        }
        HistoricalInterval::OneHour => {
            let start_ts = args.start.unwrap_or(today - chrono::Duration::days(730))
                .and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            ticker.chart_range(Interval::OneHour, start_ts, end_ts).await?
                .candles.iter().map(bar_from_candle).collect()
        }
        HistoricalInterval::OneDay => match args.start {
            Some(start) => {
                let start_ts = start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                ticker.chart_range(Interval::OneDay, start_ts, end_ts).await?
                    .candles.iter().map(bar_from_candle).collect()
            }
            None => {
                // TenYears gives reliable daily bars; Max auto-downsamples to monthly.
                ticker.chart(Interval::OneDay, TimeRange::TenYears).await?
                    .candles.iter()
                    .filter(|c| c.timestamp <= end_ts)
                    .map(bar_from_candle)
                    .collect()
            }
        },
    };

    let total = all_bars.len();
    if total == 0 {
        anyhow::bail!("no candle data returned for {symbol}");
    }

    // Fetch chart events (dividends, splits) — already cached from the chart request
    let dividends = ticker.dividends(TimeRange::TenYears).await.unwrap_or_default();
    let splits = ticker.splits(TimeRange::TenYears).await.unwrap_or_default();

    // Pre-sort event timestamps for binary search
    let div_timestamps: Vec<i64> = dividends.iter().map(|d| d.timestamp).collect();
    let div_amounts: Vec<f64> = dividends.iter().map(|d| d.amount).collect();
    let split_timestamps: Vec<i64> = splits.iter().map(|s| s.timestamp).collect();
    let split_ratios: Vec<f64> = splits.iter().map(|s| s.numerator / s.denominator.max(1.0)).collect();

    tracing::info!(
        symbol = %symbol, bars = total, dividends = dividends.len(),
        splits = splits.len(), interval = ?args.interval, "emitting JSONL rows"
    );

    let rolling_stats = RollingStatsStore::new();
    let seconds_per_day: f64 = 86_400.0;
    let div_decay_tau: f64 = 14.0;    // 14-day decay for dividend proximity
    let split_decay_tau: f64 = 7.0;   // 7-day decay for split events (sharper)

    for i in 0..total {
        let buf = PriceBuffer::prefix(&all_bars, i);
        let bar = &all_bars[i];
        let bar_ts = bar.timestamp;

        let timestamp = chrono::DateTime::from_timestamp(bar_ts, 0).unwrap_or_default();

        // Compute dividend proximity: exp(-days_since / tau)
        // Find most recent dividend at or before this bar
        let exdiv_proximity = match div_timestamps.partition_point(|&t| t <= bar_ts) {
            0 => 0.0f32, // no dividend before this bar
            idx => {
                let days_since = (bar_ts - div_timestamps[idx - 1]) as f64 / seconds_per_day;
                let amount = div_amounts[idx - 1];
                // Scale by normalized dividend amount (typical ~0.2-2.0, log-compress)
                let mag = (1.0 + amount).ln().min(2.0);
                (mag * (-days_since / div_decay_tau).exp()) as f32
            }
        };

        // Compute split signal: log2(ratio) * exp(-days_since / tau)
        // Ratio > 1 = forward split (e.g., 4:1), < 1 = reverse split
        let split_signal = match split_timestamps.partition_point(|&t| t <= bar_ts) {
            0 => 0.0f32,
            idx => {
                let days_since = (bar_ts - split_timestamps[idx - 1]) as f64 / seconds_per_day;
                let ratio = split_ratios[idx - 1];
                let mag = ratio.log2().abs().min(4.0); // cap at 16:1 splits
                (mag * (-days_since / split_decay_tau).exp()) as f32
            }
        };

        // Build event_state: fill slots 7 (split) and 9 (dividend)
        let mut event_state = [0.0f32; 10];
        event_state[7] = split_signal;
        event_state[9] = exdiv_proximity;

        let row = FeatureRow {
            timestamp,
            symbol: symbol.clone(),
            asset_class: classify_asset(symbol),
            price_volume: crate::features::price_volume::assemble(Some(&buf), &rolling_stats, symbol),
            technical: crate::features::technical::assemble(Some(&buf), &rolling_stats, symbol),
            event_state,
            ..FeatureRow::default()
        };

        emit_row(&row);

        if i > 0 && i % 10_000 == 0 {
            tracing::info!(symbol = %symbol, progress = i, total, "historical progress");
        }
    }

    tracing::info!(symbol = %symbol, rows = total, "historical collection complete");
    Ok(())
}

fn bar_from_candle(c: &finance_query::Candle) -> Bar {
    Bar {
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
        timestamp: c.timestamp,
    }
}
