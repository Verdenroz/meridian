use crate::args::{HistoricalArgs, HistoricalInterval};
use crate::features::{classify_asset, FeatureRow};
use crate::output::emit_row;
use crate::store::{Bar, PriceBuffer, RollingStatsStore};
use finance_query::{Interval, TimeRange, Ticker};

/// Historical batch collection mode.
///
/// Downloads chart data for a single symbol and emits JSONL rows.
/// Fundamental, options, and risk blocks are filled with zeros in this mode.
///
/// Yahoo Finance availability per interval:
///   1m  — last 29 days  (chunking handled internally by chart_range)
///   5m  — last 58 days  (chunking handled internally by chart_range)
///   1h  — last ~2 years (chart_range with 730-day default window)
///   1d  — full history  (chart(TimeRange::Max) when no start given)
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
                // No start given — fetch full history back to IPO
                ticker.chart(Interval::OneDay, TimeRange::Max).await?
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

    tracing::info!(symbol = %symbol, bars = total, interval = ?args.interval, "emitting JSONL rows");

    let rolling_stats = RollingStatsStore::new();

    for i in 0..total {
        let buf = PriceBuffer::prefix(&all_bars, i);
        let bar = &all_bars[i];

        let timestamp = chrono::DateTime::from_timestamp(bar.timestamp, 0).unwrap_or_default();

        let row = FeatureRow {
            timestamp,
            symbol: symbol.clone(),
            asset_class: classify_asset(symbol),
            price_volume: crate::features::price_volume::assemble(Some(&buf), &rolling_stats, symbol),
            technical: crate::features::technical::assemble(Some(&buf), &rolling_stats, symbol),
            // Remaining blocks: zeros in historical mode
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
