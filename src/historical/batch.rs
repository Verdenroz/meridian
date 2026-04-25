use crate::args::{HistoricalArgs, HistoricalInterval};
use crate::features::{FeatureRow, classify_asset};
use crate::output::emit_row;
use crate::store::{Bar, PriceBuffer, RollingStatsStore};
use finance_query::{Interval, Ticker, TimeRange};
use std::collections::{BTreeMap, HashMap};

/// Historical batch collection mode.
///
/// Downloads chart data for a single symbol and emits JSONL rows.
/// Fundamental, options, and risk blocks are still filled with zeros in this mode.
/// Event state block is populated with dividend and split signals from chart events.
/// Market-wide block is populated from finance-query benchmark series where available.
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
    let end_ts = end_date
        .and_hms_opt(23, 59, 59)
        .unwrap()
        .and_utc()
        .timestamp();

    let all_bars = fetch_historical_bars(&ticker, &args, end_ts).await?;
    let market_lookup = build_market_feature_lookup(&args, end_ts).await;

    let total = all_bars.len();
    if total == 0 {
        anyhow::bail!("no candle data returned for {symbol}");
    }

    // Fetch chart events (dividends, splits) — already cached from the chart request
    let dividends = ticker
        .dividends(TimeRange::TenYears)
        .await
        .unwrap_or_default();
    let splits = ticker.splits(TimeRange::TenYears).await.unwrap_or_default();

    // Pre-sort event timestamps for binary search
    let div_timestamps: Vec<i64> = dividends.iter().map(|d| d.timestamp).collect();
    let div_amounts: Vec<f64> = dividends.iter().map(|d| d.amount).collect();
    let split_timestamps: Vec<i64> = splits.iter().map(|s| s.timestamp).collect();
    let split_ratios: Vec<f64> = splits
        .iter()
        .map(|s| s.numerator / s.denominator.max(1.0))
        .collect();

    tracing::info!(
        symbol = %symbol, bars = total, dividends = dividends.len(),
        splits = splits.len(), interval = ?args.interval, "emitting JSONL rows"
    );

    let rolling_stats = RollingStatsStore::new();
    let seconds_per_day: f64 = 86_400.0;
    let div_decay_tau: f64 = 14.0; // 14-day decay for dividend proximity
    let split_decay_tau: f64 = 7.0; // 7-day decay for split events (sharper)

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
            price_volume: crate::features::price_volume::assemble(
                Some(&buf),
                &rolling_stats,
                symbol,
            ),
            technical: crate::features::technical::assemble(Some(&buf), &rolling_stats, symbol),
            event_state,
            market: market_lookup.vector_at(bar_ts),
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

#[derive(Default)]
struct MarketFeatureLookup {
    returns: HashMap<&'static str, BTreeMap<i64, f32>>,
    closes: HashMap<&'static str, BTreeMap<i64, f64>>,
}

impl MarketFeatureLookup {
    fn vector_at(&self, ts: i64) -> [f32; 96] {
        let mut out = [0.0f32; 96];

        let spx = self.ret("spx", ts);
        let nasdaq = self.ret("nasdaq", ts);
        let russell = self.ret("russell", ts);
        let dow = self.ret("dow", ts);
        let vix_level = self.close("vix", ts) as f32;
        let vix_ret = self.ret("vix", ts);
        let dxy = self.ret("dxy", ts);
        let gold = self.ret("gold", ts);
        let oil = self.ret("oil", ts);
        let btc_close = self.close("btc", ts);
        let eth_close = self.close("eth", ts);
        let btc_ret = self.ret("btc", ts);
        let eth_ret = self.ret("eth", ts);

        // Volatility & sentiment block (partial historical proxies)
        out[0] = (vix_level / 100.0).clamp(0.0, 2.0);
        // out[6] carries a rescaled VIX with a wider dynamic range than out[0]:
        // VIX=20 ≈ 1.0 (neutral), 40 ≈ 2.0 (elevated), 80 ≈ 4.0 (crisis).
        // Raw level would blow the FDA pipeline's ±10 per-feature bound.
        out[6] = (vix_level / 20.0).clamp(0.0, 5.0);
        out[7] = vix_ret;
        out[10] = dxy;
        out[11] = ((gold - oil) * 0.5).clamp(-2.0, 2.0);

        // Sector/industry proxy block
        out[36] = spx;
        out[37] = nasdaq;
        out[38] = russell;
        out[39] = dow;
        let dispersion = stddev([spx, nasdaq, russell, dow]);
        out[47] = dispersion;
        let (best_idx, worst_idx) = argmax_min([spx, nasdaq, russell, dow]);
        out[48] = best_idx as f32 / 10.0;
        out[49] = worst_idx as f32 / 10.0;
        out[50] = (spx - nasdaq).abs();
        out[51] = [spx, nasdaq, russell, dow]
            .iter()
            .filter(|r| **r > 0.0)
            .count() as f32
            / 4.0;

        // World indices & cross-asset block
        out[52] = spx;
        out[53] = nasdaq;
        out[54] = russell;
        out[55] = dow;
        out[61] = dxy;
        out[64] = gold;
        out[65] = oil;
        out[66] = (btc_close as f32 / 100_000.0).clamp(0.0, 2.0);
        out[67] = btc_ret;
        out[68] = (eth_close as f32 / 10_000.0).clamp(0.0, 2.0);
        out[69] = eth_ret;
        let total_proxy = (btc_close.max(0.0) + eth_close.max(0.0)).max(1e-6);
        out[70] = ((total_proxy.ln() as f32) / 12.0).clamp(0.0, 2.0);
        out[71] = (btc_close / total_proxy) as f32;

        // Breadth/flow block (proxy)
        let breadth = [spx, nasdaq, russell, dow]
            .iter()
            .filter(|r| **r > 0.0)
            .count() as f32;
        out[72] = (breadth / 4.0) * 2.0 - 1.0; // [-1, 1]
        let max_r = [spx, nasdaq, russell, dow]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_r = [spx, nasdaq, russell, dow]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        out[73] = (max_r - min_r).clamp(0.0, 4.0);
        out[74] = spx.abs();
        out[75] = nasdaq.abs();
        out[76] = russell.abs();
        out[77] = ((spx * nasdaq) + (spx * russell) + (nasdaq * russell)) / 3.0;
        out[78] = breadth / 4.0;
        out[79] = (spx + nasdaq + russell + dow) / 4.0;
        out[80] = max_r.max(0.0);
        out[81] = if oil.abs() > 1e-6 {
            (spx.abs() / oil.abs()).min(5.0)
        } else {
            0.0
        };

        out
    }

    fn ret(&self, key: &'static str, ts: i64) -> f32 {
        self.returns
            .get(key)
            .and_then(|m| m.range(..=ts).next_back().map(|(_, v)| *v))
            .unwrap_or(0.0)
    }

    fn close(&self, key: &'static str, ts: i64) -> f64 {
        self.closes
            .get(key)
            .and_then(|m| m.range(..=ts).next_back().map(|(_, v)| *v))
            .unwrap_or(0.0)
    }
}

async fn build_market_feature_lookup(args: &HistoricalArgs, end_ts: i64) -> MarketFeatureLookup {
    const MARKET_PROXIES: [(&str, &str); 10] = [
        ("spx", "^GSPC"),
        ("nasdaq", "^IXIC"),
        ("russell", "^RUT"),
        ("dow", "^DJI"),
        ("vix", "^VIX"),
        ("dxy", "UUP"),
        ("gold", "GC=F"),
        ("oil", "CL=F"),
        ("btc", "BTC-USD"),
        ("eth", "ETH-USD"),
    ];

    let mut out = MarketFeatureLookup::default();
    for (alias, symbol) in MARKET_PROXIES {
        match Ticker::new(symbol).await {
            Ok(ticker) => match fetch_historical_bars(&ticker, args, end_ts).await {
                Ok(bars) if bars.len() >= 2 => {
                    out.returns.insert(alias, build_return_series(&bars));
                    out.closes.insert(alias, build_close_series(&bars));
                }
                Ok(_) => {
                    tracing::warn!(proxy = %symbol, "market proxy returned insufficient bars");
                }
                Err(err) => {
                    tracing::warn!(proxy = %symbol, err = %err, "failed to fetch market proxy bars");
                }
            },
            Err(err) => {
                tracing::warn!(proxy = %symbol, err = %err, "failed to initialize market proxy ticker");
            }
        }
    }
    out
}

async fn fetch_historical_bars(
    ticker: &Ticker,
    args: &HistoricalArgs,
    end_ts: i64,
) -> anyhow::Result<Vec<Bar>> {
    let today = chrono::Utc::now().date_naive();
    let bars = match args.interval {
        HistoricalInterval::OneMinute => {
            let start_ts = args
                .start
                .unwrap_or(today - chrono::Duration::days(29))
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            ticker
                .chart_range(Interval::OneMinute, start_ts, end_ts)
                .await?
                .candles
                .iter()
                .map(bar_from_candle)
                .collect()
        }
        HistoricalInterval::FiveMinutes => {
            let start_ts = args
                .start
                .unwrap_or(today - chrono::Duration::days(58))
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            ticker
                .chart_range(Interval::FiveMinutes, start_ts, end_ts)
                .await?
                .candles
                .iter()
                .map(bar_from_candle)
                .collect()
        }
        HistoricalInterval::OneHour => {
            let start_ts = args
                .start
                .unwrap_or(today - chrono::Duration::days(730))
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp();
            ticker
                .chart_range(Interval::OneHour, start_ts, end_ts)
                .await?
                .candles
                .iter()
                .map(bar_from_candle)
                .collect()
        }
        HistoricalInterval::OneDay => match args.start {
            Some(start) => {
                let start_ts = start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                ticker
                    .chart_range(Interval::OneDay, start_ts, end_ts)
                    .await?
                    .candles
                    .iter()
                    .map(bar_from_candle)
                    .collect()
            }
            None => ticker
                .chart(Interval::OneDay, TimeRange::TenYears)
                .await?
                .candles
                .iter()
                .filter(|c| c.timestamp <= end_ts)
                .map(bar_from_candle)
                .collect(),
        },
    };
    Ok(bars)
}

fn build_return_series(bars: &[Bar]) -> BTreeMap<i64, f32> {
    let mut out = BTreeMap::new();
    for w in bars.windows(2) {
        let prev = &w[0];
        let cur = &w[1];
        if prev.close > 0.0 && cur.close > 0.0 {
            let ret = (cur.close / prev.close).ln() as f32;
            out.insert(cur.timestamp, ret.clamp(-5.0, 5.0));
        }
    }
    out
}

fn build_close_series(bars: &[Bar]) -> BTreeMap<i64, f64> {
    let mut out = BTreeMap::new();
    for b in bars {
        out.insert(b.timestamp, b.close.max(0.0));
    }
    out
}

fn stddev(vals: [f32; 4]) -> f32 {
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
    var.sqrt()
}

fn argmax_min(vals: [f32; 4]) -> (usize, usize) {
    let mut best = 0usize;
    let mut worst = 0usize;
    for i in 1..vals.len() {
        if vals[i] > vals[best] {
            best = i;
        }
        if vals[i] < vals[worst] {
            worst = i;
        }
    }
    (best, worst)
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

#[cfg(test)]
mod tests {
    use super::{MarketFeatureLookup, argmax_min, build_close_series, build_return_series, stddev};
    use crate::store::Bar;
    use std::collections::BTreeMap;

    #[test]
    fn return_series_uses_log_returns() {
        let bars = vec![
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 100.0,
                volume: 1,
                timestamp: 10,
            },
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 110.0,
                volume: 1,
                timestamp: 20,
            },
        ];
        let series = build_return_series(&bars);
        let r = series.get(&20).copied().unwrap_or_default();
        assert!(r > 0.09 && r < 0.11);
    }

    #[test]
    fn close_series_aligns_timestamps() {
        let bars = vec![
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 42.0,
                volume: 1,
                timestamp: 11,
            },
            Bar {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 43.0,
                volume: 1,
                timestamp: 12,
            },
        ];
        let series = build_close_series(&bars);
        assert_eq!(series.get(&11).copied().unwrap_or_default(), 42.0);
        assert_eq!(series.get(&12).copied().unwrap_or_default(), 43.0);
    }

    #[test]
    fn stddev_and_extrema_helpers_are_stable() {
        let d = stddev([0.0, 1.0, 2.0, 3.0]);
        assert!(d > 1.11 && d < 1.13);
        assert_eq!(argmax_min([0.1, -2.0, 0.3, 0.2]), (2, 1));
    }

    #[test]
    fn market_vector_populates_proxy_slots() {
        let mut lookup = MarketFeatureLookup::default();
        for key in [
            "spx", "nasdaq", "russell", "dow", "dxy", "gold", "oil", "btc", "eth", "vix",
        ] {
            let mut series = BTreeMap::new();
            series.insert(100, 0.01);
            lookup.returns.insert(key, series);
        }

        let mut vix_close = BTreeMap::new();
        vix_close.insert(100, 20.0);
        lookup.closes.insert("vix", vix_close);

        let mut btc_close = BTreeMap::new();
        btc_close.insert(100, 90_000.0);
        lookup.closes.insert("btc", btc_close);

        let mut eth_close = BTreeMap::new();
        eth_close.insert(100, 3_000.0);
        lookup.closes.insert("eth", eth_close);

        let out = lookup.vector_at(100);
        assert!(out[52] > 0.0); // SPX proxy return
        assert!(out[61] > 0.0); // DXY proxy return
        assert!(out[66] > 0.0); // BTC level proxy
        assert!(out[68] > 0.0); // ETH level proxy
        assert!(out[72] > 0.0); // breadth proxy
    }
}
