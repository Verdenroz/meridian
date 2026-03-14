use finance_query::indicators::{bollinger_bands, macd, rsi, sma, stochastic};
use finance_query::{Interval, Ticker, TimeRange};
use serde_json::json;
use std::env;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

fn parse_arg(args: &[String], key: &str) -> Result<String, Box<dyn Error>> {
    for window in args.windows(2) {
        if window[0] == key {
            return Ok(window[1].clone());
        }
    }
    Err(format!("missing required argument: {key}").into())
}

fn parse_optional_arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == key)
        .map(|window| window[1].clone())
}

fn safe_return(current: f64, previous: f64) -> f64 {
    if previous == 0.0 || current <= 0.0 || previous <= 0.0 {
        0.0
    } else {
        (current / previous).ln()
    }
}

fn f64_from_option_series(series: &[Option<f64>], index: usize) -> f64 {
    series
        .get(index)
        .and_then(|value| value.as_ref().copied())
        .unwrap_or(0.0)
}

fn mean(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return 0.0;
    }
    slice.iter().sum::<f64>() / slice.len() as f64
}

fn std_dev(slice: &[f64]) -> f64 {
    if slice.len() < 2 {
        return 0.0;
    }
    let avg = mean(slice);
    let variance = slice
        .iter()
        .map(|value| {
            let delta = *value - avg;
            delta * delta
        })
        .sum::<f64>()
        / slice.len() as f64;
    variance.sqrt()
}

fn z_score(values: &[f64], index: usize, window: usize) -> f64 {
    if index + 1 < window {
        return 0.0;
    }
    let slice = &values[(index + 1 - window)..=index];
    let sigma = std_dev(slice);
    if sigma == 0.0 {
        0.0
    } else {
        (values[index] - mean(slice)) / sigma
    }
}

fn rolling_avg(values: &[f64], index: usize, window: usize) -> f64 {
    if index + 1 < window {
        return mean(&values[..=index]);
    }
    mean(&values[(index + 1 - window)..=index])
}

fn hash_text_to_embedding(texts: &[String], dim: usize) -> Vec<f64> {
    let mut embedding = vec![0.0; dim];
    for text in texts {
        for token in text.split_whitespace() {
            let hash = token
                .as_bytes()
                .iter()
                .fold(0usize, |acc, byte| acc.wrapping_mul(31).wrapping_add(*byte as usize));
            embedding[hash % dim] += 1.0;
        }
    }
    let norm = embedding.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

fn sector_embedding(sector: Option<&str>) -> f64 {
    if let Some(value) = sector {
        let hash = value
            .as_bytes()
            .iter()
            .fold(0u32, |acc, byte| acc.wrapping_mul(33).wrapping_add(*byte as u32));
        (hash % 1000) as f64 / 1000.0
    } else {
        0.0
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let symbol = parse_arg(&args, "--symbol")?;
    let _start = parse_arg(&args, "--start")?;
    let _end = parse_arg(&args, "--end")?;
    let asset_class = parse_optional_arg(&args, "--asset-class").unwrap_or_else(|| "equity".to_string());
    let max_rows = parse_optional_arg(&args, "--max-rows")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512);

    let ticker = Ticker::new(&symbol).await?;
    let chart = ticker.chart(Interval::OneDay, TimeRange::OneMonth).await?;
    if chart.candles.is_empty() {
        return Err("no candles returned by finance-query".into());
    }

    let closes: Vec<f64> = chart.candles.iter().map(|candle| candle.close).collect();
    let highs: Vec<f64> = chart.candles.iter().map(|candle| candle.high).collect();
    let lows: Vec<f64> = chart.candles.iter().map(|candle| candle.low).collect();
    let volumes: Vec<f64> = chart.candles.iter().map(|candle| candle.volume as f64).collect();
    let ranges: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .map(|(high, low)| (high - low).abs())
        .collect();

    let rsi_14 = rsi(&closes, 14).unwrap_or_else(|_| vec![None; closes.len()]);
    let macd_values = macd(&closes, 12, 26, 9).ok();
    let stochastic_values = stochastic(&highs, &lows, &closes, 14, 3, 3).ok();
    let bb = bollinger_bands(&closes, 20, 2.0).ok();
    let _sma_20 = sma(&closes, 20);
    let _sma_50 = sma(&closes, 50);
    let _sma_200 = sma(&closes, 200);

    let quote = ticker.quote().await.ok();
    let financial_data = ticker.financial_data().await.ok().flatten();
    let key_stats = ticker.key_stats().await.ok().flatten();
    let asset_profile = ticker.asset_profile().await.ok().flatten();
    let news = ticker.news().await.unwrap_or_default();

    let options = ticker.options(None).await.ok();
    let options_features = if let Some(chain) = options.as_ref() {
        let calls = chain.calls();
        let puts = chain.puts();

        let call_iv_avg = if calls.is_empty() {
            0.0
        } else {
            calls
                .iter()
                .map(|contract| contract.implied_volatility.unwrap_or(0.0))
                .sum::<f64>()
                / calls.len() as f64
        };
        let put_iv_avg = if puts.is_empty() {
            0.0
        } else {
            puts.iter()
                .map(|contract| contract.implied_volatility.unwrap_or(0.0))
                .sum::<f64>()
                / puts.len() as f64
        };
        let call_volume = calls
            .iter()
            .map(|contract| contract.volume.unwrap_or(0) as f64)
            .sum::<f64>();
        let put_volume = puts
            .iter()
            .map(|contract| contract.volume.unwrap_or(0) as f64)
            .sum::<f64>();
        let put_call_ratio = if call_volume == 0.0 {
            0.0
        } else {
            put_volume / call_volume
        };
        let expirations = chain.expiration_dates();
        let term_structure = if expirations.len() >= 2 {
            let first = expirations[0] as f64;
            let last = expirations[expirations.len() - 1] as f64;
            if first.abs() < f64::EPSILON {
                0.0
            } else {
                (last - first) / first.abs()
            }
        } else {
            0.0
        };

        [
            0.5 * (call_iv_avg + put_iv_avg),
            put_call_ratio,
            put_iv_avg - call_iv_avg,
            term_structure,
            0.0,
        ]
    } else {
        [0.0; 5]
    };

    let sector = asset_profile
        .as_ref()
        .and_then(|profile| profile.sector.as_deref());
    let news_titles: Vec<String> = news.iter().map(|article| article.title.clone()).collect();
    let text_embedding = hash_text_to_embedding(&news_titles, 64);

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let next_expiry_days = options
        .as_ref()
        .and_then(|chain| {
            chain
                .expiration_dates()
                .into_iter()
                .filter(|timestamp| *timestamp > now_unix)
                .min()
        })
        .map(|expiry| ((expiry - now_unix) as f64 / 86_400.0).max(0.0))
        .unwrap_or(0.0);
    let earnings_week = if news
        .iter()
        .any(|article| article.title.to_lowercase().contains("earnings"))
    {
        1.0
    } else {
        0.0
    };

    let market_price = quote
        .as_ref()
        .and_then(|value| value.regular_market_price.as_ref().and_then(|number| number.raw))
        .unwrap_or(0.0);
    let market_cap = quote
        .as_ref()
        .and_then(|value| value.market_cap.as_ref().and_then(|number| number.raw))
        .unwrap_or(0) as f64;
    let trailing_eps = key_stats
        .as_ref()
        .and_then(|stats| stats.trailing_eps.as_ref().and_then(|number| number.raw))
        .unwrap_or(0.0);
    let profit_margin = financial_data
        .as_ref()
        .and_then(|data| data.profit_margins.as_ref().and_then(|number| number.raw))
        .unwrap_or(0.0);
    let total_revenue = financial_data
        .as_ref()
        .and_then(|data| data.total_revenue.as_ref().and_then(|number| number.raw))
        .unwrap_or(0) as f64;

    for (index, candle) in chart.candles.iter().enumerate().take(max_rows) {
        let prev_close = if index > 0 {
            closes[index - 1]
        } else {
            candle.close
        };
        let vol_sma_20 = rolling_avg(&volumes, index, 20).max(1e-9);

        let ret_1m = if index >= 1 {
            safe_return(candle.close, closes[index - 1])
        } else {
            0.0
        };
        let ret_5m = if index >= 5 {
            safe_return(candle.close, closes[index - 5])
        } else {
            0.0
        };
        let ret_15m = if index >= 15 {
            safe_return(candle.close, closes[index - 15])
        } else {
            0.0
        };
        let ret_60m = if index >= 60 {
            safe_return(candle.close, closes[index - 60])
        } else {
            0.0
        };
        let ret_1d = if index >= 390 {
            safe_return(candle.close, closes[index - 390])
        } else {
            0.0
        };

        let macd_hist = macd_values
            .as_ref()
            .map(|result| f64_from_option_series(&result.histogram, index))
            .unwrap_or(0.0);
        let stoch_k = stochastic_values
            .as_ref()
            .map(|result| f64_from_option_series(&result.k, index))
            .unwrap_or(0.0);
        let bb_percent_b = bb
            .as_ref()
            .map(|bands| {
                let upper = f64_from_option_series(&bands.upper, index);
                let lower = f64_from_option_series(&bands.lower, index);
                if (upper - lower).abs() < f64::EPSILON {
                    0.0
                } else {
                    (candle.close - lower) / (upper - lower)
                }
            })
            .unwrap_or(0.0);

        let atr_proxy = rolling_avg(&ranges, index, 14);
        let rv_5 = if index >= 4 {
            std_dev(&closes[(index - 4)..=index])
        } else {
            0.0
        };
        let rv_20 = if index >= 19 {
            std_dev(&closes[(index - 19)..=index])
        } else {
            0.0
        };
        let rv_ratio = if rv_20 == 0.0 { 0.0 } else { rv_5 / rv_20 };
        let obv_proxy = if index > 0 {
            let sign = if candle.close > prev_close {
                1.0
            } else if candle.close < prev_close {
                -1.0
            } else {
                0.0
            };
            sign * candle.volume as f64
        } else {
            0.0
        };
        let vwap_proxy = {
            let pv_sum = (0..=index).map(|row| closes[row] * volumes[row]).sum::<f64>();
            let volume_sum = volumes[..=index].iter().sum::<f64>().max(1e-9);
            pv_sum / volume_sum
        };

        let technical = vec![
            z_score(&closes, index, 20),
            z_score(&closes, index, 50),
            z_score(&closes, index, 200),
            if candle.close == 0.0 {
                0.0
            } else {
                ranges[index] / candle.close
            },
            f64_from_option_series(&rsi_14, index),
            if atr_proxy == 0.0 {
                0.0
            } else {
                macd_hist / atr_proxy
            },
            stoch_k,
            ret_5m,
            if candle.close == 0.0 {
                0.0
            } else {
                atr_proxy / candle.close
            },
            bb_percent_b,
            if atr_proxy == 0.0 {
                0.0
            } else {
                (candle.close - rolling_avg(&closes, index, 20)) / atr_proxy
            },
            rv_ratio,
            obv_proxy,
            if (candle.high - candle.low).abs() < f64::EPSILON {
                0.0
            } else {
                ((candle.close - candle.low) - (candle.high - candle.close))
                    / (candle.high - candle.low)
            },
            if vwap_proxy == 0.0 {
                0.0
            } else {
                (candle.close / vwap_proxy) - 1.0
            },
            if candle.close >= rolling_avg(&closes, index, 20) {
                1.0
            } else {
                -1.0
            },
            if candle.close >= rolling_avg(&closes, index, 26) {
                1.0
            } else {
                -1.0
            },
            if candle.close >= prev_close { 1.0 } else { -1.0 },
            if candle.close >= rolling_avg(&closes, index, 20) {
                1.0
            } else {
                0.0
            },
            if ranges[index] == 0.0 {
                0.0
            } else {
                (ret_1m.abs() / ranges[index]).min(1.0)
            },
        ];

        let fundamental = vec![
            market_price,
            if market_cap <= 0.0 { 0.0 } else { market_cap.ln() },
            trailing_eps,
            profit_margin,
            if total_revenue <= 0.0 {
                0.0
            } else {
                total_revenue.ln()
            },
            rolling_avg(&closes, index, 20),
            rv_20,
            news.len() as f64,
            sector_embedding(sector),
            if asset_profile.is_some() { 1.0 } else { 0.0 },
        ];

        let row = json!({
            "timestamp": candle.timestamp,
            "asset_class": asset_class,
            "price_volume": [
                safe_return(candle.open, prev_close),
                safe_return(candle.high, candle.open.max(1e-9)),
                safe_return(candle.low, candle.open.max(1e-9)),
                safe_return(candle.close, prev_close),
                safe_return(candle.volume as f64, vol_sma_20),
                ret_1m,
                ret_5m,
                ret_15m,
                ret_60m,
                ret_1d
            ],
            "technical": technical,
            "options": options_features,
            "fundamental": fundamental,
            "text": text_embedding,
            "asset_calendar": [
                if next_expiry_days <= 0.0 { 0.0 } else { 1.0 / (1.0 + next_expiry_days) },
                0.0,
                earnings_week
            ]
        });

        println!("{}", serde_json::to_string(&row)?);
    }

    Ok(())
}
