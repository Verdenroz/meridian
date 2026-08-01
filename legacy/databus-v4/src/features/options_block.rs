use crate::store::{CachedValue, OptionsData, PriceBuffer, RollingStatsStore};
use dashmap::DashMap;
use finance_query::OptionContract;
use std::sync::Arc;

/// Block 3: Options-Derived — 12 features.
///
/// Sourced from `ticker.options()` polled every 15 minutes.
/// Assets without listed options return zeros.
pub fn assemble(
    options_cache: &Arc<DashMap<String, CachedValue<OptionsData>>>,
    prices: &Arc<DashMap<String, PriceBuffer>>,
    symbol: &str,
    stats: &Arc<RollingStatsStore>,
    earnings_proximity: f32,
) -> [f32; 12] {
    let Some(cached) = options_cache.get(symbol) else {
        return [0.0f32; 12];
    };
    let calls = &cached.value.calls;
    let puts = &cached.value.puts;
    if calls.is_empty() && puts.is_empty() {
        return [0.0f32; 12];
    }

    let all_calls: Vec<&OptionContract> = calls.iter().collect();
    let all_puts: Vec<&OptionContract> = puts.iter().collect();

    if all_calls.is_empty() && all_puts.is_empty() {
        return [0.0f32; 12];
    }

    let now_ts = chrono::Utc::now().timestamp();

    // ── IV metrics ────────────────────────────────────────────────────────────

    let atm_iv = atm_implied_vol(&all_calls, &all_puts);
    let iv_pct = stats.zscore(symbol, "atm_iv", atm_iv);

    // IV term structure: near-term (≤45 DTE) vs far-term (>45 DTE) IV spread
    let term_slope: f32 = iv_term_slope(&all_calls, &all_puts, now_ts);

    let (put_skew, call_skew) = skew_approx(&all_puts, &all_calls, atm_iv);

    // ── Volume / OI metrics ───────────────────────────────────────────────────

    let call_vol: f64 = all_calls.iter().filter_map(|c| c.volume).map(|v| v as f64).sum();
    let put_vol: f64 = all_puts.iter().filter_map(|c| c.volume).map(|v| v as f64).sum();
    let call_oi: f64 = all_calls.iter().filter_map(|c| c.open_interest).map(|v| v as f64).sum();
    let put_oi: f64 = all_puts.iter().filter_map(|c| c.open_interest).map(|v| v as f64).sum();
    let total_vol = call_vol + put_vol;
    let total_oi = call_oi + put_oi;

    let pc_vol = if call_vol > 0.0 {
        (put_vol / call_vol).ln().clamp(-3.0, 3.0) as f32
    } else {
        0.0
    };
    let pc_oi = if call_oi > 0.0 {
        (put_oi / call_oi).ln().clamp(-3.0, 3.0) as f32
    } else {
        0.0
    };

    let oi_z = stats.zscore(symbol, "total_oi", total_oi);

    let vol_oi_ratio = if total_oi > 0.0 {
        (total_vol / total_oi).clamp(0.0, 5.0) as f32
    } else {
        0.0
    };

    let vol_mean = stats.get(symbol, "opt_vol_mean").mean;
    let vol_std = stats.get(symbol, "opt_vol_mean").std_dev();
    let unusual_vol: f32 = if vol_std > 0.0 && (total_vol - vol_mean) / vol_std > 2.0 {
        1.0
    } else {
        0.0
    };
    stats.update(symbol, "opt_vol_mean", total_vol);

    // Max pain: strike minimizing aggregate option-seller payout; proximity to current price
    let last_price = prices.get(symbol)
        .and_then(|b| b.closes(1).into_iter().next())
        .unwrap_or(0.0);
    let max_pain_prox: f32 = max_pain_proximity(&all_calls, &all_puts, last_price);

    // IV crush proximity: IV elevation × earnings imminence interaction
    let iv_crush_prox: f32 =
        ((atm_iv as f32 * earnings_proximity + 0.25 * vol_oi_ratio) / 3.0).clamp(0.0, 1.0);

    [
        atm_iv.clamp(0.0, 2.0) as f32,
        iv_pct,
        term_slope,
        put_skew,
        call_skew,
        pc_vol,
        pc_oi,
        oi_z,
        vol_oi_ratio,
        max_pain_prox,
        unusual_vol,
        iv_crush_prox,
    ]
}

/// Near-term (≤45 DTE) vs far-term (>45 DTE) average IV difference.
/// Positive = normal contango (far > near); negative = backwardation / stress.
fn iv_term_slope(calls: &[&OptionContract], puts: &[&OptionContract], now_ts: i64) -> f32 {
    let mut near_ivs: Vec<f64> = Vec::new();
    let mut far_ivs: Vec<f64> = Vec::new();

    for c in calls.iter().chain(puts.iter()) {
        let (Some(exp), Some(iv)) = (c.expiration, c.implied_volatility) else {
            continue;
        };
        if iv <= 0.0 || iv > 10.0 {
            continue;
        }
        let days = ((exp - now_ts) / 86400).max(0);
        if days <= 45 {
            near_ivs.push(iv);
        } else {
            far_ivs.push(iv);
        }
    }

    if near_ivs.is_empty() || far_ivs.is_empty() {
        return 0.0;
    }

    let mean_near = near_ivs.iter().sum::<f64>() / near_ivs.len() as f64;
    let mean_far = far_ivs.iter().sum::<f64>() / far_ivs.len() as f64;
    ((mean_far - mean_near) as f32).clamp(-2.0, 2.0)
}

/// Max pain = strike minimizing total dollar payout to option holders.
/// Returns |current_price - max_pain| / current_price, clamped to [0, 1].
fn max_pain_proximity(calls: &[&OptionContract], puts: &[&OptionContract], last_price: f64) -> f32 {
    if last_price <= 0.0 {
        return 0.0;
    }

    // Collect unique strikes from both sides
    let mut strikes: Vec<f64> = calls
        .iter()
        .chain(puts.iter())
        .map(|c| c.strike)
        .collect();
    strikes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    strikes.dedup_by(|a, b| (*a - *b).abs() < 0.01);

    if strikes.is_empty() {
        return 0.0;
    }

    // For each candidate strike S, total payout = Σ call_oi*(S-K)+ + Σ put_oi*(K-S)+
    let mut min_pain = f64::MAX;
    let mut max_pain_strike = last_price;

    for &s in &strikes {
        let call_pain: f64 = calls
            .iter()
            .map(|c| {
                let oi = c.open_interest.unwrap_or(0) as f64;
                oi * (s - c.strike).max(0.0) * 100.0
            })
            .sum();
        let put_pain: f64 = puts
            .iter()
            .map(|c| {
                let oi = c.open_interest.unwrap_or(0) as f64;
                oi * (c.strike - s).max(0.0) * 100.0
            })
            .sum();
        let total = call_pain + put_pain;
        if total < min_pain {
            min_pain = total;
            max_pain_strike = s;
        }
    }

    ((max_pain_strike - last_price).abs() / last_price).clamp(0.0, 1.0) as f32
}

fn atm_implied_vol(calls: &[&OptionContract], puts: &[&OptionContract]) -> f64 {
    let ivs: Vec<f64> = calls
        .iter()
        .chain(puts.iter())
        .filter_map(|c| c.implied_volatility)
        .filter(|&iv| iv > 0.0 && iv < 10.0)
        .collect();
    if ivs.is_empty() {
        0.0
    } else {
        ivs.iter().sum::<f64>() / ivs.len() as f64
    }
}

/// Approximate 25-delta skew using upper quartile of strikes by IV.
fn skew_approx(puts: &[&OptionContract], calls: &[&OptionContract], atm_iv: f64) -> (f32, f32) {
    if atm_iv <= 0.0 {
        return (0.0, 0.0);
    }
    let put_ivs: Vec<f64> = puts.iter().filter_map(|c| c.implied_volatility).collect();
    let call_ivs: Vec<f64> = calls.iter().filter_map(|c| c.implied_volatility).collect();

    let q_put = quartile_high(&put_ivs).unwrap_or(atm_iv);
    let q_call = quartile_high(&call_ivs).unwrap_or(atm_iv);

    let put_skew = ((q_put - atm_iv) / atm_iv).clamp(-2.0, 2.0) as f32;
    let call_skew = ((atm_iv - q_call) / atm_iv).clamp(-2.0, 2.0) as f32;
    (put_skew, call_skew)
}

fn quartile_high(vals: &[f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (sorted.len() as f64 * 0.75) as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}
