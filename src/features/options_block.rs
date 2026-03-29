use crate::store::{CachedValue, OptionsData, RollingStatsStore};
use dashmap::DashMap;
use finance_query::OptionContract;
use std::sync::Arc;

/// Block 3: Options-Derived — 12 features.
///
/// Sourced from `ticker.options()` polled every 15 minutes.
/// Assets without listed options return zeros.
pub fn assemble(
    options_cache: &Arc<DashMap<String, CachedValue<OptionsData>>>,
    symbol: &str,
    stats: &Arc<RollingStatsStore>,
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

    // ── IV metrics ────────────────────────────────────────────────────────────

    // 30-day ATM IV: average IV of near-money options
    let atm_iv = atm_implied_vol(&all_calls, &all_puts);

    // IV percentile (rolling z-score as proxy for now)
    let iv_pct = stats.zscore(symbol, "atm_iv", atm_iv);

    // IV term structure: not available without per-expiry chains (stub)
    let term_slope: f32 = 0.0;

    // 25-delta skew approximations
    let (put_skew, call_skew) = skew_approx(&all_puts, &all_calls, atm_iv);

    // ── Volume / OI metrics ───────────────────────────────────────────────────

    let call_vol: f64 = all_calls.iter().filter_map(|c| c.volume).map(|v| v as f64).sum();
    let put_vol: f64 = all_puts.iter().filter_map(|c| c.volume).map(|v| v as f64).sum();
    let call_oi: f64 = all_calls.iter().filter_map(|c| c.open_interest).map(|v| v as f64).sum();
    let put_oi: f64 = all_puts.iter().filter_map(|c| c.open_interest).map(|v| v as f64).sum();
    let total_vol = call_vol + put_vol;
    let total_oi = call_oi + put_oi;

    let pc_vol = if call_vol > 0.0 { (put_vol / call_vol).ln().clamp(-3.0, 3.0) as f32 } else { 0.0 };
    let pc_oi = if call_oi > 0.0 { (put_oi / call_oi).ln().clamp(-3.0, 3.0) as f32 } else { 0.0 };

    // Total OI change (z-scored day-over-day)
    let oi_z = stats.zscore(symbol, "total_oi", total_oi);

    // Volume/OI ratio
    let vol_oi_ratio = if total_oi > 0.0 {
        (total_vol / total_oi).clamp(0.0, 5.0) as f32
    } else {
        0.0
    };

    // Unusual volume flag (binary)
    let vol_mean = stats.get(symbol, "opt_vol_mean").mean;
    let vol_std = stats.get(symbol, "opt_vol_mean").std_dev();
    let unusual_vol: f32 = if vol_std > 0.0 && (total_vol - vol_mean) / vol_std > 2.0 { 1.0 } else { 0.0 };
    stats.update(symbol, "opt_vol_mean", total_vol);

    // Max pain proximity (stub: 0 without current price)
    let max_pain_prox: f32 = 0.0;

    // IV crush proximity (stub: 0 without earnings date)
    let iv_crush_prox: f32 = 0.0;

    [
        atm_iv.clamp(0.0, 2.0) as f32, // 30-day ATM IV
        iv_pct,                          // IV percentile (z-scored)
        term_slope,                      // IV term structure slope
        put_skew,                        // 25-delta put skew
        call_skew,                       // 25-delta call skew
        pc_vol,                          // Put/call volume ratio (log)
        pc_oi,                           // Put/call OI ratio (log)
        oi_z,                            // Total OI change z-score
        vol_oi_ratio,                    // Volume/OI ratio
        max_pain_prox,                   // Max pain proximity
        unusual_vol,                     // Unusual volume flag
        iv_crush_prox,                   // IV crush proximity
    ]
}

fn atm_implied_vol(calls: &[&OptionContract], puts: &[&OptionContract]) -> f64 {
    let ivs: Vec<f64> = calls
        .iter()
        .chain(puts.iter())
        .filter_map(|c| c.implied_volatility)
        .filter(|&iv| iv > 0.0 && iv < 10.0)
        .collect();
    if ivs.is_empty() { 0.0 } else { ivs.iter().sum::<f64>() / ivs.len() as f64 }
}

/// Approximate 25-delta skew using lower/upper quartile of strikes by IV.
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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (sorted.len() as f64 * 0.75) as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}
