//! Historical-mode Block 5 (Risk Analytics) computation.
//!
//! In live mode `RiskData` comes from `ticker.risk(Interval, TimeRange,
//! benchmark)` and is refreshed daily by `src/poller/risk.rs`. For
//! historical-mode shards we need point-in-time risk values per bar, which
//! the live API can't return without N=`bars.len()` separate calls. So we
//! compute the 8 spec §6.2 Block 5 metrics ourselves from the trailing
//! 252-day window of OHLCV — this is what the live `ticker.risk()` API
//! does internally.
//!
//! Spec §6.2 Block 5 (line 552-559) lists:
//!   - VaR 95% / 99% (1d)         → empirical percentiles of daily returns
//!   - Parametric VaR 95%         → -Φ⁻¹(0.05) × σ (Gaussian assumption)
//!   - Sharpe (1yr)               → mean / std × √252 (excess over 0)
//!   - Sortino (1yr)              → mean / downside_std × √252
//!   - Beta vs SPY                → cov(asset, spy) / var(spy)
//!   - Max drawdown (1yr)         → (peak - trough) / peak
//!   - Max DD recovery periods    → bars from trough back to peak

use crate::store::{Bar, RiskData};

/// Sliding-window length matching spec §6.2 line 548 ("OneYear").
pub const RISK_WINDOW: usize = 252;

/// Compute per-bar risk metrics from the trailing window of asset bars
/// aligned to the trailing window of SPY bars. Inputs are slices ending
/// at the bar of interest (most recent last). Returns zeros if either
/// window has fewer than ~30 observations — the metrics are noise-dominated
/// at that sample size.
pub fn compute_risk_from_bars(asset_bars: &[Bar], spy_bars: &[Bar]) -> RiskData {
    let asset_returns = log_returns(asset_bars);
    let spy_returns = log_returns(spy_bars);
    if asset_returns.len() < 30 {
        return RiskData::default();
    }

    let var_95 = empirical_var(&asset_returns, 0.05);
    let var_99 = empirical_var(&asset_returns, 0.01);
    let (mean_r, std_r) = mean_std(&asset_returns);
    // Gaussian-VaR at 95%: VaR = μ - 1.645σ (one-sided). We report the loss
    // magnitude as a positive number, matching `parametric_var_95` field
    // convention used in the live risk poller.
    let parametric_var_95 = -(mean_r - 1.6449 * std_r);

    let sharpe = if std_r > 1e-9 {
        mean_r / std_r * (252.0_f64).sqrt()
    } else {
        0.0
    };
    let downside_std = downside_std(&asset_returns);
    let sortino = if downside_std > 1e-9 {
        mean_r / downside_std * (252.0_f64).sqrt()
    } else {
        0.0
    };

    let beta = compute_beta(&asset_returns, &spy_returns);

    let (max_dd, recovery) = max_drawdown_and_recovery(asset_bars);

    RiskData {
        var_95,
        var_99,
        parametric_var_95,
        sharpe,
        sortino,
        beta,
        max_drawdown: max_dd,
        max_dd_recovery: recovery as f64,
    }
}

fn log_returns(bars: &[Bar]) -> Vec<f64> {
    if bars.len() < 2 {
        return Vec::new();
    }
    bars.windows(2)
        .filter_map(|w| {
            if w[0].close > 0.0 && w[1].close > 0.0 {
                Some((w[1].close / w[0].close).ln())
            } else {
                None
            }
        })
        .collect()
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn downside_std(xs: &[f64]) -> f64 {
    let mut downs: Vec<f64> = xs.iter().filter(|x| **x < 0.0).copied().collect();
    if downs.is_empty() {
        return 0.0;
    }
    let n = downs.len() as f64;
    // Population-std of negative-return subset; matches Sortino convention.
    let mean = downs.iter().sum::<f64>() / n;
    downs.iter_mut().for_each(|x| *x = (*x - mean).powi(2));
    (downs.iter().sum::<f64>() / n).sqrt()
}

fn empirical_var(returns: &[f64], alpha: f64) -> f64 {
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((alpha * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);
    // Convert quantile loss to positive-magnitude convention.
    -sorted[idx]
}

fn compute_beta(asset_returns: &[f64], spy_returns: &[f64]) -> f64 {
    // Align to common length (slice from the right so the most recent bars
    // are paired up). Asset and SPY may have slightly different bar counts
    // because of holiday calendars or missing data.
    let n = asset_returns.len().min(spy_returns.len());
    if n < 20 {
        return 1.0; // neutral default when sample too small
    }
    let a = &asset_returns[asset_returns.len() - n..];
    let s = &spy_returns[spy_returns.len() - n..];
    let (mean_a, _) = mean_std(a);
    let (mean_s, std_s) = mean_std(s);
    if std_s < 1e-9 {
        return 1.0;
    }
    let cov: f64 = a
        .iter()
        .zip(s.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_s))
        .sum::<f64>()
        / n as f64;
    cov / (std_s * std_s)
}

fn max_drawdown_and_recovery(bars: &[Bar]) -> (f64, usize) {
    if bars.len() < 2 {
        return (0.0, 1);
    }
    let mut peak = bars[0].close;
    let mut peak_idx = 0usize;
    let mut max_dd = 0.0_f64;
    let mut max_dd_trough_idx = 0usize;
    let mut max_dd_peak_idx = 0usize;

    for (i, bar) in bars.iter().enumerate() {
        if bar.close > peak {
            peak = bar.close;
            peak_idx = i;
        }
        if peak > 0.0 {
            let dd = (peak - bar.close) / peak;
            if dd > max_dd {
                max_dd = dd;
                max_dd_trough_idx = i;
                max_dd_peak_idx = peak_idx;
            }
        }
    }

    // Recovery: bars from trough until close again ≥ peak. If no recovery
    // observed within the window, return remaining bars (clipped).
    let recovery = bars
        .iter()
        .enumerate()
        .skip(max_dd_trough_idx)
        .find(|(_, b)| b.close >= bars[max_dd_peak_idx].close)
        .map(|(i, _)| i - max_dd_trough_idx)
        .unwrap_or(bars.len() - max_dd_trough_idx)
        .max(1);

    (max_dd, recovery)
}
