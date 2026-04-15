use crate::store::{CachedValue, FundamentalData, RollingStatsStore};
use dashmap::DashMap;
use std::sync::Arc;

/// Block 4: Fundamental & Ownership — 30 features.
///
/// Updated quarterly (or daily for analyst/insider data).
/// Uses cross-sectional percentile ranks for PE, EV/EBITDA, debt/equity, mktcap.
pub fn assemble(
    fundamentals: &Arc<DashMap<String, CachedValue<FundamentalData>>>,
    symbol: &str,
    stats: &Arc<RollingStatsStore>,
) -> [f32; 30] {
    let Some(cached) = fundamentals.get(symbol) else {
        return [0.0f32; 30];
    };
    let d = &cached.value;

    // Helper: optional f64 → f32 with default
    let opt = |v: Option<f64>, default: f32| v.map(|x| x as f32).unwrap_or(default);
    let zs = |feat: &str, v: Option<f64>| -> f32 {
        match v {
            Some(x) => stats.zscore(symbol, feat, x),
            None => 0.0,
        }
    };

    // Cross-sectional percentile rank (approximate via z-score across the symbol universe)
    let pct = |feat: &str, v: Option<f64>| -> f32 {
        match v {
            Some(x) => {
                let z = stats.zscore(symbol, feat, x);
                // Convert z-score to approximate percentile [0,1] via sigmoid
                (1.0 / (1.0 + (-z * 0.4).exp())) as f32
            }
            None => 0.5, // neutral when missing
        }
    };

    // Clamp helper
    let cl = |v: f32, lo: f32, hi: f32| v.clamp(lo, hi);

    [
        // ── Financial Fundamentals (12) ───────────────────────────────────────
        pct("trailing_pe", d.trailing_pe),
        pct("forward_pe", d.forward_pe),
        cl(opt(d.peg_ratio, 2.5) / 5.0, 0.0, 1.0), // PEG [0,5]→[0,1]
        pct("ev_to_ebitda", d.ev_to_ebitda),
        cl(opt(d.dividend_yield, 0.0), 0.0, 0.2), // raw yield
        cl(opt(d.profit_margins, 0.0), -1.0, 1.0), // raw [-1,1]
        cl(opt(d.revenue_growth, 0.0), -1.0, 1.0), // winsorized ±100%
        cl(opt(d.earnings_growth, 0.0), -1.0, 1.0),
        pct("debt_to_equity", d.debt_to_equity),
        zs(
            "fcf_yield",
            d.free_cashflow
                .zip(d.market_cap)
                .map(|(f, m)| if m > 0.0 { f / m } else { 0.0 }),
        ),
        cl(opt(d.return_on_equity, 0.0), -1.0, 1.0),
        cl(opt(d.current_ratio, 0.0) / 5.0, 0.0, 1.0), // [0,5]→[0,1]
        // ── Earnings Intelligence (6) ─────────────────────────────────────────
        zs("earnings_surprise", d.last_earnings_surprise_pct),
        cl(d.earnings_surprise_streak as f32 / 5.0, -1.0, 1.0),
        zs("est_rev_30d", d.next_q_est_revision_30d),
        zs("est_rev_90d", d.next_q_est_revision_90d),
        zs("rev_est_rev_30d", d.rev_est_revision_30d),
        opt(d.days_since_earnings.map(|d| 1.0 / (1.0 + d)), 0.0),
        // ── Analyst & Ratings (4) ─────────────────────────────────────────────
        cl(opt(d.analyst_consensus, 0.0), -1.0, 1.0),
        cl(opt(d.strong_buy_fraction, 0.0), 0.0, 1.0),
        zs("rating_momentum_30d", d.rating_momentum_30d),
        cl(d.recent_grade_direction as f32, -1.0, 1.0),
        // ── Ownership & Insider Activity (6) ─────────────────────────────────
        zs("net_insider_shares", d.net_insider_shares_90d),
        cl(
            (d.insider_buy_count_90d.ln() as f32).max(0.0) / 5.0,
            0.0,
            1.0,
        ),
        zs("net_insider_value", d.net_insider_value_90d),
        cl(opt(d.insider_ownership_pct, 0.0), 0.0, 1.0),
        cl(opt(d.institutional_pct, 0.0), 0.0, 1.0),
        zs("inst_delta", d.institutional_delta_qtr),
        // ── Asset Identity (2) ───────────────────────────────────────────────
        cl(d.sector_idx as f32 / 10.0, 0.0, 1.0), // sector scalar
        cl(d.mktcap_log_rank as f32, 0.0, 1.0),   // log-rank percentile
    ]
}
