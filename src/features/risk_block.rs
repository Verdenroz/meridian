use crate::store::{CachedValue, RiskData};
use dashmap::DashMap;
use std::sync::Arc;

/// Block 5: Risk Analytics — 8 features.
///
/// Sourced from `ticker.risk()`, refreshed daily.
pub fn assemble(
    risk_cache: &Arc<DashMap<String, CachedValue<RiskData>>>,
    symbol: &str,
) -> [f32; 8] {
    let Some(cached) = risk_cache.get(symbol) else {
        return [0.0f32; 8];
    };
    let d = &cached.value;

    [
        d.var_95 as f32,
        d.var_99 as f32,
        d.parametric_var_95 as f32,
        (d.sharpe as f32).clamp(-3.0, 3.0),
        (d.sortino as f32).clamp(-3.0, 3.0),
        d.beta as f32,
        d.max_drawdown as f32,                           // [0,1]
        (d.max_dd_recovery.ln() as f32).max(0.0) / 5.0, // log-scaled recovery periods
    ]
}
