use crate::events::detector::detect_macro_release;
use crate::store::{CachedValue, DataStore, MacroData, YieldCurveData};
use chrono::Datelike;
use finance_query::fred;
use std::sync::Arc;

/// 12 FRED series IDs in the order defined in spec Section 6.3.
const FRED_SERIES: &[&str] = &[
    "FEDFUNDS", // Fed Funds Rate
    "UNRATE",   // Unemployment Rate
    "CPIAUCSL", // CPI YoY
    "CPILFESL", // Core CPI YoY
    "ICSA",     // Initial Jobless Claims
    "CCSA",     // Continuing Claims
    "MANEMP",   // ISM Manufacturing proxy
    "RSXFS",    // Retail Sales MoM
    "INDPRO",   // Industrial Production
    "HOUST",    // Housing Starts
    "UMCSENT",  // Consumer Confidence
    "USSLIND",  // Leading Economic Index
];

/// Poll yield curve from FRED (runs every 24 hours).
pub async fn poll_yield_curve(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let current_year = chrono::Utc::now().year() as u32;
    let yields = fred::treasury_yields(current_year).await?;

    // Use the most recent yield data point (yields are non-empty if the request succeeded)
    let Some(latest) = yields.last() else {
        tracing::warn!("No yield data returned for year {current_year}");
        return Ok(());
    };

    let y = YieldCurveData {
        y1m: latest.y1m.unwrap_or(0.0),
        y3m: latest.y3m.unwrap_or(0.0),
        y6m: latest.y6m.unwrap_or(0.0),
        y1: latest.y1.unwrap_or(0.0),
        y2: latest.y2.unwrap_or(0.0),
        y3: latest.y3.unwrap_or(0.0),
        y5: latest.y5.unwrap_or(0.0),
        y7: latest.y7.unwrap_or(0.0),
        y10: latest.y10.unwrap_or(0.0),
        y20: latest.y20.unwrap_or(0.0),
        y30: latest.y30.unwrap_or(0.0),
        spread_2y10y: latest.y10.unwrap_or(0.0) - latest.y2.unwrap_or(0.0),
    };

    *store.yield_curve.write() = CachedValue::new(y);
    tracing::debug!("Yield curve updated");
    Ok(())
}

/// Poll all 12 FRED macro series (runs on release / daily).
pub async fn poll_fred_macro(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let mut values = [0.0f64; 12];
    let mut dates = std::array::from_fn(|_| String::new());

    let old_macro = store.fred_macro.read().value.clone();

    for (i, &series_id) in FRED_SERIES.iter().enumerate() {
        match fred::series(series_id).await {
            Ok(series) => {
                if let Some(latest) = series.observations.last() {
                    let new_value = latest.value.unwrap_or(old_macro.values[i]);
                    let new_date = latest.date.clone();
                    let old_date = &old_macro.last_observation_dates[i];
                    let old_value = old_macro.values[i];

                    // Detect macro release event
                    let hist_std = store.rolling_stats.get(series_id, "value").std_dev();
                    if let Some(event) = detect_macro_release(
                        series_id, old_date, &new_date, new_value, old_value, hist_std,
                    ) {
                        tracing::info!(series = %series_id, "MacroRelease event");
                        for sym in &store.symbols {
                            let mut es = store.event_state.entry(sym.clone()).or_default();
                            *es = es.with_event(&event);
                        }
                    }

                    store.rolling_stats.update(series_id, "value", new_value);
                    values[i] = new_value;
                    dates[i] = new_date;
                }
            }
            Err(e) => {
                tracing::warn!(series = %series_id, err = %e, "FRED series poll failed");
                values[i] = old_macro.values[i];
                dates[i] = old_macro.last_observation_dates[i].clone();
            }
        }
    }

    let macro_data = MacroData {
        values,
        last_observation_dates: dates,
    };
    *store.fred_macro.write() = CachedValue::new(macro_data);
    tracing::debug!("FRED macro updated");
    Ok(())
}
