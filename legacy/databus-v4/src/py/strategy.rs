//! Bridge between Meridian's GPU-produced weight matrix and finance-query's
//! signal-based Strategy trait.
//!
//! The model outputs a (T, N) matrix of target portfolio weights in Python.
//! This module converts weight deltas into ScaleIn/ScaleOut/Long/Exit signals
//! so the Rust backtesting engine handles all execution simulation — commissions,
//! slippage, position tracking — without GIL contention.

use std::collections::HashMap;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use finance_query::Candle;
use finance_query::Indicator;
use finance_query::backtesting::{
    BacktestConfig, Signal,
    condition::HtfIndicatorSpec,
    portfolio::{PortfolioConfig, PortfolioEngine, RebalanceMode, SymbolData},
    strategy::Strategy,
    strategy::StrategyContext,
};

/// Construct a Candle from raw values.
/// Candle is #[non_exhaustive] so we deserialize via serde_json.
pub(super) fn make_candle(
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: i64,
) -> Candle {
    serde_json::from_value(serde_json::json!({
        "timestamp": timestamp,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "adjClose": null
    }))
    .expect("candle deserialization is infallible for valid OHLCV")
}

use super::types::PyPortfolioConfig;

// Weight below this is treated as "no position desired".
const MIN_WEIGHT: f64 = 1e-4;

// Ignore weight deltas smaller than this to avoid excessive rebalancing.
const REBAL_THRESHOLD: f64 = 0.01;

/// Per-symbol strategy that looks up pre-computed target weights by bar index
/// and emits position-sizing signals via ScaleIn/ScaleOut.
#[derive(Clone)]
struct MeridianStrategy {
    weights: Vec<f64>,
    min_weight: f64,
    rebal_threshold: f64,
}

impl Strategy for MeridianStrategy {
    fn name(&self) -> &str {
        "MeridianWeights"
    }

    fn required_indicators(&self) -> Vec<(String, Indicator)> {
        // All alpha comes from the model; no technical indicators needed.
        vec![]
    }

    fn htf_requirements(&self) -> Vec<HtfIndicatorSpec> {
        vec![]
    }

    fn on_candle(&self, ctx: &StrategyContext) -> Signal {
        let idx = ctx.index;
        let target = self.weights.get(idx).copied().unwrap_or(0.0);

        // Estimate current allocation as position value / portfolio equity.
        // When flat, position is None and current = 0.
        let current = ctx.position.map_or(0.0, |pos| {
            let pos_value = pos.quantity * ctx.close();
            if ctx.equity > 0.0 {
                pos_value / ctx.equity
            } else {
                0.0
            }
        });

        if current <= 0.0 && target > self.min_weight {
            Signal::long(ctx.timestamp(), ctx.close())
        } else if current > 0.0 && target <= self.min_weight {
            Signal::exit(ctx.timestamp(), ctx.close())
        } else if current > 0.0 && target > self.min_weight {
            let delta = target - current;
            if delta > self.rebal_threshold {
                Signal::scale_in(delta / current, ctx.timestamp(), ctx.close())
            } else if delta < -self.rebal_threshold {
                Signal::scale_out((-delta) / current, ctx.timestamp(), ctx.close())
            } else {
                Signal::hold()
            }
        } else {
            Signal::hold()
        }
    }

    fn warmup_period(&self) -> usize {
        0
    }
}

/// Run a portfolio backtest using pre-computed model weights.
///
/// Args:
///     weights: numpy float64 array of shape (T, N) — target portfolio weights
///              per bar per symbol. Rows should sum to ≤1.0.
///     symbols: list of N ticker strings, column-aligned with `weights`.
///     candles: dict mapping each symbol to a (T, 6) numpy array of
///              [timestamp, open, high, low, close, volume].
///     config:  optional PortfolioConfig (defaults to spec values).
#[pyfunction]
#[pyo3(signature = (weights, symbols, candles, config = None))]
pub fn run_backtest(
    py: Python<'_>,
    weights: PyReadonlyArray2<f64>,
    symbols: Vec<String>,
    candles: HashMap<String, PyReadonlyArray2<f64>>,
    config: Option<PyPortfolioConfig>,
) -> PyResult<super::results::PyPortfolioResult> {
    let weight_arr = weights.as_array();
    let (t_bars, n_assets) = weight_arr.dim();

    if n_assets != symbols.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "weights has {} columns but {} symbols provided",
            n_assets,
            symbols.len()
        )));
    }

    // Build per-symbol weight columns and candle data outside the GIL release.
    let mut symbol_weights: HashMap<String, Vec<f64>> = HashMap::new();
    let mut symbol_data: Vec<SymbolData> = Vec::with_capacity(n_assets);

    for (col, sym) in symbols.iter().enumerate() {
        let col_weights: Vec<f64> = (0..t_bars).map(|row| weight_arr[[row, col]]).collect();
        symbol_weights.insert(sym.clone(), col_weights);

        let candle_arr = candles
            .get(sym)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "no candle data for symbol '{}'",
                    sym
                ))
            })?
            .as_array();

        let candle_rows = candle_arr.dim().0;
        if candle_rows != t_bars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "symbol '{}' has {} candle rows but weights has {} rows",
                sym, candle_rows, t_bars
            )));
        }

        let parsed_candles: Vec<Candle> = (0..candle_rows)
            .map(|i| {
                make_candle(
                    candle_arr[[i, 0]] as i64,
                    candle_arr[[i, 1]],
                    candle_arr[[i, 2]],
                    candle_arr[[i, 3]],
                    candle_arr[[i, 4]],
                    candle_arr[[i, 5]] as i64,
                )
            })
            .collect();

        symbol_data.push(SymbolData::new(sym.clone(), parsed_candles));
    }

    let portfolio_config = config.map(|c| c.inner).unwrap_or_else(|| {
        let base = BacktestConfig::builder()
            .initial_capital(1_000_000.0)
            .commission_pct(0.0010)
            .slippage_pct(0.0005)
            .bars_per_year(252.0)
            .position_size_pct(1.0)
            .close_at_end(true)
            .build()
            .expect("default config is always valid");
        PortfolioConfig::new(base)
            .max_total_positions(100)
            .rebalance(RebalanceMode::AvailableCapital)
    });

    let engine = PortfolioEngine::new(portfolio_config);

    // Release the GIL for the entire rayon-parallel backtest.
    let result = py.allow_threads(move || {
        engine.run(&symbol_data, |sym: &str| {
            let w = symbol_weights.get(sym).cloned().unwrap_or_default();
            MeridianStrategy {
                weights: w,
                min_weight: MIN_WEIGHT,
                rebal_threshold: REBAL_THRESHOLD,
            }
        })
    });

    match result {
        Ok(portfolio_result) => Ok(super::results::PyPortfolioResult {
            inner: portfolio_result,
        }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
    }
}
