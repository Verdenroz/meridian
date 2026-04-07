//! Pre-built benchmark strategies exposed to Python.
//!
//! These wrap finance-query's prebuilt strategies (SMA crossover, RSI reversal,
//! Bollinger) plus a simple buy-and-hold, so the Meridian model can be compared
//! against standard baselines using the same backtesting engine.

use std::collections::HashMap;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use finance_query::Candle;
use finance_query::Indicator;
use finance_query::backtesting::{
    BacktestConfig, Signal,
    condition::HtfIndicatorSpec,
    strategy::{Strategy, StrategyContext, SmaCrossover, RsiReversal, BollingerMeanReversion},
    portfolio::{PortfolioEngine, PortfolioConfig, RebalanceMode, SymbolData},
};

use super::results::PyPortfolioResult;
use super::strategy::make_candle;
use super::types::PyPortfolioConfig;

// ---------------------------------------------------------------------------
// BuyAndHold: simplest possible baseline
// ---------------------------------------------------------------------------

/// Enter long on the first bar and hold until close.
#[derive(Clone)]
struct BuyAndHold;

impl Strategy for BuyAndHold {
    fn name(&self) -> &str { "BuyAndHold" }

    fn required_indicators(&self) -> Vec<(String, Indicator)> { vec![] }
    fn htf_requirements(&self) -> Vec<HtfIndicatorSpec> { vec![] }

    fn on_candle(&self, ctx: &StrategyContext) -> Signal {
        if ctx.index == 0 && !ctx.has_position() {
            Signal::long(ctx.timestamp(), ctx.close())
        } else {
            Signal::hold()
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: parse candle dict shared across all benchmark functions
// ---------------------------------------------------------------------------

fn parse_candle_dict(
    candles: &HashMap<String, PyReadonlyArray2<f64>>,
) -> PyResult<Vec<SymbolData>> {
    let mut data = Vec::with_capacity(candles.len());
    for (sym, arr) in candles {
        let a = arr.as_array();
        let rows = a.dim().0;
        let parsed: Vec<Candle> = (0..rows)
            .map(|i| make_candle(
                a[[i, 0]] as i64, a[[i, 1]], a[[i, 2]],
                a[[i, 3]], a[[i, 4]], a[[i, 5]] as i64,
            ))
            .collect();
        data.push(SymbolData::new(sym.clone(), parsed));
    }
    Ok(data)
}

fn default_portfolio_config() -> PortfolioConfig {
    let base = BacktestConfig::builder()
        .initial_capital(1_000_000.0)
        .commission_pct(0.0010)
        .slippage_pct(0.0005)
        .bars_per_year(252.0 * 390.0)
        .position_size_pct(1.0)
        .close_at_end(true)
        .build()
        .expect("default config is always valid");
    PortfolioConfig::new(base)
        .max_total_positions(100)
        .rebalance(RebalanceMode::AvailableCapital)
}

// ---------------------------------------------------------------------------
// Python-facing benchmark functions
// ---------------------------------------------------------------------------

/// Run buy-and-hold benchmark across all symbols.
#[pyfunction]
#[pyo3(signature = (candles, config = None))]
pub fn run_buy_and_hold(
    py: Python<'_>,
    candles: HashMap<String, PyReadonlyArray2<f64>>,
    config: Option<PyPortfolioConfig>,
) -> PyResult<PyPortfolioResult> {
    let symbol_data = parse_candle_dict(&candles)?;
    let cfg = config.map(|c| c.inner).unwrap_or_else(default_portfolio_config);
    let engine = PortfolioEngine::new(cfg);

    let result = py.allow_threads(move || {
        engine.run(&symbol_data, |_sym: &str| BuyAndHold)
    });

    match result {
        Ok(r) => Ok(PyPortfolioResult { inner: r }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
    }
}

/// Run SMA crossover benchmark across all symbols.
#[pyfunction]
#[pyo3(signature = (candles, fast_period = 10, slow_period = 30, config = None))]
pub fn run_sma_crossover(
    py: Python<'_>,
    candles: HashMap<String, PyReadonlyArray2<f64>>,
    fast_period: usize,
    slow_period: usize,
    config: Option<PyPortfolioConfig>,
) -> PyResult<PyPortfolioResult> {
    let symbol_data = parse_candle_dict(&candles)?;
    let cfg = config.map(|c| c.inner).unwrap_or_else(default_portfolio_config);
    let engine = PortfolioEngine::new(cfg);

    let result = py.allow_threads(move || {
        engine.run(&symbol_data, |_sym: &str| {
            SmaCrossover::new(fast_period, slow_period)
        })
    });

    match result {
        Ok(r) => Ok(PyPortfolioResult { inner: r }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
    }
}

/// Run RSI reversal benchmark across all symbols.
#[pyfunction]
#[pyo3(signature = (candles, period = 14, oversold = 30.0, overbought = 70.0, config = None))]
pub fn run_rsi_reversal(
    py: Python<'_>,
    candles: HashMap<String, PyReadonlyArray2<f64>>,
    period: usize,
    oversold: f64,
    overbought: f64,
    config: Option<PyPortfolioConfig>,
) -> PyResult<PyPortfolioResult> {
    let symbol_data = parse_candle_dict(&candles)?;
    let cfg = config.map(|c| c.inner).unwrap_or_else(default_portfolio_config);
    let engine = PortfolioEngine::new(cfg);

    let result = py.allow_threads(move || {
        engine.run(&symbol_data, |_sym: &str| {
            let mut strat = RsiReversal::new(period);
            strat.oversold = oversold;
            strat.overbought = overbought;
            strat
        })
    });

    match result {
        Ok(r) => Ok(PyPortfolioResult { inner: r }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
    }
}

/// Run Bollinger Bands mean reversion benchmark across all symbols.
#[pyfunction]
#[pyo3(signature = (candles, period = 20, std_dev = 2.0, config = None))]
pub fn run_bollinger(
    py: Python<'_>,
    candles: HashMap<String, PyReadonlyArray2<f64>>,
    period: usize,
    std_dev: f64,
    config: Option<PyPortfolioConfig>,
) -> PyResult<PyPortfolioResult> {
    let symbol_data = parse_candle_dict(&candles)?;
    let cfg = config.map(|c| c.inner).unwrap_or_else(default_portfolio_config);
    let engine = PortfolioEngine::new(cfg);

    let result = py.allow_threads(move || {
        engine.run(&symbol_data, |_sym: &str| {
            BollingerMeanReversion::new(period, std_dev)
        })
    });

    match result {
        Ok(r) => Ok(PyPortfolioResult { inner: r }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
    }
}
