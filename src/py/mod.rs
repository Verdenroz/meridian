//! PyO3 bindings for finance-query's backtesting engine.
//!
//! Exposes the Rust-native portfolio backtest, walk-forward validation,
//! and Monte Carlo simulation to Python so the Meridian model's GPU-produced
//! weight matrices can be evaluated without GIL contention.

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod types;
pub mod strategy;
pub mod results;
pub mod benchmarks;
pub mod comparison;
pub mod validation;

/// Python module registered as `meridian_py`.
#[cfg(feature = "python")]
#[pymodule]
fn meridian_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Config types
    m.add_class::<types::PyBacktestConfig>()?;
    m.add_class::<types::PyPortfolioConfig>()?;

    // Result types
    m.add_class::<results::PyPerformanceMetrics>()?;
    m.add_class::<results::PyPortfolioResult>()?;
    m.add_class::<results::PyBacktestResult>()?;

    // Comparison & validation result types
    m.add_class::<comparison::PyComparisonReport>()?;
    m.add_class::<validation::PyPercentileStats>()?;
    m.add_class::<validation::PyMonteCarloResult>()?;

    // Core backtest
    m.add_function(wrap_pyfunction!(strategy::run_backtest, m)?)?;

    // Benchmarks
    m.add_function(wrap_pyfunction!(benchmarks::run_buy_and_hold, m)?)?;
    m.add_function(wrap_pyfunction!(benchmarks::run_sma_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(benchmarks::run_rsi_reversal, m)?)?;
    m.add_function(wrap_pyfunction!(benchmarks::run_bollinger, m)?)?;

    // Comparison
    m.add_function(wrap_pyfunction!(comparison::run_comparison, m)?)?;

    // Validation
    m.add_function(wrap_pyfunction!(validation::run_monte_carlo, m)?)?;

    Ok(())
}
