//! Python-facing wrappers for finance-query's backtest result types.
//!
//! Each wrapper holds the Rust struct and exposes key fields as Python
//! properties. Heavy serialization (e.g. full equity curve) goes through
//! `to_dict()` for pandas integration.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use finance_query::backtesting::{
    BacktestResult, PerformanceMetrics,
    portfolio::PortfolioResult,
};

// ---------------------------------------------------------------------------
// PerformanceMetrics
// ---------------------------------------------------------------------------

/// All computed performance metrics from a backtest run.
#[pyclass(name = "PerformanceMetrics")]
#[derive(Clone)]
pub struct PyPerformanceMetrics {
    pub(crate) inner: PerformanceMetrics,
}

#[pymethods]
impl PyPerformanceMetrics {
    // -- Core return metrics --
    #[getter] fn total_return_pct(&self) -> f64 { self.inner.total_return_pct }
    #[getter] fn annualized_return_pct(&self) -> f64 { self.inner.annualized_return_pct }

    // -- Risk-adjusted --
    #[getter] fn sharpe_ratio(&self) -> f64 { self.inner.sharpe_ratio }
    #[getter] fn sortino_ratio(&self) -> f64 { self.inner.sortino_ratio }
    #[getter] fn calmar_ratio(&self) -> f64 { self.inner.calmar_ratio }
    #[getter] fn omega_ratio(&self) -> f64 { self.inner.omega_ratio }
    #[getter] fn serenity_ratio(&self) -> f64 { self.inner.serenity_ratio }

    // -- Drawdown --
    #[getter] fn max_drawdown_pct(&self) -> f64 { self.inner.max_drawdown_pct }
    #[getter] fn max_drawdown_duration(&self) -> i64 { self.inner.max_drawdown_duration }
    #[getter] fn ulcer_index(&self) -> f64 { self.inner.ulcer_index }
    #[getter] fn recovery_factor(&self) -> f64 { self.inner.recovery_factor }

    // -- Trade statistics --
    #[getter] fn total_trades(&self) -> usize { self.inner.total_trades }
    #[getter] fn winning_trades(&self) -> usize { self.inner.winning_trades }
    #[getter] fn losing_trades(&self) -> usize { self.inner.losing_trades }
    #[getter] fn win_rate(&self) -> f64 { self.inner.win_rate }
    #[getter] fn profit_factor(&self) -> f64 { self.inner.profit_factor }
    #[getter] fn avg_trade_return_pct(&self) -> f64 { self.inner.avg_trade_return_pct }
    #[getter] fn avg_win_pct(&self) -> f64 { self.inner.avg_win_pct }
    #[getter] fn avg_loss_pct(&self) -> f64 { self.inner.avg_loss_pct }
    #[getter] fn avg_trade_duration(&self) -> f64 { self.inner.avg_trade_duration }
    #[getter] fn largest_win(&self) -> f64 { self.inner.largest_win }
    #[getter] fn largest_loss(&self) -> f64 { self.inner.largest_loss }
    #[getter] fn max_consecutive_wins(&self) -> usize { self.inner.max_consecutive_wins }
    #[getter] fn max_consecutive_losses(&self) -> usize { self.inner.max_consecutive_losses }
    #[getter] fn long_trades(&self) -> usize { self.inner.long_trades }
    #[getter] fn short_trades(&self) -> usize { self.inner.short_trades }

    // -- Costs & signals --
    #[getter] fn total_commission(&self) -> f64 { self.inner.total_commission }
    #[getter] fn total_signals(&self) -> usize { self.inner.total_signals }
    #[getter] fn executed_signals(&self) -> usize { self.inner.executed_signals }
    #[getter] fn time_in_market_pct(&self) -> f64 { self.inner.time_in_market_pct }

    // -- Advanced --
    #[getter] fn kelly_criterion(&self) -> f64 { self.inner.kelly_criterion }
    #[getter] fn sqn(&self) -> f64 { self.inner.sqn }
    #[getter] fn expectancy(&self) -> f64 { self.inner.expectancy }
    #[getter] fn tail_ratio(&self) -> f64 { self.inner.tail_ratio }
    #[getter] fn total_dividend_income(&self) -> f64 { self.inner.total_dividend_income }

    /// Serialize all metrics to a Python dict for pandas DataFrame integration.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("total_return_pct", self.inner.total_return_pct)?;
        d.set_item("annualized_return_pct", self.inner.annualized_return_pct)?;
        d.set_item("sharpe_ratio", self.inner.sharpe_ratio)?;
        d.set_item("sortino_ratio", self.inner.sortino_ratio)?;
        d.set_item("calmar_ratio", self.inner.calmar_ratio)?;
        d.set_item("omega_ratio", self.inner.omega_ratio)?;
        d.set_item("serenity_ratio", self.inner.serenity_ratio)?;
        d.set_item("max_drawdown_pct", self.inner.max_drawdown_pct)?;
        d.set_item("max_drawdown_duration", self.inner.max_drawdown_duration)?;
        d.set_item("ulcer_index", self.inner.ulcer_index)?;
        d.set_item("recovery_factor", self.inner.recovery_factor)?;
        d.set_item("total_trades", self.inner.total_trades)?;
        d.set_item("win_rate", self.inner.win_rate)?;
        d.set_item("profit_factor", self.inner.profit_factor)?;
        d.set_item("avg_trade_return_pct", self.inner.avg_trade_return_pct)?;
        d.set_item("kelly_criterion", self.inner.kelly_criterion)?;
        d.set_item("sqn", self.inner.sqn)?;
        d.set_item("expectancy", self.inner.expectancy)?;
        d.set_item("tail_ratio", self.inner.tail_ratio)?;
        d.set_item("total_commission", self.inner.total_commission)?;
        d.set_item("time_in_market_pct", self.inner.time_in_market_pct)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "PerformanceMetrics(sharpe={:.4}, sortino={:.4}, max_dd={:.4}, return={:.2}%)",
            self.inner.sharpe_ratio,
            self.inner.sortino_ratio,
            self.inner.max_drawdown_pct,
            self.inner.total_return_pct,
        )
    }
}

// ---------------------------------------------------------------------------
// PortfolioResult
// ---------------------------------------------------------------------------

/// Full portfolio backtest result with per-symbol breakdown.
#[pyclass(name = "PortfolioResult")]
#[derive(Clone)]
pub struct PyPortfolioResult {
    pub(crate) inner: PortfolioResult,
}

#[pymethods]
impl PyPortfolioResult {
    /// Aggregate metrics across all symbols.
    #[getter]
    fn metrics(&self) -> PyPerformanceMetrics {
        PyPerformanceMetrics {
            inner: self.inner.portfolio_metrics.clone(),
        }
    }

    #[getter] fn initial_capital(&self) -> f64 { self.inner.initial_capital }
    #[getter] fn final_equity(&self) -> f64 { self.inner.final_equity }

    /// List of symbols that were backtested.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols.keys().cloned().collect()
    }

    /// Get per-symbol backtest result.
    fn symbol_result(&self, symbol: &str) -> PyResult<PyBacktestResult> {
        self.inner
            .symbols
            .get(symbol)
            .map(|r| PyBacktestResult { inner: r.clone() })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "no result for symbol '{}'", symbol
                ))
            })
    }

    /// Portfolio equity curve as list of (timestamp, equity, drawdown_pct) tuples.
    fn equity_curve(&self) -> Vec<(i64, f64, f64)> {
        self.inner
            .portfolio_equity_curve
            .iter()
            .map(|pt| (pt.timestamp, pt.equity, pt.drawdown_pct))
            .collect()
    }

    fn __repr__(&self) -> String {
        let m = &self.inner.portfolio_metrics;
        format!(
            "PortfolioResult(symbols={}, sharpe={:.4}, max_dd={:.4}, return={:.2}%)",
            self.inner.symbols.len(),
            m.sharpe_ratio,
            m.max_drawdown_pct,
            m.total_return_pct,
        )
    }
}

// ---------------------------------------------------------------------------
// BacktestResult (single-symbol)
// ---------------------------------------------------------------------------

/// Single-symbol backtest result.
#[pyclass(name = "BacktestResult")]
#[derive(Clone)]
pub struct PyBacktestResult {
    pub(crate) inner: BacktestResult,
}

#[pymethods]
impl PyBacktestResult {
    #[getter] fn symbol(&self) -> &str { &self.inner.symbol }
    #[getter] fn strategy_name(&self) -> &str { &self.inner.strategy_name }
    #[getter] fn initial_capital(&self) -> f64 { self.inner.initial_capital }
    #[getter] fn final_equity(&self) -> f64 { self.inner.final_equity }

    #[getter]
    fn metrics(&self) -> PyPerformanceMetrics {
        PyPerformanceMetrics {
            inner: self.inner.metrics.clone(),
        }
    }

    /// Equity curve as list of (timestamp, equity, drawdown_pct) tuples.
    fn equity_curve(&self) -> Vec<(i64, f64, f64)> {
        self.inner
            .equity_curve
            .iter()
            .map(|pt| (pt.timestamp, pt.equity, pt.drawdown_pct))
            .collect()
    }

    /// Number of trades executed.
    #[getter]
    fn num_trades(&self) -> usize {
        self.inner.trades.len()
    }

    fn __repr__(&self) -> String {
        let m = &self.inner.metrics;
        format!(
            "BacktestResult(symbol={}, sharpe={:.4}, return={:.2}%, trades={})",
            self.inner.symbol,
            m.sharpe_ratio,
            m.total_return_pct,
            self.inner.trades.len(),
        )
    }
}
