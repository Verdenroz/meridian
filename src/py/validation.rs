//! Validation wrappers: Monte Carlo simulation for statistical significance.
//!
//! Monte Carlo resampling reshuffles the trade sequence to estimate the
//! probability distribution of key metrics. This tells us whether the
//! observed performance is skill or luck.
//!
//! Walk-forward validation is handled in Python (the model must re-compute
//! weights for each window), but each window's backtest still runs through
//! the Rust engine via run_backtest.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use finance_query::backtesting::{MonteCarloConfig, MonteCarloMethod, PercentileStats};

use super::results::PyBacktestResult;

// ---------------------------------------------------------------------------
// PyPercentileStats
// ---------------------------------------------------------------------------

#[pyclass(name = "PercentileStats")]
#[derive(Clone)]
pub struct PyPercentileStats {
    inner: PercentileStats,
}

#[pymethods]
impl PyPercentileStats {
    #[getter]
    fn p5(&self) -> f64 {
        self.inner.p5
    }
    #[getter]
    fn p25(&self) -> f64 {
        self.inner.p25
    }
    #[getter]
    fn p50(&self) -> f64 {
        self.inner.p50
    }
    #[getter]
    fn p75(&self) -> f64 {
        self.inner.p75
    }
    #[getter]
    fn p95(&self) -> f64 {
        self.inner.p95
    }
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("p5", self.inner.p5)?;
        d.set_item("p25", self.inner.p25)?;
        d.set_item("p50", self.inner.p50)?;
        d.set_item("p75", self.inner.p75)?;
        d.set_item("p95", self.inner.p95)?;
        d.set_item("mean", self.inner.mean)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "PercentileStats(p5={:.3}, p25={:.3}, p50={:.3}, p75={:.3}, p95={:.3})",
            self.inner.p5, self.inner.p25, self.inner.p50, self.inner.p75, self.inner.p95,
        )
    }
}

// ---------------------------------------------------------------------------
// PyMonteCarloResult
// ---------------------------------------------------------------------------

#[pyclass(name = "MonteCarloResult")]
#[derive(Clone)]
pub struct PyMonteCarloResult {
    num_simulations: usize,
    total_return: PercentileStats,
    max_drawdown: PercentileStats,
    sharpe_ratio: PercentileStats,
    profit_factor: PercentileStats,
}

#[pymethods]
impl PyMonteCarloResult {
    #[getter]
    fn num_simulations(&self) -> usize {
        self.num_simulations
    }

    #[getter]
    fn total_return(&self) -> PyPercentileStats {
        PyPercentileStats {
            inner: self.total_return.clone(),
        }
    }

    #[getter]
    fn max_drawdown(&self) -> PyPercentileStats {
        PyPercentileStats {
            inner: self.max_drawdown.clone(),
        }
    }

    #[getter]
    fn sharpe_ratio(&self) -> PyPercentileStats {
        PyPercentileStats {
            inner: self.sharpe_ratio.clone(),
        }
    }

    #[getter]
    fn profit_factor(&self) -> PyPercentileStats {
        PyPercentileStats {
            inner: self.profit_factor.clone(),
        }
    }

    /// Check deployment criteria from Section 12 of the spec:
    /// P5 return > 0, P25 Sharpe > 0.5, P95 max drawdown < 0.30
    fn passes_deployment_criteria(&self) -> bool {
        self.total_return.p5 > 0.0 && self.sharpe_ratio.p25 > 0.5 && self.max_drawdown.p95 < 0.30
    }

    fn __repr__(&self) -> String {
        format!(
            "MonteCarloResult(n={}, return_p50={:.2}%, dd_p95={:.4}, sharpe_p25={:.3})",
            self.num_simulations,
            self.total_return.p50,
            self.max_drawdown.p95,
            self.sharpe_ratio.p25,
        )
    }
}

// ---------------------------------------------------------------------------
// Python entry point
// ---------------------------------------------------------------------------

/// Run Monte Carlo simulation on a completed single-symbol backtest.
///
/// Reshuffles the trade sequence to estimate the probability distribution
/// of returns, drawdowns, and risk-adjusted metrics.
///
/// Args:
///     result: A BacktestResult from a single-symbol backtest.
///     num_simulations: Number of reshuffled equity curves (default 1000).
///     method: "shuffle" (default) or "block_bootstrap".
///     seed: Optional PRNG seed for reproducibility.
#[pyfunction]
#[pyo3(signature = (result, num_simulations = 1000, method = "shuffle", seed = None))]
pub fn run_monte_carlo(
    py: Python<'_>,
    result: PyBacktestResult,
    num_simulations: usize,
    method: &str,
    seed: Option<u64>,
) -> PyResult<PyMonteCarloResult> {
    let mc_method = match method {
        "shuffle" | "iid" => MonteCarloMethod::IidShuffle,
        "block_bootstrap" => MonteCarloMethod::BlockBootstrap {
            block_size: (num_simulations as f64).sqrt() as usize,
        },
        "stationary" => MonteCarloMethod::StationaryBootstrap {
            mean_block_size: (num_simulations as f64).sqrt() as usize,
        },
        "parametric" => MonteCarloMethod::Parametric,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown method '{}'. Valid: shuffle, block_bootstrap, stationary, parametric",
                method
            )));
        }
    };

    let mut config = MonteCarloConfig::new()
        .num_simulations(num_simulations)
        .method(mc_method);

    if let Some(s) = seed {
        config = config.seed(s);
    }

    let inner_result = result.inner;

    // Pure Rust computation — release the GIL.
    let mc_result = py.allow_threads(move || config.run(&inner_result));

    Ok(PyMonteCarloResult {
        num_simulations: mc_result.num_simulations,
        total_return: mc_result.total_return,
        max_drawdown: mc_result.max_drawdown,
        sharpe_ratio: mc_result.sharpe_ratio,
        profit_factor: mc_result.profit_factor,
    })
}
