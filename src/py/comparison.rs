//! Strategy comparison wrappers.
//!
//! Thin layer over finance-query's BacktestComparison that accepts
//! labelled PyPortfolioResult or PyBacktestResult entries, ranks them
//! by a chosen metric, and returns a Python-friendly report.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use finance_query::backtesting::{
    BacktestComparison, OptimizeMetric,
};

use super::results::PyBacktestResult;

/// Parse a metric name string into the Rust enum.
fn parse_metric(metric: &str) -> PyResult<OptimizeMetric> {
    match metric.to_lowercase().as_str() {
        "sharpe" | "sharpe_ratio" => Ok(OptimizeMetric::SharpeRatio),
        "sortino" | "sortino_ratio" => Ok(OptimizeMetric::SortinoRatio),
        "calmar" | "calmar_ratio" => Ok(OptimizeMetric::CalmarRatio),
        "return" | "total_return" => Ok(OptimizeMetric::TotalReturn),
        "profit_factor" => Ok(OptimizeMetric::ProfitFactor),
        "win_rate" => Ok(OptimizeMetric::WinRate),
        "drawdown" | "min_drawdown" => Ok(OptimizeMetric::MinDrawdown),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown metric '{}'. Valid: sharpe, sortino, calmar, return, profit_factor, win_rate, drawdown",
            metric
        ))),
    }
}

/// Comparison report: strategies ranked by a chosen metric.
#[pyclass(name = "ComparisonReport")]
pub struct PyComparisonReport {
    winner_label: String,
    rows: Vec<ComparisonRowData>,
    metric: String,
}

/// Flattened row data for Python access.
#[derive(Clone)]
struct ComparisonRowData {
    label: String,
    rank: usize,
    sharpe: f64,
    sortino: f64,
    total_return: f64,
    max_dd: f64,
    total_trades: usize,
    rank_score: f64,
}

#[pymethods]
impl PyComparisonReport {
    /// Label of the winning strategy.
    #[getter]
    fn winner(&self) -> &str {
        &self.winner_label
    }

    /// Metric used for ranking.
    #[getter]
    fn metric(&self) -> &str {
        &self.metric
    }

    /// Number of strategies compared.
    #[getter]
    fn count(&self) -> usize {
        self.rows.len()
    }

    /// Ranked table as list of dicts, best-first.
    fn table<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.rows
            .iter()
            .map(|r| {
                let d = PyDict::new(py);
                d.set_item("label", &r.label)?;
                d.set_item("rank", r.rank)?;
                d.set_item("sharpe_ratio", r.sharpe)?;
                d.set_item("sortino_ratio", r.sortino)?;
                d.set_item("total_return_pct", r.total_return)?;
                d.set_item("max_drawdown_pct", r.max_dd)?;
                d.set_item("total_trades", r.total_trades)?;
                d.set_item("rank_score", r.rank_score)?;
                Ok(d)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ComparisonReport(metric={}, winner={}, count={})",
            self.metric, self.winner_label, self.rows.len()
        )
    }
}

/// Compare multiple single-symbol backtest results by a chosen metric.
///
/// Args:
///     results: list of (label, BacktestResult) tuples.
///     metric: one of "sharpe", "sortino", "calmar", "return",
///             "profit_factor", "win_rate", "drawdown".
#[pyfunction]
#[pyo3(signature = (results, metric = "sharpe"))]
pub fn run_comparison(
    results: Vec<(String, PyBacktestResult)>,
    metric: &str,
) -> PyResult<PyComparisonReport> {
    let opt_metric = parse_metric(metric)?;

    let mut cmp = BacktestComparison::new();
    for (label, r) in results {
        cmp = cmp.add(label, r.inner);
    }
    let report = cmp.ranked_by(opt_metric);

    let winner_label = report.winner().to_string();
    let rows: Vec<ComparisonRowData> = report
        .table()
        .iter()
        .map(|r| ComparisonRowData {
            label: r.label.clone(),
            rank: r.rank,
            sharpe: r.sharpe_ratio,
            sortino: r.sortino_ratio,
            total_return: r.total_return_pct,
            max_dd: r.max_drawdown_pct,
            total_trades: r.total_trades,
            rank_score: r.rank_score,
        })
        .collect();

    Ok(PyComparisonReport {
        winner_label,
        rows,
        metric: metric.to_string(),
    })
}
