//! Python-facing config wrappers for finance-query's backtesting engine.
//!
//! These types translate Python keyword arguments into the Rust builder API,
//! applying Meridian spec defaults (Section 12): 1M capital, 10bps commission,
//! 5bps slippage, 252 trading days × 390 minutes per day.

use pyo3::prelude::*;

use finance_query::backtesting::BacktestConfig;
use finance_query::backtesting::portfolio::{PortfolioConfig, RebalanceMode};

// -- Spec defaults (Section 12) --
const DEFAULT_CAPITAL: f64 = 1_000_000.0;
const DEFAULT_COMMISSION_PCT: f64 = 0.0010; // 10 bps
const DEFAULT_SLIPPAGE_PCT: f64 = 0.0005;   // 5 bps
const DEFAULT_BARS_PER_YEAR: f64 = 252.0 * 390.0; // minute bars
const DEFAULT_MAX_POSITIONS: usize = 100;

/// Per-trade backtest configuration with Meridian spec defaults.
#[pyclass(name = "BacktestConfig")]
#[derive(Clone)]
pub struct PyBacktestConfig {
    pub(crate) inner: BacktestConfig,
}

#[pymethods]
impl PyBacktestConfig {
    #[new]
    #[pyo3(signature = (
        initial_capital = DEFAULT_CAPITAL,
        commission_pct = DEFAULT_COMMISSION_PCT,
        slippage_pct = DEFAULT_SLIPPAGE_PCT,
        bars_per_year = DEFAULT_BARS_PER_YEAR,
        risk_free_rate = 0.0,
        allow_short = false,
    ))]
    fn new(
        initial_capital: f64,
        commission_pct: f64,
        slippage_pct: f64,
        bars_per_year: f64,
        risk_free_rate: f64,
        allow_short: bool,
    ) -> PyResult<Self> {
        let inner = BacktestConfig::builder()
            .initial_capital(initial_capital)
            .commission_pct(commission_pct)
            .slippage_pct(slippage_pct)
            .bars_per_year(bars_per_year)
            .risk_free_rate(risk_free_rate)
            .allow_short(allow_short)
            .position_size_pct(1.0)
            .close_at_end(true)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestConfig(capital={}, commission={:.4}, slippage={:.4})",
            self.inner.initial_capital,
            self.inner.commission_pct,
            self.inner.slippage_pct,
        )
    }
}

/// Portfolio-level configuration wrapping a BacktestConfig.
#[pyclass(name = "PortfolioConfig")]
#[derive(Clone)]
pub struct PyPortfolioConfig {
    pub(crate) inner: PortfolioConfig,
}

#[pymethods]
impl PyPortfolioConfig {
    #[new]
    #[pyo3(signature = (
        base = None,
        max_positions = DEFAULT_MAX_POSITIONS,
        max_allocation_per_symbol = None,
    ))]
    fn new(
        base: Option<PyBacktestConfig>,
        max_positions: usize,
        max_allocation_per_symbol: Option<f64>,
    ) -> PyResult<Self> {
        let base_config = match base {
            Some(b) => b.inner,
            None => PyBacktestConfig::new(
                DEFAULT_CAPITAL,
                DEFAULT_COMMISSION_PCT,
                DEFAULT_SLIPPAGE_PCT,
                DEFAULT_BARS_PER_YEAR,
                0.0,
                false,
            )?.inner,
        };

        let mut config = PortfolioConfig::new(base_config)
            .max_total_positions(max_positions)
            .rebalance(RebalanceMode::AvailableCapital);

        if let Some(pct) = max_allocation_per_symbol {
            config = config.max_allocation_per_symbol(pct);
        }

        Ok(Self { inner: config })
    }

    fn __repr__(&self) -> String {
        format!(
            "PortfolioConfig(max_positions={:?})",
            self.inner.max_total_positions,
        )
    }
}
