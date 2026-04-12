//! Fetch real OHLCV candle data from Yahoo Finance via finance-query.
//!
//! Exposes `fetch_candles(symbols, interval, range)` to Python so the
//! backtesting module can use actual market prices instead of reconstructed
//! synthetic prices from feature shards.

use std::collections::HashMap;

use numpy::{PyArray2, IntoPyArray, ndarray::Array2};
use pyo3::prelude::*;

use finance_query::{Interval, TimeRange, Ticker};

fn parse_interval(s: &str) -> PyResult<Interval> {
    match s {
        "1m" => Ok(Interval::OneMinute),
        "5m" => Ok(Interval::FiveMinutes),
        "15m" => Ok(Interval::FifteenMinutes),
        "30m" => Ok(Interval::ThirtyMinutes),
        "1h" => Ok(Interval::OneHour),
        "1d" => Ok(Interval::OneDay),
        "1wk" => Ok(Interval::OneWeek),
        "1mo" => Ok(Interval::OneMonth),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("unknown interval '{}'. Valid: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo", s),
        )),
    }
}

fn parse_range(s: &str) -> PyResult<TimeRange> {
    match s {
        "1d" => Ok(TimeRange::OneDay),
        "5d" => Ok(TimeRange::FiveDays),
        "1mo" => Ok(TimeRange::OneMonth),
        "3mo" => Ok(TimeRange::ThreeMonths),
        "6mo" => Ok(TimeRange::SixMonths),
        "1y" => Ok(TimeRange::OneYear),
        "2y" => Ok(TimeRange::TwoYears),
        "5y" => Ok(TimeRange::FiveYears),
        "10y" => Ok(TimeRange::TenYears),
        "ytd" => Ok(TimeRange::YearToDate),
        "max" => Ok(TimeRange::Max),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("unknown range '{}'. Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max", s),
        )),
    }
}

/// Fetch OHLCV candles for multiple symbols from Yahoo Finance.
///
/// Returns a dict mapping symbol → (T, 6) numpy float64 array of
/// [timestamp, open, high, low, close, volume].
///
/// Symbols that fail to fetch are silently skipped (logged to stderr).
#[pyfunction]
#[pyo3(signature = (symbols, interval = "1d", range = "10y"))]
pub fn fetch_candles(
    py: Python<'_>,
    symbols: Vec<String>,
    interval: &str,
    range: &str,
) -> PyResult<HashMap<String, Py<PyArray2<f64>>>> {
    let iv = parse_interval(interval)?;
    let tr = parse_range(range)?;

    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
    })?;

    let results: Vec<(String, Vec<[f64; 6]>)> = py.allow_threads(|| {
        rt.block_on(async {
            let mut out = Vec::new();
            for sym in &symbols {
                match fetch_one(sym, iv, tr).await {
                    Ok(rows) => out.push((sym.clone(), rows)),
                    Err(e) => eprintln!("fetch_candles: {sym}: {e}"),
                }
            }
            out
        })
    });

    let mut candles = HashMap::new();
    for (sym, rows) in results {
        let n = rows.len();
        let flat: Vec<f64> = rows.into_iter().flat_map(|r| r).collect();
        let arr = Array2::from_shape_vec((n, 6), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        candles.insert(sym, arr.into_pyarray(py).into());
    }

    Ok(candles)
}

async fn fetch_one(symbol: &str, interval: Interval, range: TimeRange) -> anyhow::Result<Vec<[f64; 6]>> {
    let ticker = Ticker::new(symbol).await?;
    let chart = ticker.chart(interval, range).await?;
    let rows: Vec<[f64; 6]> = chart.candles.iter().map(|c| {
        [
            c.timestamp as f64,
            c.open,
            c.high,
            c.low,
            c.close,
            c.volume as f64,
        ]
    }).collect();
    Ok(rows)
}
