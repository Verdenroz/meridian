use std::collections::VecDeque;

/// One OHLCV bar (minute or daily granularity).
#[derive(Clone, Debug)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub timestamp: i64, // Unix timestamp seconds
}

/// A forming bar accumulated from tick updates within the current minute.
#[derive(Default, Clone)]
pub struct FormingBar {
    pub open: Option<f64>,
    pub high: f64,
    pub low: f64,
    pub last_price: f64,
    pub day_volume_start: Option<i64>, // day_volume at the start of this minute
    pub last_day_volume: i64,
    pub minute: i64, // minute-truncated unix timestamp this bar belongs to
}

impl FormingBar {
    pub fn tick(&self, price: f64, day_volume: Option<i64>, ts: i64) -> Self {
        let minute = ts - (ts % 60);
        if self.open.is_none() || minute != self.minute {
            // New minute: start fresh
            return Self {
                open: Some(price),
                high: price,
                low: price,
                last_price: price,
                day_volume_start: day_volume,
                last_day_volume: day_volume.unwrap_or(0),
                minute,
            };
        }
        Self {
            open: self.open,
            high: self.high.max(price),
            low: self.low.min(price),
            last_price: price,
            day_volume_start: self.day_volume_start,
            last_day_volume: day_volume.unwrap_or(self.last_day_volume),
            minute: self.minute,
        }
    }

    pub fn to_bar(&self) -> Option<Bar> {
        let open = self.open?;
        let volume = match self.day_volume_start {
            Some(start) => (self.last_day_volume - start).max(0),
            None => 0,
        };
        Some(Bar {
            open,
            high: self.high,
            low: self.low,
            close: self.last_price,
            volume,
            timestamp: self.minute,
        })
    }
}

/// Ring buffer of completed OHLCV bars for one asset.
///
/// Capacity defaults to 98,280 bars (252 trading days × 390 minutes). In practice
/// the ring is grown lazily, so memory usage is proportional to actual history.
#[derive(Clone)]
pub struct PriceBuffer {
    bars: VecDeque<Bar>,
    capacity: usize,
}

impl PriceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { bars: VecDeque::with_capacity(capacity.min(1024)), capacity }
    }

    pub fn push(&mut self, bar: Bar) {
        if self.bars.len() >= self.capacity {
            self.bars.pop_front();
        }
        self.bars.push_back(bar);
    }

    pub fn len(&self) -> usize {
        self.bars.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    #[allow(dead_code)]
    pub fn last(&self) -> Option<&Bar> {
        self.bars.back()
    }

    /// Extract the most recent `n` close prices (oldest first).
    pub fn closes(&self, n: usize) -> Vec<f64> {
        let start = self.bars.len().saturating_sub(n);
        self.bars.iter().skip(start).map(|b| b.close).collect()
    }

    /// Extract the most recent `n` high prices.
    pub fn highs(&self, n: usize) -> Vec<f64> {
        let start = self.bars.len().saturating_sub(n);
        self.bars.iter().skip(start).map(|b| b.high).collect()
    }

    /// Extract the most recent `n` low prices.
    pub fn lows(&self, n: usize) -> Vec<f64> {
        let start = self.bars.len().saturating_sub(n);
        self.bars.iter().skip(start).map(|b| b.low).collect()
    }

    /// Extract the most recent `n` open prices.
    pub fn opens(&self, n: usize) -> Vec<f64> {
        let start = self.bars.len().saturating_sub(n);
        self.bars.iter().skip(start).map(|b| b.open).collect()
    }

    /// Extract the most recent `n` volume values.
    pub fn volumes(&self, n: usize) -> Vec<f64> {
        let start = self.bars.len().saturating_sub(n);
        self.bars.iter().skip(start).map(|b| b.volume as f64).collect()
    }

    /// Build from a slice of `finance_query::models::Candle` (for historical mode).
    #[allow(dead_code)]
    pub fn from_candles<C>(candles: &[C]) -> Self
    where
        C: HasOhlcv,
    {
        let mut buf = Self::new(candles.len().max(512));
        for c in candles {
            buf.push(Bar {
                open: c.open(),
                high: c.high(),
                low: c.low(),
                close: c.close(),
                volume: c.volume(),
                timestamp: c.timestamp(),
            });
        }
        buf
    }

    /// Slice up to index `i` (inclusive), for bar-by-bar historical replay.
    pub fn prefix(candles: &[Bar], i: usize) -> Self {
        let end = (i + 1).min(candles.len());
        let capacity = end.max(512);
        let mut buf = Self::new(capacity);
        for c in &candles[..end] {
            buf.push(c.clone());
        }
        buf
    }
}

/// Trait implemented for any OHLCV source (used to abstract over finance-query's Candle).
#[allow(dead_code)]
pub trait HasOhlcv {
    fn open(&self) -> f64;
    fn high(&self) -> f64;
    fn low(&self) -> f64;
    fn close(&self) -> f64;
    fn volume(&self) -> i64;
    fn timestamp(&self) -> i64;
}

impl HasOhlcv for finance_query::Candle {
    fn open(&self) -> f64 { self.open }
    fn high(&self) -> f64 { self.high }
    fn low(&self) -> f64 { self.low }
    fn close(&self) -> f64 { self.close }
    fn volume(&self) -> i64 { self.volume }
    fn timestamp(&self) -> i64 { self.timestamp }
}
