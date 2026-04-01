pub mod assembler;
pub mod cross_asset;
pub mod event_block;
pub mod fundamental_block;
pub mod market_wide;
pub mod options_block;
pub mod price_volume;
pub mod risk_block;
pub mod technical;

use chrono::{DateTime, Utc};
use serde::ser::SerializeMap;
use serde::Serialize;

/// One JSONL row: a complete feature vector for one asset at one minute.
///
/// Dimensions match the Python pipeline constants in `python/train/feature_schema.py`:
///   PER_ASSET_FEATURE_DIM = 184  (120 continuous + 64 text)
///   MARKET_FEATURE_DIM    = 96
pub struct FeatureRow {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub asset_class: String,
    /// Block 1: Price & Volume (10 dims)
    pub price_volume: [f32; 10],
    /// Block 2: Technical Indicators (42 dims)
    pub technical: [f32; 42],
    /// Block 3: Options-Derived (12 dims)
    pub options: [f32; 12],
    /// Block 4: Fundamental & Ownership (30 dims)
    pub fundamental: [f32; 30],
    /// Block 5: Risk Analytics (8 dims)
    pub risk_analytics: [f32; 8],
    /// Block 6: Event State (10 decayed signals)
    pub event_state: [f32; 10],
    /// Block 7: Cross-Asset Context (8 dims)
    pub cross_asset_context: [f32; 8],
    /// Block 8: Text Embedding (64 dims) — zeros in Phase 0; wired in Phase 1
    pub text: [f32; 64],
    /// Market-Wide (96 dims)
    pub market: [f32; 96],
}

impl Default for FeatureRow {
    fn default() -> Self {
        FeatureRow {
            timestamp: DateTime::default(),
            symbol: String::new(),
            asset_class: String::new(),
            price_volume: [0.0; 10],
            technical: [0.0; 42],
            options: [0.0; 12],
            fundamental: [0.0; 30],
            risk_analytics: [0.0; 8],
            event_state: [0.0; 10],
            cross_asset_context: [0.0; 8],
            text: [0.0; 64],
            market: [0.0; 96],
        }
    }
}

impl Serialize for FeatureRow {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(13))?;
        map.serialize_entry("timestamp", &self.timestamp)?;
        map.serialize_entry("symbol", &self.symbol)?;
        map.serialize_entry("asset_class", &self.asset_class)?;
        map.serialize_entry("price_volume", self.price_volume.as_ref())?;
        map.serialize_entry("technical", self.technical.as_ref())?;
        map.serialize_entry("options", self.options.as_ref())?;
        map.serialize_entry("fundamental", self.fundamental.as_ref())?;
        map.serialize_entry("risk_analytics", self.risk_analytics.as_ref())?;
        map.serialize_entry("event_state", self.event_state.as_ref())?;
        map.serialize_entry("cross_asset_context", self.cross_asset_context.as_ref())?;
        map.serialize_entry("text", self.text.as_ref())?;
        map.serialize_entry("market", self.market.as_ref())?;
        map.end()
    }
}

/// Classify an asset symbol into "equity", "etf", or "crypto".
pub fn classify_asset(symbol: &str) -> String {
    // Simple heuristic: known crypto tickers, otherwise equity
    match symbol {
        "BTC-USD" | "ETH-USD" | "DOGE-USD" | "SOL-USD" => "crypto",
        s if s.ends_with("-USD") => "crypto",
        _ => "equity",
    }
    .to_string()
}
