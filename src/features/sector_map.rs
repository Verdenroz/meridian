/// GICS sector → sector ETF symbol mapping for the canonical 100-symbol universe.
///
/// Returns `None` for ETFs, crypto tickers, or symbols not in the mapping
/// (sector-relative-strength features fall back to 0.0 for those).
pub fn sector_etf_for(symbol: &str) -> Option<&'static str> {
    Some(match symbol {
        // Tech / IT
        "AAPL" | "MSFT" | "NVDA" | "AVGO" | "ADBE" | "AMD" | "CRM" | "CSCO"
        | "INTC" | "ORCL" | "TXN" | "QCOM" | "INTU" | "AMAT" | "ADI"
        | "IBM" | "MU" | "NOW" | "LRCX" | "KLAC" => "XLK",
        // Financials
        "JPM" | "BAC" | "WFC" | "GS" | "MS" | "C" | "BLK" | "AXP" | "SCHW"
        | "USB" => "XLF",
        // Health Care
        "UNH" | "JNJ" | "LLY" | "PFE" | "ABBV" | "MRK" | "TMO" | "ABT"
        | "DHR" | "BMY" | "AMGN" | "ISRG" | "MDT" => "XLV",
        // Industrials
        "CAT" | "BA" | "GE" | "HON" | "UPS" | "RTX" | "LMT" | "DE" | "MMM"
        | "UNP" => "XLI",
        // Consumer Discretionary
        "AMZN" | "TSLA" | "HD" | "MCD" | "NKE" | "LOW" | "SBUX" | "TJX"
        | "BKNG" => "XLY",
        // Communication Services
        "GOOGL" | "META" | "DIS" | "NFLX" | "T" | "VZ" | "CMCSA" => "XLC",
        // Consumer Staples
        "PG" | "KO" | "PEP" | "WMT" | "COST" | "PM" | "MO" => "XLP",
        // Energy
        "XOM" | "CVX" | "COP" | "SLB" => "XLE",
        // Materials
        "LIN" | "FCX" | "APD" | "NEM" => "XLB",
        // Real Estate
        "AMT" | "CCI" | "PLD" | "EQIX" => "XLRE",
        // Utilities
        "NEE" | "DUK" | "SO" | "AEP" => "XLU",
        _ => return None,
    })
}

/// Maps a symbol to its index (0–10) into `SectorData.returns[]`.
///
/// Index order matches `poll_sector_industry` in `src/poller/market.rs`:
///   0 = Technology (XLK)
///   1 = Financial Services (XLF)
///   2 = Consumer Cyclical / Discretionary (XLY)
///   3 = Communication Services (XLC)
///   4 = Healthcare (XLV)
///   5 = Industrials (XLI)
///   6 = Consumer Defensive / Staples (XLP)
///   7 = Energy (XLE)
///   8 = Basic Materials (XLB)
///   9 = Real Estate (XLRE)
///  10 = Utilities (XLU)
pub fn sector_returns_idx_for(symbol: &str) -> Option<usize> {
    let etf = sector_etf_for(symbol)?;
    Some(match etf {
        "XLK"  => 0,
        "XLF"  => 1,
        "XLY"  => 2,
        "XLC"  => 3,
        "XLV"  => 4,
        "XLI"  => 5,
        "XLP"  => 6,
        "XLE"  => 7,
        "XLB"  => 8,
        "XLRE" => 9,
        "XLU"  => 10,
        _      => return None,
    })
}
