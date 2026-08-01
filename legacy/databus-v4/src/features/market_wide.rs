use crate::store::DataStore;
use chrono::{DateTime, Timelike, Utc};
use std::sync::Arc;

/// Assemble the 96-dimensional market-wide feature vector.
///
/// Sub-block layout (from spec Section 6.3):
///   [0..12]   Volatility & Sentiment     (12 dims)
///   [12..24]  Yield Curve                (12 dims)
///   [24..36]  FRED Macro                 (12 dims)
///   [36..52]  Sector & Industry          (16 dims)
///   [52..72]  World Indices & Cross-Asset (20 dims)
///   [72..84]  Market Breadth & Flow      (12 dims)
///   [84..96]  RSS & News Aggregate       (12 dims)
pub fn assemble_market_vector(store: &Arc<DataStore>, now: DateTime<Utc>) -> [f32; 96] {
    let mut out = [0.0f32; 96];

    // ── Volatility & Sentiment [0..12] ────────────────────────────────────────
    // Layout mirrors the historical batch path so training and inference use
    // the same feature semantics:
    //   [0]    VIX level (normalised)
    //   [1..5] VIX regime one-hot (extreme-fear / fear / neutral / greed / extreme-greed)
    //   [6]    VIX level (alt normalisation, wider range)
    //   [7]    VIX 1d log-return
    //   [8]    VIX term-structure proxy (stub; no live VIX3M data)
    //   [9]    Fear & Greed index (macro-sentiment proxy, replaces put/call ratio)
    //   [10]   DXY return (from world_indices poller)
    //   [11]   Gold–oil spread proxy
    {
        let wi = store.world_indices.read();
        let vix = wi.value.vix_level as f32;
        let vix_ret = wi.value.vix_ret as f32;
        let dxy = wi.value.dxy as f32;
        let gold = wi.value.gold as f32;
        let oil = wi.value.oil as f32;
        drop(wi);

        out[0] = (vix / 100.0).clamp(0.0, 2.0);
        let regime_idx = if vix >= 35.0 {
            0usize // extreme fear
        } else if vix >= 28.0 {
            1 // fear
        } else if vix >= 20.0 {
            2 // neutral
        } else if vix >= 14.0 {
            3 // greed
        } else {
            4 // extreme greed
        };
        out[1 + regime_idx] = 1.0;
        out[6] = (vix / 20.0).clamp(0.0, 5.0);
        out[7] = vix_ret;
        out[8] = (vix / (18.0 + 250.0 * vix_ret.abs() + 1.0)).clamp(0.0, 2.0);
        out[9] = store.fear_greed.read().value.value as f32 / 100.0;
        out[10] = dxy;
        out[11] = ((gold - oil) * 0.5).clamp(-2.0, 2.0);
    }

    // ── Yield Curve [12..24] ──────────────────────────────────────────────────
    {
        let yc = store.yield_curve.read();
        let y = &yc.value;
        out[12] = y.y1m as f32;
        out[13] = y.y3m as f32;
        out[14] = y.y6m as f32;
        out[15] = y.y1 as f32;
        out[16] = y.y2 as f32;
        out[17] = y.y3 as f32;
        out[18] = y.y5 as f32;
        out[19] = y.y7 as f32;
        out[20] = y.y10 as f32;
        out[21] = y.y20 as f32;
        out[22] = y.y30 as f32;
        out[23] = y.spread_2y10y as f32;
    }

    // ── FRED Macro [24..36] ───────────────────────────────────────────────────
    {
        let macro_d = store.fred_macro.read();
        for (i, &v) in macro_d.value.values.iter().enumerate().take(12) {
            out[24 + i] = v as f32;
        }
    }

    // ── Sector & Industry [36..52] ────────────────────────────────────────────
    {
        let sec = store.sector.read();
        let s = &sec.value;
        for (i, &r) in s.returns.iter().enumerate() {
            out[36 + i] = r as f32; // 11 sector returns
        }
        out[47] = s.dispersion as f32;
        out[48] = s.best_sector as f32 / 10.0;
        out[49] = s.worst_sector as f32 / 10.0;
        out[50] = s.rotation_velocity as f32;
        out[51] = s.breadth_pct_above_sma200 as f32;
    }

    // ── World Indices & Cross-Asset [52..72] ──────────────────────────────────
    {
        let wi = store.world_indices.read();
        let w = &wi.value;
        // SPX, IXIC, RUT, DJI, FTSE, DAX, N225, HSI, SSEC returns
        let idx_order = [
            "^GSPC",
            "^IXIC",
            "^RUT",
            "^DJI",
            "^FTSE",
            "^GDAXI",
            "^N225",
            "^HSI",
            "000001.SS",
        ];
        for (i, sym) in idx_order.iter().enumerate() {
            out[52 + i] = w.returns.get(*sym).copied().unwrap_or(0.0) as f32;
        }
        out[61] = w.dxy as f32; // DXY proxy (UUP return)
        out[62] = w.eurusd as f32;
        out[63] = w.usdjpy as f32;
        out[64] = w.gold as f32;
        out[65] = w.oil as f32;

        let crypto = store.crypto.read();
        let c = &crypto.value;
        out[66] = (c.btc_price / 100_000.0).clamp(0.0, 2.0) as f32; // price normalized
        out[67] = c.btc_return_1d as f32;
        out[68] = (c.eth_price / 10_000.0).clamp(0.0, 2.0) as f32;
        out[69] = c.eth_return_1d as f32;
        out[70] = (c.total_mktcap.ln() / 30.0).clamp(0.0, 1.0) as f32;
        out[71] = c.btc_dominance as f32;
    }

    // ── Market Breadth & Flow [72..84] ────────────────────────────────────────
    {
        let ms = store.market_summary.read();
        let m = &ms.value;
        out[72] = m.advance_decline as f32;
        out[73] = m.new_highs_lows as f32;
        out[74] = m.spy_vol_5d as f32;
        out[75] = m.qqq_vol_5d as f32;
        out[76] = m.iwm_vol_5d as f32;
        out[77] = m.cross_asset_corr as f32;
        out[78] = m.trending_count as f32 / 100.0;
        out[79] = m.trending_portfolio_overlap as f32;
        out[80] = m.gainers_momentum as f32;
        out[81] = m.actives_vol_ratio as f32;
        out[82] = market_hours_encode(now);
        out[83] = minutes_since_market_event(now);
    }

    // ── RSS & News Aggregate [84..96] ─────────────────────────────────────────
    {
        let rss = store.rss_feed.read();
        let r = &rss.value;
        out[84] = r.sentiment_1h as f32;
        out[85] = r.sentiment_24h as f32;
        out[86] = (r.volume_1h.ln() as f32).max(0.0);
        out[87] = (r.volume_24h.ln() as f32).max(0.0);
        out[88] = r.fed_count_24h as f32;
        out[89] = r.sec_count_24h as f32;
        out[90] = r.crypto_sentiment as f32;
        out[91] = r.macro_sentiment as f32;
        out[92] = r.tech_sentiment as f32;
        out[93] = r.business_sentiment as f32;
        out[94] = r.news_surprise as f32;
        out[95] = r.topic_diversity as f32;
    }

    out
}

/// Encode market hours state: 0=closed, 0.33=pre, 0.67=open, 1.0=post
fn market_hours_encode(now: DateTime<Utc>) -> f32 {
    // NYSE hours: 9:30–16:00 ET = 14:30–21:00 UTC (EST) / 13:30–20:00 UTC (EDT)
    // Approximate using UTC hour only
    let hour = now.hour();
    let minute = now.minute();
    let mins = hour * 60 + minute;
    if mins >= 830 && mins < 930 {
        0.33 // pre-market ~4:00–9:30 ET (simplified)
    } else if mins >= 930 && mins < 1320 {
        0.67 // regular hours 9:30–16:00 ET → UTC 14:30–21:00
    } else if mins >= 1320 && mins < 1440 {
        1.0 // after-hours
    } else {
        0.0 // closed
    }
}

/// Minutes since the most recent market open/close transition.
fn minutes_since_market_event(now: DateTime<Utc>) -> f32 {
    let hour = now.hour();
    let minute = now.minute();
    let mins = hour * 60 + minute;
    // NYSE open at 14:30 UTC (930 ET), close at 21:00 UTC (1620 ET — simplified)
    let open_utc = 14 * 60 + 30;
    let close_utc = 21 * 60;
    let since_open = (mins as i32 - open_utc as i32).abs() as f32;
    let since_close = (mins as i32 - close_utc as i32).abs() as f32;
    since_open.min(since_close) / 60.0 // normalized to hours
}
