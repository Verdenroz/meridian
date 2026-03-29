use crate::store::{PriceBuffer, RollingStatsStore};
use finance_query::indicators;
use std::sync::Arc;

/// Block 2: Technical Indicators — 42 features.
///
/// Each indicator is normalized to a consistent range. Bounded oscillators
/// ([0,100] or [-100,0]) use fixed affine transforms. Unbounded indicators
/// (MACD, ROC, OBV slope, etc.) use rolling z-scores via `RollingStatsStore`.
pub fn assemble(buf: Option<&PriceBuffer>, stats: &Arc<RollingStatsStore>, symbol: &str) -> [f32; 42] {
    let Some(buf) = buf else { return [0.0f32; 42] };

    // Need at least 200 bars for SMA(200); shorter histories produce partial results.
    let min_bars = 5;
    if buf.len() < min_bars {
        return [0.0f32; 42];
    }

    let closes = buf.closes(300);
    let highs = buf.highs(300);
    let lows = buf.lows(300);
    let opens = buf.opens(300);
    let volumes = buf.volumes(300);

    let n = closes.len();
    let last = closes[n - 1];

    // ── Helper closures ──────────────────────────────────────────────────────

    // Last value of a Vec<Option<f64>>, defaulting to fallback
    let last_val = |v: Vec<Option<f64>>, fallback: f64| -> f64 {
        v.last().and_then(|x| *x).unwrap_or(fallback)
    };

    // Z-score using rolling stats, also updates the running stats
    let zs = |feat: &str, x: f64| -> f32 { stats.zscore(symbol, feat, x) };

    // Scale [lo, hi] → [-1, 1]
    let scale = |x: f64, lo: f64, hi: f64| -> f32 {
        ((x - lo) / (hi - lo) * 2.0 - 1.0).clamp(-1.0, 1.0) as f32
    };

    // Clamp and normalize
    let clamp01 = |x: f64| x.clamp(0.0, 1.0) as f32;

    // ── ATR (needed for MA normalization) ───────────────────────────────────
    let atr14 = indicators::atr(&highs, &lows, &closes, 14)
        .ok()
        .and_then(|v| v.last().and_then(|x| *x))
        .unwrap_or(last * 0.01)
        .max(1e-8);

    let ma_norm = |ma: f64| -> f32 {
        ((last - ma) / atr14).clamp(-5.0, 5.0) as f32
    };

    // ── Moving Averages (8 features) ─────────────────────────────────────────
    let sma20  = last_val(indicators::sma(&closes, 20),  last);
    let sma50  = last_val(indicators::sma(&closes, 50),  last);
    let sma200 = last_val(indicators::sma(&closes, 200), last);
    let ema12  = last_val(indicators::ema(&closes, 12),  last);
    let dema20 = last_val(indicators::dema(&closes, 20).unwrap_or_default(), last);
    let hma20  = last_val(indicators::hma(&closes, 20).unwrap_or_default(),  last);
    let alma20 = last_val(
        indicators::alma(&closes, 20, 0.85, 6.0).unwrap_or_default(),
        last,
    );
    let mcg20 = last_val(
        indicators::mcginley_dynamic(&closes, 20).unwrap_or_default(),
        last,
    );

    // ── Momentum (11 features) ───────────────────────────────────────────────
    let rsi14 = last_val(
        indicators::rsi(&closes, 14).unwrap_or_default(),
        50.0,
    );

    let macd = indicators::macd(&closes, 12, 26, 9).ok();
    let macd_hist = macd
        .as_ref()
        .and_then(|m| m.histogram.last().and_then(|x| *x))
        .unwrap_or(0.0);

    let stoch = indicators::stochastic(&highs, &lows, &closes, 14, 3, 3).ok();
    let stoch_k = stoch.as_ref().and_then(|s| s.k.last().and_then(|x| *x)).unwrap_or(50.0);
    let stoch_d = stoch.as_ref().and_then(|s| s.d.last().and_then(|x| *x)).unwrap_or(50.0);

    let srsi = indicators::stochastic_rsi(&closes, 14, 14, 3, 3).ok();
    let srsi_k = srsi.as_ref().and_then(|s| s.k.last().and_then(|x| *x)).unwrap_or(0.5);

    let cci20 = last_val(
        indicators::cci(&highs, &lows, &closes, 20).unwrap_or_default(),
        0.0,
    );
    let willr14 = last_val(
        indicators::williams_r(&highs, &lows, &closes, 14).unwrap_or_default(),
        -50.0,
    );
    let cmo14 = last_val(indicators::cmo(&closes, 14).unwrap_or_default(), 0.0);
    let roc10 = last_val(indicators::roc(&closes, 10).unwrap_or_default(), 0.0);
    let ao = last_val(
        indicators::awesome_oscillator(&highs, &lows, 5, 34).unwrap_or_default(),
        0.0,
    );
    let coppock = last_val(
        indicators::coppock_curve(&closes, 14, 11, 10).unwrap_or_default(),
        0.0,
    );

    // ── Volatility (5 features) ──────────────────────────────────────────────
    let bb = indicators::bollinger_bands(&closes, 20, 2.0).ok();
    let bb_upper = bb.as_ref().and_then(|b| b.upper.last().and_then(|x| *x)).unwrap_or(last * 1.02);
    let bb_lower = bb.as_ref().and_then(|b| b.lower.last().and_then(|x| *x)).unwrap_or(last * 0.98);
    let bb_pct_b = if bb_upper > bb_lower {
        (last - bb_lower) / (bb_upper - bb_lower)
    } else {
        0.5
    };

    let kc = indicators::keltner_channels(&highs, &lows, &closes, 20, 14, 1.5).ok();
    let kc_upper = kc.as_ref().and_then(|k| k.upper.last().and_then(|x| *x)).unwrap_or(last * 1.02);
    let kc_lower = kc.as_ref().and_then(|k| k.lower.last().and_then(|x| *x)).unwrap_or(last * 0.98);
    let kc_pos = if kc_upper > kc_lower {
        (last - kc_lower) / (kc_upper - kc_lower)
    } else {
        0.5
    };

    let dc = indicators::donchian_channels(&highs, &lows, 20).ok();
    let dc_upper = dc.as_ref().and_then(|d| d.upper.last().and_then(|x| *x)).unwrap_or(last);
    let dc_lower = dc.as_ref().and_then(|d| d.lower.last().and_then(|x| *x)).unwrap_or(last);
    let dc_pos = if dc_upper > dc_lower {
        (last - dc_lower) / (dc_upper - dc_lower)
    } else {
        0.5
    };

    let tr_last = indicators::true_range(&highs, &lows, &closes)
        .ok()
        .and_then(|v| v.last().and_then(|x| *x))
        .unwrap_or(atr14);
    let tr_atr_ratio = (tr_last / atr14).min(5.0);

    // ── Volume (6 features) ──────────────────────────────────────────────────
    let obv_vals = indicators::obv(&closes, &volumes).unwrap_or_default();
    let obv_slope = compute_slope(&obv_vals, 5);

    let cmf20 = last_val(
        indicators::cmf(&highs, &lows, &closes, &volumes, 20).unwrap_or_default(),
        0.0,
    );
    let chaikin = last_val(
        indicators::chaikin_oscillator(&highs, &lows, &closes, &volumes).unwrap_or_default(),
        0.0,
    );
    let mfi14 = last_val(
        indicators::mfi(&highs, &lows, &closes, &volumes, 14).unwrap_or_default(),
        50.0,
    );
    let vwap = last_val(
        indicators::vwap(&highs, &lows, &closes, &volumes).unwrap_or_default(),
        last,
    );
    let vwma20 = last_val(
        indicators::vwma(&closes, &volumes, 20).unwrap_or_default(),
        last,
    );

    // ── Trend (6 features) ───────────────────────────────────────────────────
    let adx14 = last_val(
        indicators::adx(&highs, &lows, &closes, 14).unwrap_or_default(),
        0.0,
    );
    let aroon = indicators::aroon(&highs, &lows, 25).ok();
    let aroon_osc = {
        let up = aroon.as_ref().and_then(|a| a.aroon_up.last().and_then(|x| *x)).unwrap_or(50.0);
        let dn = aroon.as_ref().and_then(|a| a.aroon_down.last().and_then(|x| *x)).unwrap_or(50.0);
        up - dn
    };

    let ichi = indicators::ichimoku(&highs, &lows, &closes, 9, 26, 26, 26).ok();
    let span_a = ichi.as_ref().and_then(|i| i.leading_span_a.last().and_then(|x| *x)).unwrap_or(last);
    let span_b = ichi.as_ref().and_then(|i| i.leading_span_b.last().and_then(|x| *x)).unwrap_or(last);
    let cloud_top = span_a.max(span_b);
    let cloud_bot = span_a.min(span_b);
    let ichi_pos: f32 = if last > cloud_top {
        1.0
    } else if last < cloud_bot {
        -1.0
    } else {
        0.0
    };

    let psar = indicators::parabolic_sar(&highs, &lows, &closes, 0.02, 0.2).ok();
    let psar_val = psar.and_then(|v| v.last().and_then(|x| *x)).unwrap_or(last);
    let psar_dir: f32 = if last > psar_val { 1.0 } else { -1.0 };

    let st = indicators::supertrend(&highs, &lows, &closes, 10, 3.0).ok();
    let st_dir: f32 = st
        .and_then(|v| v.is_uptrend.last().and_then(|x| *x))
        .map(|up| if up { 1.0 } else { -1.0 })
        .unwrap_or(0.0);

    let chop = last_val(
        indicators::choppiness_index(&highs, &lows, &closes, 14).unwrap_or_default(),
        50.0,
    );

    // ── Other (6 features) ───────────────────────────────────────────────────
    let bop = last_val(
        indicators::balance_of_power(&opens, &highs, &lows, &closes, None).unwrap_or_default(),
        0.0,
    );
    let bbp = indicators::bull_bear_power(&highs, &lows, &closes, 13).ok();
    let bull_power = bbp.as_ref().and_then(|b| b.bull_power.last().and_then(|x| *x)).unwrap_or(0.0);
    let bear_power = bbp.as_ref().and_then(|b| b.bear_power.last().and_then(|x| *x)).unwrap_or(0.0);

    let er = indicators::elder_ray(&highs, &lows, &closes, 13).ok();
    let er_bull = er.as_ref().and_then(|e| e.bull_power.last().and_then(|x| *x)).unwrap_or(0.0);
    let er_bear = er.as_ref().and_then(|e| e.bear_power.last().and_then(|x| *x)).unwrap_or(0.0);

    let ad_vals = indicators::accumulation_distribution(&highs, &lows, &closes, &volumes)
        .unwrap_or_default();
    let ad_slope = compute_slope(&ad_vals, 5);

    // ── Assemble [f32; 42] ───────────────────────────────────────────────────
    [
        // Moving Averages (8)
        ma_norm(sma20),
        ma_norm(sma50),
        ma_norm(sma200),
        ma_norm(ema12),
        ma_norm(dema20),
        ma_norm(hma20),
        ma_norm(alma20),
        ma_norm(mcg20),
        // Momentum (11)
        scale(rsi14, 0.0, 100.0),
        zs("macd_hist", macd_hist / atr14),
        scale(stoch_k, 0.0, 100.0),
        scale(stoch_d, 0.0, 100.0),
        scale(srsi_k, 0.0, 1.0),
        (cci20 / 300.0).clamp(-1.0, 1.0) as f32,
        scale(willr14, -100.0, 0.0),
        scale(cmo14, -100.0, 100.0),
        zs("roc10", roc10),
        zs("ao", ao / atr14),
        zs("coppock", coppock),
        // Volatility (5)
        (atr14 / last).clamp(0.0, 0.1) as f32 * 10.0, // ATR/price × 10 → [0,1]
        clamp01(bb_pct_b),
        clamp01(kc_pos),
        clamp01(dc_pos),
        clamp01(tr_atr_ratio / 5.0),
        // Volume (6)
        zs("obv_slope", obv_slope),
        cmf20.clamp(-1.0, 1.0) as f32,
        zs("chaikin", chaikin / atr14),
        scale(mfi14, 0.0, 100.0),
        ma_norm(vwap),
        ma_norm(vwma20),
        // Trend (6)
        clamp01(adx14 / 100.0),
        scale(aroon_osc, -100.0, 100.0),
        ichi_pos,
        psar_dir,
        st_dir,
        clamp01(chop / 100.0),
        // Other (6)
        bop.clamp(-1.0, 1.0) as f32,
        zs("bull_power", bull_power / atr14),
        zs("bear_power", bear_power / atr14),
        zs("er_bull", er_bull / atr14),
        zs("er_bear", er_bear / atr14),
        zs("ad_slope", ad_slope),
    ]
}

/// Compute OLS slope of the last `window` values (normalized by std).
fn compute_slope(vals: &[Option<f64>], window: usize) -> f64 {
    let v: Vec<f64> = vals.iter().rev().take(window).filter_map(|x| *x).collect();
    if v.len() < 2 {
        return 0.0;
    }
    let n = v.len() as f64;
    let sum_x: f64 = (0..v.len()).map(|i| i as f64).sum();
    let sum_y: f64 = v.iter().sum();
    let sum_xy: f64 = v.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..v.len()).map(|i| (i as f64).powi(2)).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        0.0
    } else {
        (n * sum_xy - sum_x * sum_y) / denom
    }
}
