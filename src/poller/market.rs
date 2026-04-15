use crate::events::{EventRecord, detector::detect_trending_events};
use crate::store::{
    CachedValue, CryptoMarket, DataStore, FearGreedData, MarketSummaryData, SectorData,
    WorldIndices,
};
use finance_query::{IndicesRegion, Region, Sector, crypto, finance};
use std::collections::HashSet;
use std::sync::Arc;

pub async fn poll_world_indices(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let americas = finance::indices(Some(IndicesRegion::Americas)).await;
    let europe = finance::indices(Some(IndicesRegion::Europe)).await;
    let asia = finance::indices(Some(IndicesRegion::AsiaPacific)).await;

    let mut wi = WorldIndices::default();

    for result in [americas, europe, asia] {
        if let Ok(batch) = result {
            for (sym, quote) in &batch.quotes {
                let ret = quote
                    .regular_market_change_percent
                    .as_ref()
                    .and_then(|v| v.raw)
                    .unwrap_or(0.0)
                    / 100.0;
                wi.returns.insert(sym.clone(), ret);
            }
        }
    }

    for (sym, field) in [("UUP", "dxy"), ("GLD", "gold"), ("USO", "oil")] {
        if let Some(quote) = wi.returns.get(sym).copied() {
            match field {
                "dxy" => wi.dxy = quote,
                "gold" => wi.gold = quote,
                "oil" => wi.oil = quote,
                _ => {}
            }
        }
    }

    *store.world_indices.write() = CachedValue::new(wi);
    tracing::debug!("World indices updated");
    Ok(())
}

pub async fn poll_crypto_market(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let btc = crypto::coin("bitcoin", "usd").await?;
    let eth = crypto::coin("ethereum", "usd").await?;
    let coins = crypto::coins("usd", 100).await?;

    let total_mktcap: f64 = coins.iter().filter_map(|c| c.market_cap).sum();
    let btc_mktcap = btc.market_cap.unwrap_or(0.0);
    let btc_dominance = if total_mktcap > 0.0 {
        btc_mktcap / total_mktcap
    } else {
        0.0
    };

    let prev = store.crypto.read().value.btc_price;
    let btc_price = btc.current_price.unwrap_or(prev);
    let eth_prev = store.crypto.read().value.eth_price;
    let eth_price = eth.current_price.unwrap_or(eth_prev);

    let cm = CryptoMarket {
        btc_price,
        btc_return_1d: btc.price_change_percentage_24h.unwrap_or(0.0) / 100.0,
        eth_price,
        eth_return_1d: eth.price_change_percentage_24h.unwrap_or(0.0) / 100.0,
        total_mktcap,
        btc_dominance,
    };

    *store.crypto.write() = CachedValue::new(cm);
    tracing::debug!("Crypto market updated");
    Ok(())
}

pub async fn poll_fear_greed(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let fg = finance::fear_and_greed().await?;
    let classification = classify_fear_greed(fg.value);
    let data = FearGreedData {
        value: fg.value as f64,
        classification,
    };
    *store.fear_greed.write() = CachedValue::new(data);
    tracing::debug!(value = fg.value, "Fear & Greed updated");
    Ok(())
}

fn classify_fear_greed(value: u8) -> u8 {
    match value {
        0..=24 => 0,
        25..=44 => 1,
        45..=55 => 2,
        56..=75 => 3,
        _ => 4,
    }
}

pub async fn poll_sector_industry(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let sectors = [
        Sector::Technology,
        Sector::FinancialServices,
        Sector::ConsumerCyclical,
        Sector::CommunicationServices,
        Sector::Healthcare,
        Sector::Industrials,
        Sector::ConsumerDefensive,
        Sector::Energy,
        Sector::BasicMaterials,
        Sector::RealEstate,
        Sector::Utilities,
    ];

    let mut returns = [0.0f64; 11];
    for (i, sector) in sectors.iter().enumerate() {
        match finance::sector(*sector).await {
            Ok(data) => {
                let ret = data
                    .performance
                    .as_ref()
                    .and_then(|p| p.day_change_percent.as_ref())
                    .and_then(|v| v.raw)
                    .unwrap_or(0.0);
                returns[i] = ret;
            }
            Err(e) => {
                tracing::warn!(sector = ?sector, err = %e, "sector poll failed");
            }
        }
    }

    let mean = returns.iter().sum::<f64>() / 11.0;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 11.0;
    let dispersion = variance.sqrt();

    let best = returns
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let worst = returns
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let sector_data = SectorData {
        returns,
        dispersion,
        best_sector: best,
        worst_sector: worst,
        rotation_velocity: 0.0,
        breadth_pct_above_sma200: 0.0,
    };

    *store.sector.write() = CachedValue::new(sector_data);
    tracing::debug!("Sector data updated");
    Ok(())
}

pub async fn poll_market_summary(store: &Arc<DataStore>) -> anyhow::Result<()> {
    let summary = finance::market_summary(Some(Region::UnitedStates)).await?;

    let old_trending: HashSet<String> = {
        store
            .symbols
            .iter()
            .filter(|_s| store.market_summary.read().value.trending_count > 0.0)
            .cloned()
            .collect()
    };

    let trending_result = finance::trending(Some(Region::UnitedStates)).await;
    let new_trending: HashSet<String> = trending_result
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.symbol)
        .collect();

    let events: Vec<EventRecord> = detect_trending_events(&old_trending, &new_trending);
    for event in events {
        if let Some(sym) = &event.symbol {
            if store.symbols.contains(sym) {
                let mut es = store.event_state.entry(sym.clone()).or_default();
                *es = es.with_event(&event);
                tracing::info!(symbol = %sym, "TrendingEntry event");
            }
        }
    }

    // Compute advance/decline ratio: regular_market_change is FormattedValue<f64>
    let adv = summary
        .iter()
        .filter(|q| {
            q.regular_market_change
                .as_ref()
                .and_then(|v| v.raw)
                .unwrap_or(0.0)
                > 0.0
        })
        .count() as f64;
    let dec = summary
        .iter()
        .filter(|q| {
            q.regular_market_change
                .as_ref()
                .and_then(|v| v.raw)
                .unwrap_or(0.0)
                < 0.0
        })
        .count() as f64;
    let total = (adv + dec).max(1.0);

    let ms_data = MarketSummaryData {
        advance_decline: ((adv - dec) / total).clamp(-1.0, 1.0),
        new_highs_lows: 0.0,
        spy_vol_5d: 0.0,
        qqq_vol_5d: 0.0,
        iwm_vol_5d: 0.0,
        cross_asset_corr: 0.0,
        trending_count: new_trending.len() as f64,
        trending_portfolio_overlap: new_trending
            .intersection(&store.symbols.iter().cloned().collect())
            .count() as f64,
        gainers_momentum: 0.0,
        actives_vol_ratio: 0.0,
        market_hours_state: 0.0,
        minutes_since_event: 0.0,
    };

    *store.market_summary.write() = CachedValue::new(ms_data);
    tracing::debug!("Market summary updated");
    Ok(())
}
