# MERIDIAN: Regime-Adaptive State-Space Portfolio Model

## Architectural Specification v3.0 — Full Data Integration

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Novel Contributions](#3-novel-contributions)
4. [Hardware Target & Constraints](#4-hardware-target--constraints)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Data Ingestion Architecture](#6-data-ingestion-architecture)
   - 6.1 [Data Bus Design](#61-data-bus-design)
   - 6.2 [Per-Asset Feature Assembly (120 dims)](#62-per-asset-feature-assembly-120-dims)
   - 6.3 [Market-Wide Feature Assembly (96 dims)](#63-market-wide-feature-assembly-96-dims)
   - 6.4 [Event Stream Architecture](#64-event-stream-architecture)
   - 6.5 [Text Corpus Pipeline](#65-text-corpus-pipeline)
   - 6.6 [Portfolio State Vector (201 dims)](#66-portfolio-state-vector-201-dims)
   - 6.7 [Financial Calendar Embedding (20 dims)](#67-financial-calendar-embedding-20-dims)
7. [Model Components](#7-model-components)
   - 7.1 [Input Projection Layer with Variable Selection](#71-input-projection-layer-with-variable-selection)
   - 7.2 [Regime Detector](#72-regime-detector)
   - 7.3 [Per-Asset Temporal Backbone (RA-GDN Stack)](#73-per-asset-temporal-backbone-ra-gdn-stack)
   - 7.4 [Regime-Routed Sparse Mixture of Experts](#74-regime-routed-sparse-mixture-of-experts)
   - 7.5 [Temporal Cascade](#75-temporal-cascade)
   - 7.6 [Cross-Asset Fusion Module](#76-cross-asset-fusion-module)
   - 7.7 [Financial Episodic Memory](#77-financial-episodic-memory)
   - 7.8 [Portfolio Output Heads](#78-portfolio-output-heads)
   - 7.9 [Auxiliary Prediction Head](#79-auxiliary-prediction-head)
   - 7.10 [MarketReader Text Encoder](#710-marketreader-text-encoder)
8. [Continuous Learning System](#8-continuous-learning-system)
   - 8.1 [Tier 1 — Surprise-Gated Delta Dynamics](#81-tier-1--surprise-gated-delta-dynamics)
   - 8.2 [Tier 2 — Normalization Adaptation](#82-tier-2--normalization-adaptation)
   - 8.3 [Tier 3 — Periodic Consolidation (EWC + Ternary LoRA)](#83-tier-3--periodic-consolidation-ewc--ternary-lora)
   - 8.4 [Adaptation Feedback Loop](#84-adaptation-feedback-loop)
9. [Quantization Strategy](#9-quantization-strategy)
10. [Training Pipeline](#10-training-pipeline)
    - 10.1 [Data Sources & Preprocessing](#101-data-sources--preprocessing)
    - 10.2 [Phase 1 — Self-Supervised Pre-training](#102-phase-1--self-supervised-pre-training)
    - 10.3 [Phase 2 — Portfolio Optimization Training](#103-phase-2--portfolio-optimization-training)
    - 10.4 [Phase 3 — Meta-Learning for Plasticity](#104-phase-3--meta-learning-for-plasticity)
    - 10.5 [Phase 4 — Quantization-Aware Training](#105-phase-4--quantization-aware-training)
11. [Two-Head Strategy (Meridian-R / Meridian-S)](#11-two-head-strategy-meridian-r--meridian-s)
12. [Historical Backtesting Framework](#12-historical-backtesting-framework)
    - 12.1 [Meridian as a Backtesting Strategy](#121-meridian-as-a-backtesting-strategy)
    - 12.2 [Benchmark Comparisons](#122-benchmark-comparisons)
    - 12.3 [Walk-Forward Validation](#123-walk-forward-validation)
    - 12.4 [Monte Carlo Confidence Intervals](#124-monte-carlo-confidence-intervals)
    - 12.5 [Regime-Segmented Analysis](#125-regime-segmented-analysis)
13. [Paper Trading System](#13-paper-trading-system)
    - 13.1 [Architecture](#131-architecture)
    - 13.2 [Performance Tracking](#132-performance-tracking)
    - 13.3 [Adaptation Monitoring](#133-adaptation-monitoring)
    - 13.4 [Graduation Criteria (Paper → Live)](#134-graduation-criteria-paper--live)
14. [Inference Runtime on Pi 5](#14-inference-runtime-on-pi-5)
15. [Memory Budget](#15-memory-budget)
16. [Development Roadmap](#16-development-roadmap)
17. [Risk Analysis & Mitigations](#17-risk-analysis--mitigations)
18. [Appendix A — Parameter Count Breakdown](#appendix-a--parameter-count-breakdown)
19. [Appendix B — Complete finance-query Integration Map](#appendix-b--complete-finance-query-integration-map)
20. [Appendix C — Hyperparameter Reference](#appendix-c--hyperparameter-reference)
21. [Appendix D — Event Type Catalog](#appendix-d--event-type-catalog)

---

## 1. Executive Summary

Meridian is a novel CPU-native neural network architecture designed to manage a portfolio of up to 100 US-listed assets (equities, ETFs, crypto) 24/7 on a Raspberry Pi 5. It ingests **every available data stream** from finance-query — price, volume, 42 technical indicators, 30+ RSS/news feeds, full options chains, EDGAR filings, XBRL fundamentals, earnings transcripts, analyst ratings, insider transactions, institutional ownership, macro series, yield curves, sector/industry data, Fear & Greed, world indices, trending tickers, and crypto market data — producing allocation weights every minute through a shared backbone with two output heads.

The model makes **proactive decisions** by processing discrete events (insider buys, analyst upgrades, SEC filings, earnings surprises, FOMC announcements) as they arrive, not just reacting to price changes. It is designed for both historical backtesting via finance-query's backtesting engine and real-time paper trading, with a continuous learning system that measurably improves over time.

Key metrics:

| Metric | Value |
|---|---|
| Total parameters | ~95M (35M active per forward pass) |
| Input features per asset | 120 scalar + 64 text embedding = 184 |
| Market-wide features | 96 scalar |
| Event types tracked | 18 discrete event categories |
| Text sources ingested | 30+ RSS feeds + per-ticker news + transcripts |
| Quantized model size | ~23 MB |
| Runtime memory footprint | ~180 MB |
| Inference latency (100 assets) | ~5–10 ms |
| Data pipeline latency | ~200–500 ms (network-bound) |
| Adaptation mechanisms | 3-tier (sub-second / hourly / daily) |
| Training compute | ~250–400 H100-hours |

---

## 2. Design Philosophy

Six interlocking principles govern every architectural decision:

**Information maximalism.** The model should ingest every available signal — not just price and volume, but the full constellation of fundamental, sentiment, ownership, regulatory, macro, and cross-asset data that professional portfolio managers use. More information, properly compressed and weighted, strictly dominates less information for decision-making. The Variable Selection Network ensures irrelevant signals are gated to zero rather than adding noise.

**Temporal nativeness.** Financial markets are continuous processes with multi-scale dynamics. The Gated DeltaNet backbone models time as a first-class citizen through a recurrent state that captures infinite history, updates in O(1) per step, and requires no re-processing of context.

**Event-driven proactivity.** Markets move on discrete events: an insider purchase, an analyst upgrade, a Fed statement, an earnings miss. Meridian processes events as first-class inputs — not by waiting for their price impact, but by incorporating the event signal immediately when it arrives. The model learns the typical price dynamics following each event type and can position proactively.

**Regime awareness.** Markets change. Meridian addresses this at three levels: recurrent dynamics adapt based on detected regime, MoE routes to specialized sub-networks, and continuous learning updates parameters using prediction errors.

**Extreme efficiency.** Every operation must justify its compute cost on a Cortex-A76. Ternary weights from inception, O(1) recurrence, no quadratic attention over long sequences.

**Separation of understanding from strategy.** The backbone learns market state. The output heads learn allocation strategy. This enables modular head swapping and backtesting flexibility.

---

## 3. Novel Contributions

Seven architectural innovations distinguish Meridian from existing work. These are unchanged from v2.0:

1. **Regime-Aware Gated DeltaNet (RA-GDN)** — regime embedding modulates forgetting gate α and update strength β
2. **Surprise-Gated Plasticity (SGP)** — delta rule's prediction error drives adaptive learning rates
3. **Temporal Cascade with Adaptive Speed** — multi-resolution GDN stacks with surprise-gated downward injection
4. **Financial Episodic Memory** — FAISS k-NN buffer of surprising market states
5. **External Regime-Routed Sparse MoE** — HMM/neural posterior drives expert selection
6. **Financial Calendar Embedding** — market-specific temporal encoding
7. **Ternary-Native Architecture with Precision Highway** — FP16 state, ternary weights

**New in v3.0:**

8. **Multi-Modal Event Fusion** — discrete financial events (insider trades, analyst changes, filings, macro releases) are encoded as typed event tokens and injected into the temporal backbone alongside continuous features, enabling proactive positioning before price impact materializes.

9. **Hierarchical Text Aggregation** — MarketReader processes text at three granularities: headline-level (RSS titles, 4-dim sentiment vector), summary-level (article bodies, 32-dim topic+sentiment vector), and deep-level (earnings transcripts, 64-dim full embedding), with attention-weighted fusion per asset.

---

## 4. Hardware Target & Constraints

Unchanged from v2.0 — Raspberry Pi 5 (8GB), BCM2712 quad-core Cortex-A76 @ 2.4 GHz, NEON SIMD. Training on H100 cloud (500-hour budget).

---

## 5. High-Level Architecture

```
═══════════════════════════════════════════════════════════════════
                     DATA INGESTION LAYER
═══════════════════════════════════════════════════════════════════

CONTINUOUS STREAMS (every minute)                    EVENT STREAMS (as they arrive)
├── PriceStream (100 tickers)                        ├── RSS/News (30+ sources)
├── Technical Indicators (42 indicators)             ├── Insider Transactions
├── VIX, DXY proxy, Oil, Gold, BTC                   ├── Analyst Upgrades/Downgrades
├── Fear & Greed Index                               ├── SEC Filings (EDGAR)
├── Market Summary & Breadth                         ├── Earnings Releases
└── Portfolio State                                  ├── FOMC / Macro Releases
                                                     ├── Options Flow Anomalies
PERIODIC UPDATES                                     ├── Institutional Ownership Changes
├── Options Chains (every 15 min)                    └── Trending Ticker Signals
├── Fundamentals/XBRL (quarterly)
├── Yield Curve (daily)
├── Sector/Industry Performance (daily)
├── Earnings Transcripts (quarterly)
├── Analyst Ratings Snapshot (daily)
├── Institutional Ownership (quarterly)
├── Insider Holdings Snapshot (daily)
└── World Indices (hourly)

         │                                    │
         ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA BUS                                  │
│  Normalizes, aligns, and routes all data to model components    │
│  Per-asset: 120 continuous features + 64 text embedding = 184   │
│  Market-wide: 96 features                                       │
│  Events: typed event tokens with embeddings                     │
│  Portfolio: 201 features                                        │
│  Calendar: 20 features                                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
═══════════════════════════════════════════════════════════════════
                       MODEL CORE
═══════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────┐
│  INPUT PROJECTION LAYER                               │
│  Per-asset: VSN-Gated Linear(184 → D) + RMSNorm      │
│  Market:    Linear(96 → D) + RMSNorm                  │
│  Portfolio: Linear(201 → D) + RMSNorm                 │
│  Calendar:  Linear(20 → D)                            │
│  Events:    EventEncoder(event → D)                   │
│  D = 768                                              │
└───────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  REGIME DETECTOR                                      │
│  Input: market-wide (96) + portfolio summary + regime │
│  Output: regime_embedding(64), posteriors(4), velocity│
└───────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  PER-ASSET TEMPORAL BACKBONE (RA-GDN)                 │
│  8 RA-GDN blocks + 3 FMoE blocks (4 experts each)    │
│  Shared weights, separate state per asset             │
│  Event tokens injected at natural arrival time        │
└───────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  TEMPORAL CASCADE                                     │
│  Hourly / Daily / Weekly GDN layers                   │
│  Bidirectional with surprise-gated injection          │
└───────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  CROSS-ASSET FUSION                                   │
│  110+ tokens: 100 assets + market + portfolio         │
│              + 8 episodic memory retrievals            │
│  2 layers MHA with sector bias prior                  │
└───────────────────────────────────────────────────────┘
         │
         ├──────────────┬──────────────────┐
         ▼              ▼                  ▼
   ┌──────────┐  ┌──────────┐  ┌────────────────┐
   │RETURN HD │  │SHARPE HD │  │AUXILIARY HEAD   │
   │(Merid-R) │  │(Merid-S) │  │(surprise comp) │
   └──────────┘  └──────────┘  └────────────────┘
         │
         ▼
═══════════════════════════════════════════════════════════════════
                  EXECUTION LAYER (abstracted)
═══════════════════════════════════════════════════════════════════
  Paper Trading Engine ←→ Backtesting Engine (finance-query)
```

---

## 6. Data Ingestion Architecture

This is the most significant expansion from v2.0. Meridian now ingests the full breadth of finance-query's API surface.

### 6.1 Data Bus Design

The Data Bus is a Rust process running on the Pi that:

1. **Maintains persistent connections** to `PriceStream` for real-time prices
2. **Polls periodic APIs** on configurable schedules
3. **Subscribes to event streams** (RSS feeds, news) via background tasks
4. **Normalizes all data** into model-ready feature vectors using rolling statistics
5. **Routes data** to the correct model input channels
6. **Caches everything** with staleness tracking — the model always has a complete feature vector, even if some APIs are temporarily unavailable

```
DataBus Architecture:

┌─────────────────────────────────────────────┐
│  SCHEDULER                                   │
│                                              │
│  every 1s:   PriceStream → price_buffer     │
│  every 1m:   assemble_features() → model    │
│  every 5m:   poll_rss_feeds()               │
│  every 15m:  poll_options_chains()           │
│  every 1h:   poll_world_indices()            │
│  every 1h:   poll_crypto_market()            │
│  every 6h:   poll_insider_transactions()     │
│  every 24h:  poll_analyst_ratings()          │
│  every 24h:  poll_yield_curve()              │
│  every 24h:  poll_sector_industry()          │
│  every 24h:  poll_fear_greed()               │
│  every 24h:  poll_market_summary()           │
│  on_release: poll_fred_macro()               │
│  quarterly:  poll_fundamentals()             │
│  quarterly:  poll_earnings_transcripts()     │
│  quarterly:  poll_institutional_ownership()  │
│                                              │
│  EVENT DETECTOR (continuous):                │
│    RSS diff → new_article_event              │
│    insider_tx diff → insider_event           │
│    grading diff → analyst_event              │
│    sec_filings diff → filing_event           │
│    earnings actual vs estimate → earnings_ev │
│    FRED release → macro_event                │
│    options anomaly detector → options_event   │
│    trending diff → trending_event            │
└─────────────────────────────────────────────┘
```

### 6.2 Per-Asset Feature Assembly (120 dims)

Every minute, each of 100 assets gets a 120-dimensional continuous feature vector plus a 64-dimensional text embedding, for 184 effective dims total.

#### Block 1: Price & Volume (10 features)

Unchanged from v2.0.

| Feature | Computation | Source |
|---|---|---|
| Open (normalized) | `ln(open / prev_close)` | `PriceStream` / `chart()` |
| High (normalized) | `ln(high / open)` | `PriceStream` / `chart()` |
| Low (normalized) | `ln(low / open)` | `PriceStream` / `chart()` |
| Close (normalized) | `ln(close / prev_close)` | `PriceStream` / `chart()` |
| Volume (normalized) | `ln(volume / SMA_20(volume))` | `PriceStream` / `chart()` |
| Return 1m | `ln(close / close_{t-1})` | Computed |
| Return 5m | `ln(close / close_{t-5})` | Computed |
| Return 15m | `ln(close / close_{t-15})` | Computed |
| Return 60m | `ln(close / close_{t-60})` | Computed |
| Return 1d | `ln(close / close_{t-390})` | Computed |

#### Block 2: Technical Indicators — Full Suite (42 features)

**Expanded from 20 to 42** — now using every indicator finance-query provides. Each indicator is z-scored with rolling 252-day statistics.

**Moving Averages (8 features):**

| Feature | Source | Encoding |
|---|---|---|
| SMA(20) z-score | `indicators::sma(&closes, 20)` | `(price - SMA) / ATR` |
| SMA(50) z-score | `indicators::sma(&closes, 50)` | `(price - SMA) / ATR` |
| SMA(200) z-score | `indicators::sma(&closes, 200)` | `(price - SMA) / ATR` |
| EMA(12) z-score | `indicators::ema(&closes, 12)` | `(price - EMA) / ATR` |
| DEMA(20) z-score | `indicators::dema(&closes, 20)` | `(price - DEMA) / ATR` |
| HMA(20) z-score | `indicators::hma(&closes, 20)` | `(price - HMA) / ATR` |
| ALMA(20) z-score | `indicators::alma(&closes, 20, ...)` | `(price - ALMA) / ATR` |
| McGinley(20) z-score | `indicators::mcginley_dynamic(&closes, 20)` | `(price - McG) / ATR` |

All MA z-scores are normalized by ATR to make them comparable across assets with different volatility levels.

**Momentum (11 features):**

| Feature | Source | Range |
|---|---|---|
| RSI(14) | `indicators::rsi(&closes, 14)` | [0, 100] → scaled to [-1, 1] |
| MACD histogram / ATR | `indicators::macd(&closes, 12, 26, 9)` | Normalized |
| Stochastic %K(14,3,3) | `indicators::stochastic(h,l,c, 14, 3, 3)` | [0, 100] → [-1, 1] |
| Stochastic %D(14,3,3) | Same call, `.d` field | [0, 100] → [-1, 1] |
| Stochastic RSI %K | `indicators::stochastic_rsi(c, 14, 14, 3, 3)` | [0, 1] → [-1, 1] |
| CCI(20) | `indicators::cci(&highs, &lows, &closes, 20)` | Clamped [-3, 3] → [-1, 1] |
| Williams %R(14) | `indicators::williams_r(&highs, &lows, &closes, 14)` | [-100, 0] → [-1, 1] |
| CMO(14) | `indicators::cmo(&closes, 14)` | [-100, 100] → [-1, 1] |
| ROC(10) | `indicators::roc(&closes, 10)` | Z-scored |
| Awesome Oscillator | `indicators::awesome_oscillator(&highs, &lows, 5, 34)` | Normalized by ATR |
| Coppock Curve | `indicators::coppock_curve(&closes, 14, 11, 10)` | Z-scored |

**Volatility (5 features):**

| Feature | Source | Encoding |
|---|---|---|
| ATR(14) / price | `indicators::atr(&h, &l, &c, 14)` | Percentage of price |
| Bollinger %B(20,2) | `indicators::bollinger_bands(&c, 20, 2.0)` | [0, 1] typically |
| Keltner position | `indicators::keltner_channels(&h, &l, &c, 20, 1.5)` | Position within channel |
| Donchian position | `indicators::donchian_channels(&h, &l, 20)` | Position within channel |
| True Range / ATR ratio | `indicators::true_range(&h, &l, &c)` / ATR(14) | Spike detector |

**Volume (6 features):**

| Feature | Source | Encoding |
|---|---|---|
| OBV 5-period slope | `indicators::obv(&closes, &volumes)` | Linear regression slope |
| CMF(20) | `indicators::cmf(&h, &l, &c, &v, 20)` | [-1, 1] |
| Chaikin Oscillator | `indicators::chaikin_oscillator(&h, &l, &c, &v, 3, 10)` | Z-scored |
| MFI(14) | `indicators::mfi(&h, &l, &c, &v, 14)` | [0, 100] → [-1, 1] |
| VWAP deviation | `indicators::vwap(&h, &l, &c, &v)` | `(price - VWAP) / ATR` |
| VWMA(20) deviation | `indicators::vwma(&closes, &volumes, 20)` | `(price - VWMA) / ATR` |

**Trend (6 features):**

| Feature | Source | Encoding |
|---|---|---|
| ADX(14) | `indicators::adx(&h, &l, &c, 14)` | [0, 100] → [0, 1] |
| Aroon oscillator | `indicators::aroon(&h, &l, 25)` | [-100, 100] → [-1, 1] |
| Ichimoku cloud position | `indicators::ichimoku(h, l, c, 9, 26, 26, 26)` | -1 (below) / 0 (inside) / +1 (above) |
| Parabolic SAR direction | `indicators::parabolic_sar(&h, &l, 0.02, 0.2)` | -1 (bearish) / +1 (bullish) |
| SuperTrend direction | `indicators::supertrend(&h, &l, &c, 10, 3.0)` | -1 / +1 |
| Choppiness Index(14) | `indicators::choppiness_index(&h, &l, &c, 14)` | [0, 100] → [0, 1] |

**Other (6 features):**

| Feature | Source | Encoding |
|---|---|---|
| Balance of Power | `indicators::balance_of_power(&o, &h, &l, &c)` | [-1, 1] |
| Bull Power(13) | `indicators::bull_bear_power(&h, &l, &c, 13)` | Z-scored |
| Bear Power(13) | Same call | Z-scored |
| Elder Ray Bull(13) | `indicators::elder_ray(&h, &l, &c, 13)` | Z-scored |
| Elder Ray Bear(13) | Same call | Z-scored |
| Accumulation/Distribution | `indicators::accumulation_distribution(&h, &l, &c, &v)` | 5-period slope |

**Total technical indicators: 42 features.**

#### Block 3: Options-Derived (12 features)

**Expanded from 5 to 12** — extracting the full options chain signal. Sourced from `ticker.options(date)` polled every 15 minutes.

| Feature | Computation | Notes |
|---|---|---|
| 30-day ATM IV | Weighted avg of near-money call/put IV | Primary vol signal |
| IV percentile (1yr) | Percentile rank of current IV vs 252-day history | High/low IV context |
| IV term structure slope | Front month IV / back month IV ratio | Contango/backwardation |
| 25-delta put skew | 25δ put IV − ATM IV | Crash fear premium |
| 25-delta call skew | ATM IV − 25δ call IV | Upside demand |
| Put/call volume ratio | Total put vol / total call vol | Directional sentiment |
| Put/call OI ratio | Total put OI / total call OI | Longer-term positioning |
| Total OI change (daily) | ΔOI / OI_{yesterday} | Position building/unwinding |
| Volume/OI ratio | Today's volume / total OI | Activity intensity |
| Max pain proximity | `(price - max_pain) / ATR` | Options pinning signal |
| Unusual volume flag | Binary: volume > 2σ above 20-day average | Activity spike |
| IV crush proximity | Days to earnings < 5 AND IV > 80th percentile | Pre-earnings vol signal |

For assets without listed options (some ETFs, crypto): filled with a **learned 12-dim default embedding** per asset class.

#### Block 4: Fundamental & Ownership (30 features)

**Expanded from 10 to 30** — now incorporating the full depth of `financial_data()`, `key_stats()`, `earnings_trend()`, `earnings_history()`, insider activity, institutional ownership, and analyst ratings.

**Financial Fundamentals (12 features):**

| Feature | Source | Encoding |
|---|---|---|
| Trailing PE | `summary_detail().trailing_pe` | Cross-sectional percentile [0,1] |
| Forward PE | `summary_detail().forward_pe` | Cross-sectional percentile |
| PEG ratio | `key_stats().peg_ratio` | Clamped [0, 5] → [0, 1] |
| EV/EBITDA | `key_stats().enterprise_to_ebitda` | Cross-sectional percentile |
| Dividend yield | `summary_detail().dividend_yield` | Raw (bounded) |
| Profit margin | `financial_data().profit_margins` | Raw [-1, 1] |
| Revenue growth | `financial_data().revenue_growth` | Winsorized ±100% |
| Earnings growth | `financial_data().earnings_growth` | Winsorized ±100% |
| Debt-to-equity | `financial_data().debt_to_equity` | Cross-sectional percentile |
| Free cash flow yield | `financial_data().free_cashflow / market_cap` | Z-scored |
| Return on equity | `financial_data().return_on_equity` | Raw |
| Current ratio | `financial_data().current_ratio` | Clamped [0, 5] |

**Earnings Intelligence (6 features):**

| Feature | Source | Encoding |
|---|---|---|
| Last earnings surprise % | `earnings_history()` → most recent actual vs estimate | Z-scored |
| Earnings surprise streak | Count of consecutive beats/misses | [-5, 5] → [-1, 1] |
| Next-Q estimate revision (30d) | `earnings_trend()` → estimate change over 30 days | Z-scored |
| Next-Q estimate revision (90d) | `earnings_trend()` → estimate change over 90 days | Z-scored |
| Revenue estimate revision (30d) | `earnings_trend()` → revenue estimate change | Z-scored |
| Days since last earnings | From `calendar_events()` | `1 / (1 + days)` |

**Analyst & Ratings (4 features):**

| Feature | Source | Encoding |
|---|---|---|
| Analyst consensus score | `recommendation_trend()` → weighted avg of buy/hold/sell | [-1 (sell), +1 (buy)] |
| Strong buy fraction | `recommendation_trend()` → strongBuy / total | [0, 1] |
| Rating momentum (30d) | `grading_history()` → net upgrades minus downgrades in 30 days | Z-scored |
| Most recent grade direction | `grading_history()` → latest action | -1 (downgrade) / 0 / +1 (upgrade) |

**Ownership & Insider Activity (6 features):**

| Feature | Source | Encoding |
|---|---|---|
| Net insider shares (90d) | `insider_transactions()` → net bought - sold in 90 days | Z-scored |
| Insider buy count (90d) | `insider_transactions()` → count of buy transactions | Log-scaled |
| Net insider value (90d) | `insider_transactions()` → net dollar value | Z-scored |
| Insider ownership % | `insider_holders()` → total shares / outstanding | [0, 1] |
| Institutional ownership % | `major_holders()` → insitutional percentage | [0, 1] |
| Institutional Δ (quarterly) | `institution_ownership()` → change in top-10 total shares | Z-scored |

**Asset Identity (2 features):**

| Feature | Source | Encoding |
|---|---|---|
| Sector embedding | `asset_profile()` → GICS sector | Learned scalar per sector (11→1) |
| Market cap log-rank | `summary_detail().market_cap` | `ln(mktcap)` percentile [0, 1] |

#### Block 5: Risk Analytics (8 features)

**New block.** Sourced from `ticker.risk(Interval::OneDay, TimeRange::OneYear, Some("SPY"))`, refreshed daily.

| Feature | Source | Encoding |
|---|---|---|
| VaR 95% (1d) | `risk_summary.var_95` | Raw |
| VaR 99% (1d) | `risk_summary.var_99` | Raw |
| Parametric VaR 95% | `risk_summary.parametric_var_95` | Raw |
| Sharpe ratio (1yr) | `risk_summary.sharpe` | Clamped [-3, 3] |
| Sortino ratio (1yr) | `risk_summary.sortino` | Clamped [-3, 3] |
| Beta (vs SPY) | `risk_summary.beta` | Raw |
| Max drawdown (1yr) | `risk_summary.max_drawdown` | [0, 1] |
| Max DD recovery periods | `risk_summary.max_drawdown_recovery_periods` | Log-scaled |

#### Block 6: Event State (10 features)

**New block.** Encodes the recency and intensity of discrete events that have occurred for this asset. Each feature represents a decayed signal from the most recent event of that type.

| Feature | Event Type | Encoding |
|---|---|---|
| Insider buy recency | Last insider buy transaction | `exp(-minutes_since / (24*60))` |
| Insider buy magnitude | Value of last insider buy | Dollar amount, log-scaled, decayed |
| Analyst upgrade recency | Last upgrade in grading_history | `exp(-days_since / 30)` |
| Analyst downgrade recency | Last downgrade | `exp(-days_since / 30)` |
| SEC filing recency | Last SEC filing (any type) | `exp(-days_since / 7)` |
| 10-K/10-Q recency | Last annual/quarterly filing | `exp(-days_since / 30)` |
| News intensity (24h) | Count of news articles in last 24h | Log-scaled |
| News sentiment (24h) | Average MarketReader sentiment of recent news | [-1, +1] |
| Earnings proximity | Days to next earnings from `calendar_events()` | `exp(-days / 14)` |
| Ex-dividend proximity | Days to next ex-div from `calendar_events()` | `exp(-days / 7)` |

#### Block 7: Cross-Asset Context (8 features)

**New block.** Per-asset features derived from the asset's relationship to the broader market, computed by the data pipeline.

| Feature | Computation | Notes |
|---|---|---|
| Relative strength (20d) | Asset return / SPY return (20d) | Momentum vs market |
| Sector relative strength | Asset return / sector return (20d) | Within-sector momentum |
| Correlation to SPY (60d) | Rolling 60-day correlation | Beta-like but realized |
| Correlation change (Δ20d) | Correlation today - 20d ago | Decorrelation signal |
| Is trending ticker | `finance::trending(Region::US)` membership | Binary |
| Screener membership count | Number of `finance::screener()` lists containing this ticker | [0, 15] → [0, 1] |
| Recommendation similarity rank | `recommendations(5)` → overlap with portfolio | Count of recommended tickers in portfolio |
| Dividend analytics trend | `dividend_analytics()` payout trend | -1 (declining) / 0 / +1 (growing) |

#### Block 8: Text Embedding (64 features)

From MarketReader (Section 7.10) processing the full text corpus pipeline (Section 6.5).

**Total per-asset features: 10 + 42 + 12 + 30 + 8 + 10 + 8 = 120 continuous + 64 text = 184 effective dims.**

At inference, the 120 continuous features are concatenated with the 64-dim text embedding to form the full 184-dim per-asset input.

---

### 6.3 Market-Wide Feature Assembly (96 dims)

**Expanded from 24 to 96** — now incorporating world indices, full yield curve, sector/industry performance, crypto market state, currency movements, and richer macro signals.

#### Volatility & Sentiment (12 features)

| Feature | Source | Dims |
|---|---|---|
| VIX level | `Ticker::new("^VIX").price()` | 1 |
| VIX 1d change | Computed from VIX history | 1 |
| VIX term structure | VIX / VIX3M ratio (contango/backwardation) | 1 |
| Fear & Greed index | `finance::fear_and_greed()` | 1 |
| Fear & Greed classification | One-hot (extreme fear / fear / neutral / greed / extreme greed) | 5 |
| Market-wide put/call ratio | Aggregate from CBOE data or options universe | 1 |
| HY spread proxy | HYG/IEF price ratio | 1 |
| Credit stress | LQD/TLT ratio | 1 |

#### Yield Curve (12 features)

All 12 maturities from `fred::treasury_yields(year)`:

| Feature | Source |
|---|---|
| 1-month yield | `treasury_yield.y1m` |
| 3-month yield | `treasury_yield.y3m` |
| 6-month yield | `treasury_yield.y6m` |
| 1-year yield | `treasury_yield.y1` |
| 2-year yield | `treasury_yield.y2` |
| 3-year yield | `treasury_yield.y3` |
| 5-year yield | `treasury_yield.y5` |
| 7-year yield | `treasury_yield.y7` |
| 10-year yield | `treasury_yield.y10` |
| 20-year yield | `treasury_yield.y20` |
| 30-year yield | `treasury_yield.y30` |
| 2y-10y spread | Computed: y10 - y2 |

All yields are provided as raw values (already in percentage points). The model learns yield curve dynamics through the temporal backbone.

#### FRED Macro Series (12 features)

| Feature | FRED ID | Update Frequency |
|---|---|---|
| Fed Funds Rate | `fred::series("FEDFUNDS")` | Monthly |
| Unemployment Rate | `fred::series("UNRATE")` | Monthly |
| CPI YoY | `fred::series("CPIAUCSL")` | Monthly |
| Core CPI YoY | `fred::series("CPILFESL")` | Monthly |
| Initial Jobless Claims | `fred::series("ICSA")` | Weekly |
| Continuing Claims | `fred::series("CCSA")` | Weekly |
| ISM Manufacturing PMI | `fred::series("MANEMP")` proxy | Monthly |
| Retail Sales MoM | `fred::series("RSXFS")` | Monthly |
| Industrial Production | `fred::series("INDPRO")` | Monthly |
| Housing Starts | `fred::series("HOUST")` | Monthly |
| Consumer Confidence | `fred::series("UMCSENT")` | Monthly |
| Leading Economic Index | `fred::series("USSLIND")` | Monthly |

All macro features: raw value + Z-scored change from prior release. Held constant between releases (step function).

#### Sector & Industry Performance (16 features)

From `finance::sector(Sector::*)` across all 11 GICS sectors:

| Feature | Dims | Computation |
|---|---|---|
| Sector 1d returns (11 sectors) | 11 | Day change per sector |
| Sector dispersion | 1 | Std dev of 11 sector returns |
| Best sector ID | 1 | Index of top-performing sector |
| Worst sector ID | 1 | Index of worst-performing sector |
| Sector rotation velocity | 1 | 20d change in sector ranking correlation |
| Breadth (% above SMA200) | 1 | From `finance::market_summary()` |

#### World Indices & Cross-Asset (20 features)

From `finance::indices(IndicesRegion::*)`:

| Feature | Source | Dims |
|---|---|---|
| S&P 500 return (1d) | `finance::indices(Americas)` — SPX | 1 |
| NASDAQ return (1d) | `finance::indices(Americas)` — IXIC | 1 |
| Russell 2000 return (1d) | `finance::indices(Americas)` — RUT | 1 |
| Dow Jones return (1d) | `finance::indices(Americas)` — DJI | 1 |
| FTSE 100 return (1d) | `finance::indices(Europe)` | 1 |
| DAX return (1d) | `finance::indices(Europe)` | 1 |
| Nikkei 225 return (1d) | `finance::indices(AsiaPacific)` | 1 |
| Hang Seng return (1d) | `finance::indices(AsiaPacific)` | 1 |
| Shanghai Composite return | `finance::indices(AsiaPacific)` | 1 |
| DXY proxy (UUP) | `Ticker::new("UUP").price()` | 1 |
| EUR/USD return | From `finance::currencies()` or ticker | 1 |
| USD/JPY return | Same | 1 |
| Gold (GLD) return | `Ticker::new("GLD").price()` | 1 |
| Oil (USO) return | `Ticker::new("USO").price()` | 1 |
| BTC price + return | `crypto::coin("bitcoin", "usd")` | 2 |
| ETH price + return | `crypto::coin("ethereum", "usd")` | 2 |
| Crypto total mkt cap | `crypto::coins("usd", 1)` mkt cap | 1 |
| Crypto dominance (BTC) | BTC mkt cap / total | 1 |

#### Market Breadth & Flow (12 features)

| Feature | Source | Dims |
|---|---|---|
| Advance/decline ratio | `finance::market_summary(Region::US)` | 1 |
| New highs - new lows | `finance::market_summary(Region::US)` | 1 |
| SPY realized vol (5d/20d) | Computed from SPY chart | 1 |
| QQQ realized vol (5d/20d) | Computed from QQQ chart | 1 |
| IWM realized vol (5d/20d) | Computed from IWM chart | 1 |
| Cross-asset correlation (20d) | Avg pairwise corr across portfolio | 1 |
| Trending tickers count | `finance::trending(Region::US)` count | 1 |
| Trending overlap with portfolio | Count of trending tickers in our portfolio | 1 |
| Day gainers momentum | `finance::screener(Screener::DayGainers, 10)` avg gain | 1 |
| Most actives vol ratio | `finance::screener(Screener::MostActives, 10)` avg vol ratio | 1 |
| Market hours state | `finance::hours(Region::US)` → encoded as pre/open/post/closed | 1 |
| Minutes since market event | Minutes since last open/close transition | 1 |

#### RSS & News Aggregate Signals (12 features)

Derived from `feeds::fetch_all(sources)` across all 30+ RSS sources, processed by MarketReader:

| Feature | Computation | Dims |
|---|---|---|
| Aggregate news sentiment (1h) | Mean MarketReader sentiment across all articles in last hour | 1 |
| Aggregate news sentiment (24h) | Same, 24h window | 1 |
| News volume (1h) | Count of articles in last hour, log-scaled | 1 |
| News volume (24h) | Same, 24h | 1 |
| Fed-related news count (24h) | Articles from FederalReserve source | 1 |
| SEC-related news count (24h) | Articles from SecPressReleases, SecFilings | 1 |
| Crypto news sentiment | CoinDesk + CoinTelegraph average sentiment | 1 |
| Macro news sentiment | FederalReserve + Bea + Ecb average sentiment | 1 |
| Tech sector news sentiment | TechCrunch + VentureBeat sentiment | 1 |
| Business news sentiment | Bloomberg + FT + WSJ sentiment | 1 |
| News surprise (deviation from baseline) | Current hour volume vs EMA baseline | 1 |
| Topic diversity (1h) | Number of distinct topics in recent news | 1 |

**Total market-wide features: 12 + 12 + 12 + 16 + 20 + 12 + 12 = 96 dims.**

---

### 6.4 Event Stream Architecture

Discrete financial events are a first-class input modality. Rather than waiting for their price impact, events are encoded and injected into the model immediately.

**Event types (18 categories):**

| Category | Event Source | Priority |
|---|---|---|
| INSIDER_BUY | `insider_transactions()` | HIGH |
| INSIDER_SELL | `insider_transactions()` | HIGH |
| ANALYST_UPGRADE | `grading_history()` | HIGH |
| ANALYST_DOWNGRADE | `grading_history()` | HIGH |
| EARNINGS_BEAT | `earnings_history()` | CRITICAL |
| EARNINGS_MISS | `earnings_history()` | CRITICAL |
| SEC_FILING_10K | `sec_filings()` / `edgar::submissions()` | MEDIUM |
| SEC_FILING_10Q | Same | MEDIUM |
| SEC_FILING_8K | Same | HIGH (unscheduled = material) |
| SEC_FILING_OTHER | Same | LOW |
| FOMC_DECISION | `feeds::fetch(FederalReserve)` | CRITICAL |
| MACRO_RELEASE | FRED series update detection | HIGH |
| NEWS_HIGH_IMPACT | MarketReader score > threshold | HIGH |
| DIVIDEND_ANNOUNCED | `calendar_events()` | MEDIUM |
| SPLIT_ANNOUNCED | `splits()` change detection | MEDIUM |
| OPTIONS_UNUSUAL_ACTIVITY | Volume anomaly detector | HIGH |
| TRENDING_ENTRY | `finance::trending()` membership gained | MEDIUM |
| RATING_CHANGE | `recommendation_trend()` shift | MEDIUM |

**Event encoding:** Each event is encoded into a D-dimensional token via a learned EventEncoder:

```
EventEncoder:
  event_type_embedding: Embedding(18 → 64)    — one per event type
  magnitude_projection: Linear(1 → 64)         — event magnitude (e.g., surprise %)
  asset_id_embedding: Embedding(100 → 64)      — which asset
  time_decay: exponential decay since event
  
  event_token = Linear(192 → D) applied to concat(type_emb, magnitude_proj, asset_emb)
  event_token *= exp(-λ × minutes_since_event)
```

**Injection:** Event tokens are added to the per-asset token sequence at the minute they arrive. The RA-GDN backbone processes them alongside the regular continuous features. For market-wide events (FOMC, macro releases), the event token is broadcast to all assets.

At any given minute, typically 0–3 event tokens are active (most minutes have no events). This adds negligible compute.

**Parameters:** ~50K (embeddings + projection).

---

### 6.5 Text Corpus Pipeline

MarketReader (Section 7.10) processes text from the following sources, organized by processing priority:

**Tier 1 — Headline-only (processed every 5 minutes):**
RSS feeds polled via `feeds::fetch_all(sources)`:

| Source Group | Feeds | Count |
|---|---|---|
| Central Banks | FederalReserve, Ecb, BankOfEngland | 3 |
| Regulators | SecPressReleases, SecFilings, Cfpb | 3 |
| Major Business News | Bloomberg, FinancialTimes, WsjMarkets, TheEconomist | 4 |
| US Business News | MarketWatch, Cnbc, NytBusiness, Fortune | 4 |
| Financial Analysis | Investing, CalculatedRisk, RitholtzBigPicture, FinancialPost, FtLex | 5 |
| Crypto | CoinDesk, CoinTelegraph | 2 |
| Tech | TechCrunch, VentureBeat, HackerNews, YCombinator | 4 |
| Global | GuardianBusiness, Scmp, NikkeiAsia | 3 |
| Wire Services | BusinessWire | 1 |
| Macro | Bea, OilPrice | 2 |
| **Total** | | **31 feeds** |

Processing: Title + source → MarketReader headline encoder → 4-dim sentiment vector (positive, negative, uncertainty, relevance-to-finance).

**Tier 2 — Summary-level (processed every 15 minutes for high-impact articles):**
For articles with high headline relevance score: FeedEntry.summary (typically 1–3 paragraphs) → MarketReader summary encoder → 32-dim topic+sentiment vector.

**Tier 3 — Deep-level (processed on arrival, queued):**
- Per-ticker news from `ticker.news()` → full article text → 64-dim embedding
- Earnings transcripts from `finance::earnings_transcript(sym, quarter, year)` → 64-dim embedding
- Custom RSS feeds via `FeedSource::Custom(url)` for specialized financial blogs

**Text aggregation per asset:** For each asset i, the final 64-dim text embedding is computed by weighted pooling across all text sources with time decay:

```
text_embedding_i = Σ_j (attention_weight_j × decay_j × embedding_j)

where:
  j iterates over all text items mentioning asset i in the last 24 hours
  attention_weight_j = softmax(Linear(embedding_j) → scalar) over all j
  decay_j = exp(-minutes_since_j / (τ × 60))  
  τ = 4 hours for headlines, 12 hours for summaries, 48 hours for transcripts
```

**Text aggregation for market-wide:** Same process but pooling across all non-asset-specific articles:

```
market_text_embedding = same weighted pooling across general market news
→ contributes to the 12 RSS/News aggregate features in market-wide vector
```

---

### 6.6 Portfolio State Vector (201 dims)

Unchanged from v2.0:

| Feature | Dims |
|---|---|
| Current weight per asset | 100 |
| Unrealized P&L per asset (%) | 100 |
| Total portfolio value (log-scaled) | 1 |

---

### 6.7 Financial Calendar Embedding (20 dims)

Unchanged from v2.0. Encodes minute of session, day of week, month, market-open status, proximity to FOMC, options expiration, etc.

---

## 7. Model Components

### 7.1 Input Projection Layer with Variable Selection

The input projection now handles the expanded feature space:

```
Per-Asset (184 dims → D):
  Step 1: Variable Selection Network
    Per-feature importance: gates = softmax(Linear(regime_emb, 184)) ∈ ℝ^184
    x_selected = gates ⊙ features_184
    
  Step 2: Projection
    x_asset = Linear(184 → 768) → RMSNorm

Market (96 dims → D):
  x_market = Linear(96 → 768) → RMSNorm

Portfolio (201 dims → D):
  x_portfolio = Linear(201 → 768) → RMSNorm

Calendar (20 dims → D):
  x_calendar = Linear(20 → 768)

Event Tokens (when present):
  x_event = EventEncoder(event) → ℝ^768

Per-asset token:
  token_i = x_asset_i + x_market + x_portfolio + x_calendar
  (event tokens are additional tokens, not summed)
```

The **VSN** is critical with 184 input features — it ensures the model can dynamically ignore irrelevant signals. For example, during a crisis regime, the VSN might upweight VaR, correlation, and insider activity while downweighting valuation metrics. The regime embedding conditions the gating, so different regimes produce different feature weightings.

**Parameters:** ~245K (184×768 + 96×768 + 201×768 + 20×768 + VSN).

### 7.2 Regime Detector

Expanded input to use the richer market-wide features:

**Input:** Concatenation of:
- Market-wide features (96 dims) — full yield curve, sector performance, indices, etc.
- Portfolio summary: total value, aggregate P&L, current leverage (3 dims)
- Rolling regime state: EMA of last 60 regime embeddings (64 dims)
- **Total: 163 dims**

**Architecture:**

```
Linear(163 → 256) → SiLU → Dropout(0.1)
Linear(256 → 128) → SiLU → Dropout(0.1)
├── Linear(128 → 64)  → regime_embedding
├── Linear(128 → 4)   → softmax → regime_posteriors
└── regime_change_velocity = ||regime_t − regime_{t-1}||_2
```

The expanded input (full yield curve, world indices, sector performance, credit spreads) gives the regime detector much richer conditioning. The yield curve shape alone distinguishes recession risk, inflationary environments, and loose monetary policy regimes.

**Parameters:** ~55K

### 7.3–7.9: Model Core Components

These are architecturally identical to v2.0 Sections 6.5–6.11, with two adjustments:

1. **Input dimension to backbone:** D=768, the per-asset projection maps from 184 dims. This only affects the input projection weights, not the backbone itself.

2. **Event token processing:** When event tokens are present for an asset at a given timestep, they are processed by the backbone as additional tokens in that asset's sequence. Since the RA-GDN operates in single-step recurrent mode at inference, event tokens are simply additional inputs that update the state S before the regular feature token is processed:

```
If event_token exists for asset_i at time t:
  1. Process event_token through GDN → updates S_i
  2. Process regular feature_token through GDN → updates S_i again
  
Total: 2 GDN steps instead of 1 for that asset that minute.
This happens for ~5% of asset-minutes, adding negligible overhead.
```

**All other component specs (RA-GDN blocks, FMoE, temporal cascade, cross-asset fusion, episodic memory, output heads, auxiliary head, MarketReader) remain as specified in v2.0 Sections 6.5–6.12.**

### 7.10 MarketReader Text Encoder

Expanded to handle the three-tier text processing pipeline:

**Three sub-encoders sharing the same GDN backbone:**

```
Shared components:
  BPE Tokenizer: 4,096 vocab, trained on financial corpus
  Token Embedding: 4,096 × 256
  Position Embedding: 512 × 256
  4 GDN Blocks: D=256, n_heads=4, d_head=32, d_v=64

Tier-specific output heads:
  Headline head:   attention_pool → Linear(256 → 4)   → 4-dim sentiment
  Summary head:    attention_pool → Linear(256 → 32)  → 32-dim topic+sentiment
  Deep head:       attention_pool → Linear(256 → 64)  → 64-dim full embedding

Total parameters: ~3.5M
Quantized (INT8): ~3.5 MB
```

**Headline processing** (the most frequent): Only processes the title string (~10–20 tokens). Latency: < 0.5 ms on Pi 5. Can process ~200 headlines per second.

**Summary processing**: Processes up to 128 tokens from the article summary. Latency: ~1 ms.

**Deep processing**: Processes up to 512 tokens from full articles or earnings transcripts. Latency: ~2 ms. For earnings transcripts that exceed 512 tokens, a sliding window with max-pooling is used.

**Training corpus for MarketReader:**
1. Financial PhraseBank (sentiment classification)
2. SEC filings sentiment (EDGAR corpus)
3. Earnings call transcripts (Quartr/finance-query corpus) with post-call price direction labels
4. Financial news headlines with next-day return labels
5. Contrastive learning: articles about the same event → similar embeddings

---

## 8. Continuous Learning System

The three-tier system is unchanged from v2.0, with one addition:

### 8.1–8.3: Tiers 1–3

Identical to v2.0 Sections 7.1–7.3.

### 8.4 Adaptation Feedback Loop (New)

The continuous learning system produces measurable adaptation metrics that feed back into the system itself:

**Adaptation health dashboard (computed continuously):**

| Metric | What It Measures | Healthy Range |
|---|---|---|
| Surprise EMA | Baseline prediction error | Stable, slowly varying |
| Surprise spike frequency | How often surprises exceed 3σ | < 5% of minutes |
| Regime posterior entropy | How uncertain the regime detector is | Low = clear regime |
| State norm trajectory | GDN hidden state magnitude over time | Bounded, no monotonic growth |
| Norm parameter drift | How far Tier 2 params have moved from checkpoint | < 20% relative change |
| LoRA adapter magnitude | How large the Tier 3 adjustments are | < 10% of base weight norm |
| Auxiliary prediction R² | How well the model predicts next-step returns/vol | Declining → signal regime shift |
| Feature importance entropy | How evenly VSN distributes attention | High = confusion, low = focused |

**Adaptation triggers:**

1. If `auxiliary_R²` drops below 0.01 for 2+ hours → increase Tier 1 sensitivity (raise γ)
2. If `surprise_spike_frequency` > 15% for 1+ hours → trigger emergency Tier 3 consolidation
3. If `state_norm` exceeds 2× training mean → emergency RMSNorm
4. If `regime_posterior_entropy` stays high for 4+ hours → switch MoE to equal-weight (all-expert) mode until regime clarifies
5. If `norm_parameter_drift` > 20% → reduce Tier 2 learning rate

These rules are implemented as simple threshold checks in the data pipeline, not in the model itself. They provide a safety net for the adaptation system.

---

## 9. Quantization Strategy

Unchanged from v2.0. Mixed precision: ternary weights, FP16 hidden states, INT8 activations. Periodic RMSNorm on state. ARM NEON ternary kernels via TAB framework.

---

## 10. Training Pipeline

### 10.1 Data Sources & Preprocessing

**Expanded training data corpus:**

| Data Type | Source | Granularity | Historical Depth |
|---|---|---|---|
| Minute OHLCV (equities, ETFs) | Polygon.io / FirstRate Data | 1-minute | 5 years |
| Minute OHLCV (crypto) | Binance / CoinGecko historical | 1-minute | 5 years |
| Full technical indicators (42) | Computed via finance-query | 1-minute | 5 years |
| Options chains | CBOE / OptionMetrics | Daily snapshots | 5 years |
| Fundamentals (XBRL) | `edgar::company_facts()` / finance-query | Quarterly | 5 years |
| Earnings actuals vs estimates | finance-query `earnings_history()` | Quarterly | 5 years |
| Earnings estimate revisions | finance-query `earnings_trend()` | Monthly | 3 years |
| Analyst ratings history | finance-query `recommendation_trend()` | Monthly | 3 years |
| Analyst upgrades/downgrades | finance-query `grading_history()` | Event-driven | 5 years |
| Insider transactions | finance-query `insider_transactions()` | Event-driven | 5 years |
| Institutional ownership | finance-query `institution_ownership()` | Quarterly | 5 years |
| News headlines + summaries | finance-query RSS historical / GDELT | Event-driven | 3 years |
| Earnings transcripts | finance-query `earnings_transcript()` | Quarterly | 3 years |
| SEC filings metadata | `edgar::submissions()` | Event-driven | 5 years |
| Treasury yield curve (all mats) | `fred::treasury_yields()` | Daily | 5 years |
| FRED macro (12 series) | `fred::series()` | Various | 5 years |
| Fear & Greed index | Historical from alternative.me | Daily | 5 years |
| VIX + VIX3M | Ticker charts | 1-minute | 5 years |
| World indices (9 indices) | `finance::indices()` historical | Daily | 5 years |
| Sector performance (11 GICS) | `finance::sector()` historical | Daily | 5 years |
| Crypto market (BTC, ETH) | `crypto::coin()` historical | Hourly | 5 years |
| Screener memberships | `finance::screener()` historical | Daily | 2 years |
| Trending tickers | `finance::trending()` historical | Daily | 2 years |
| Risk analytics | `ticker.risk()` computed from price data | Daily | 5 years |

**Data scale:**
- ~490,000 minutes of US market data per equity asset
- ~2,600,000 minutes of crypto data
- ~100,000 discrete events (insider trades, filings, rating changes, etc.) per 100 assets over 5 years
- ~500,000 news headlines, ~50,000 article summaries, ~4,000 earnings transcripts
- Uncompressed: ~50 GB → stored as memory-mapped FP16 tensors

**Event reconstruction:** For historical training, events are reconstructed from the diff of periodic API snapshots. For example, insider transactions are derived by diffing consecutive `insider_transactions()` calls and assigning each transaction to the minute of its filing date.

### 10.2–10.5: Training Phases

**Phase 1 — Self-supervised pre-training:** Uses 184-dim per-asset features and 96-dim market features (280 dims total per shard row). Training tasks unchanged (next-step return, volatility, correlation prediction). Estimated: ~100 H100-hours.

**Phase 2 — Portfolio optimization:** Unchanged (DSR + return loss + volatility auxiliary). Estimated: ~80 H100-hours.

**Phase 3 — Meta-learning for plasticity:** Regime transitions now use the richer event stream for more realistic simulation. For example, a simulated "Fed surprise" episode includes the FOMC_DECISION event token, the associated news headlines, and the yield curve shift. Estimated: ~50 H100-hours.

**Phase 4 — QAT:** Unchanged. Estimated: ~20 H100-hours.

**Total: ~260 H100-hours** (within 500-hour budget, with room for experimentation).

---

## 11. Two-Head Strategy (Meridian-R / Meridian-S)

Unchanged from v2.0. Shared backbone, two heads with different loss functions. One active at inference time, or blended with regime-dependent weighting.

---

## 12. Historical Backtesting Framework

Meridian integrates directly with finance-query's backtesting engine for historical validation.

### 12.1 Meridian as a Backtesting Strategy

A custom `Strategy` implementation wraps the trained Meridian model:

```rust
// Conceptual — the actual model runs in Python/C, bridged via FFI or shared memory
struct MeridianStrategy {
    model: MeridianInference,  // Quantized model instance
    head: HeadType,             // R or S
}

impl Strategy for MeridianStrategy {
    fn on_bar(&mut self, context: &StrategyContext) -> Vec<Signal> {
        // 1. Build feature vector from context (OHLCV + indicators available in backtest)
        let features = self.build_features(context);
        
        // 2. Run model forward pass
        let weights = self.model.forward(features, self.head);
        
        // 3. Convert weight changes to signals
        let current_weights = context.current_positions();
        let mut signals = vec![];
        
        for (i, (target, current)) in weights.iter().zip(current_weights).enumerate() {
            let delta = target - current;
            if delta.abs() > MIN_TRADE_THRESHOLD {
                if delta > 0.0 {
                    signals.push(Signal::buy(context.symbol(i), delta));
                } else {
                    signals.push(Signal::sell(context.symbol(i), delta.abs()));
                }
            }
        }
        signals
    }
}
```

**Running the backtest:**

```rust
// Using finance-query's portfolio backtesting engine
let config = PortfolioConfig::new(
    BacktestConfig::default()
        .initial_capital(1_000_000.0)
        .commission_pct(0.001)     // 10 bps
        .slippage_pct(0.0005)      // 5 bps
        .allow_short(true)
        .risk_free_rate(0.05)
        .reinvest_dividends(true)
        .max_positions(Some(100))
        .bars_per_year(252.0 * 390.0)  // minute bars
)
    .max_total_positions(100)
    .rebalance(RebalanceMode::CustomWeights(meridian_weights));

let symbol_data: Vec<SymbolData> = symbols.iter()
    .map(|s| SymbolData::new(s, candles[s].clone()).with_dividends(dividends[s].clone()))
    .collect();

let result = PortfolioEngine::new(config)
    .run(&symbol_data, |sym| MeridianStrategy::new(model.clone(), HeadType::Sharpe))?;
```

### 12.2 Benchmark Comparisons

Using finance-query's `BacktestComparison`:

```rust
// Run benchmarks
let spy_result = BacktestEngine::run("SPY", &spy_candles, &BuyAndHold::new())?;
let equal_weight = PortfolioEngine::new(equal_config).run(&symbol_data, |_| BuyAndHold::new())?;
let sma_cross = PortfolioEngine::new(sma_config).run(&symbol_data, |_| SmaCrossover::new(10, 50))?;

// Compare
let comparison = BacktestComparison::new()
    .add("Meridian-R", meridian_r_result)
    .add("Meridian-S", meridian_s_result)
    .add("SPY Buy & Hold", spy_result)
    .add("Equal Weight", equal_weight)
    .add("SMA Crossover", sma_cross)
    .ranked_by(OptimizeMetric::SharpeRatio);

// Output: ranked table with all PerformanceMetrics fields
// including: sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor, etc.
```

**Required benchmarks:**

| Benchmark | Strategy | Purpose |
|---|---|---|
| SPY Buy & Hold | Passive index | "Can Meridian beat the market?" |
| Equal Weight Rebalance | Monthly equal weight across universe | "Does active allocation add value?" |
| Risk Parity | Inverse-volatility weighted | "Does Meridian's risk management add value?" |
| SMA Crossover | `SmaCrossover::new(50, 200)` per asset | "Does ML beat simple technical?" |
| RSI Reversal | `RsiReversal::new(14, 30.0, 70.0)` | "Does ML beat momentum?" |
| Ensemble | `EnsembleStrategy` (SMA + RSI + Bollinger) | "Does ML beat classical ensemble?" |

### 12.3 Walk-Forward Validation

The most rigorous test — simulates realistic deployment with periodic refit:

```rust
// Using finance-query's walk-forward framework
// In-sample: 1 year, Out-of-sample: 3 months, rolling forward
let wf_config = WalkForwardConfig::new(
    GridSearch::new()
        .param("head", ParamRange::int_range(0, 1, 1))  // R vs S
        .param("leverage", ParamRange::float_range(1.0, 2.0, 0.5))
        .optimize_for(OptimizeMetric::SharpeRatio),
    backtest_config
)
    .in_sample_bars(252 * 390)   // 1 year of minute bars
    .out_of_sample_bars(63 * 390) // 3 months OOS
    .run(&symbols, &all_candles, |params| {
        MeridianStrategy::new(model.clone(), params.head, params.leverage)
    })?;

// Key outputs:
// wf_report.aggregate_metrics — OOS-only performance (the real test)
// wf_report.consistency_ratio — fraction of OOS windows that are profitable
// wf_report.windows — per-window breakdown
```

**Consistency ratio > 0.6** is the minimum threshold for deployment consideration. Professional quant funds target > 0.7.

### 12.4 Monte Carlo Confidence Intervals

```rust
let mc = MonteCarloConfig::new()
    .seed(42)
    .num_simulations(10_000)
    .method(MonteCarloMethod::BlockBootstrap)  // preserves autocorrelation
    .run(&meridian_result);

// Outputs:
// mc.total_return: p5, p25, p50, p75, p95
// mc.max_drawdown: p5, p25, p50, p75, p95
// mc.sharpe_ratio: p5, p25, p50, p75, p95

// Deployment criteria:
// mc.total_return.p5 > 0  (95% CI positive)
// mc.sharpe_ratio.p25 > 0.5  (75% CI above 0.5 Sharpe)
// mc.max_drawdown.p95 < 0.30 (95% CI max DD under 30%)
```

### 12.5 Regime-Segmented Analysis

Using backtest result analysis methods to understand performance across market conditions:

```rust
// Time-segmented analysis
let yearly = meridian_result.by_year();      // Per-year returns
let monthly = meridian_result.by_month();    // Per-month returns
let daily = meridian_result.by_day_of_week(); // Day-of-week patterns

// Drawdown analysis
let dd_series = meridian_result.drawdown_series();
let rolling_sharpe = meridian_result.rolling_sharpe(252 * 390); // 1-year rolling

// Regime-segmented (custom analysis):
// Tag trades by regime during backtest, then:
let crisis_trades = meridian_result.trades_by_tag("crisis");
let crisis_metrics = meridian_result.metrics_by_tag("crisis");
let trending_trades = meridian_result.trades_by_tag("trending");
let trending_metrics = meridian_result.metrics_by_tag("trending");
// etc. for all 4 regimes
```

**Key questions the backtest must answer:**
1. Does Meridian outperform passive strategies after transaction costs?
2. Does Meridian's Sharpe ratio exceed 1.0 in walk-forward OOS testing?
3. Does Meridian's max drawdown stay under 25% with 95% MC confidence?
4. Does Meridian's adaptation (Tiers 1–3) measurably improve OOS performance vs. a frozen model?
5. Is Meridian's performance consistent across regimes, or does it concentrate in one?
6. Does the event stream (insider trades, analyst changes) add alpha beyond price-only features?

---

## 13. Paper Trading System

After backtesting validation, Meridian runs in paper trading mode to test real-time performance before live capital deployment.

### 13.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PAPER TRADING ENGINE                  │
│                                                         │
│  ┌───────────┐   ┌─────────────┐   ┌────────────────┐  │
│  │ DataBus   │──▶│  Meridian   │──▶│  Paper Broker  │  │
│  │(live data)│   │  (inference)│   │  (simulated)   │  │
│  └───────────┘   └─────────────┘   └────────────────┘  │
│       │                │                    │           │
│       │          ┌─────▼─────┐        ┌────▼────────┐  │
│       │          │ Adaptation│        │  Portfolio   │  │
│       │          │ System    │        │  Tracker     │  │
│       │          │(Tiers 1-3)│        │  (P&L, risk) │  │
│       │          └───────────┘        └─────────────┘  │
│       │                                     │           │
│       │               ┌─────────────────────▼────────┐  │
│       └──────────────▶│  Performance Logger          │  │
│                       │  (SQLite on NVMe SSD)        │  │
│                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Paper Broker:** Simulates trade execution with realistic friction:

```
For each minute:
  1. Meridian produces target weights w_target
  2. Compare with current paper portfolio weights w_current
  3. For each asset where |w_target - w_current| > MIN_TRADE_THRESHOLD (0.5%):
     a. Simulate execution at next minute's open price
     b. Apply slippage: 5 bps (configurable)
     c. Apply commission: 10 bps (configurable)
     d. Update paper portfolio
  4. Log: timestamp, all weights, all prices, portfolio value, all model internals
```

**Constraint:** Rebalancing at most once per minute. In practice, the model's turnover penalty in the loss function means most minutes produce very small weight changes that fall below the trade threshold.

### 13.2 Performance Tracking

The paper trading engine logs everything needed for analysis:

**Per-minute log (SQLite):**

| Field | Type | Notes |
|---|---|---|
| timestamp | u64 | Epoch minutes |
| target_weights | f32[100] | Model output |
| executed_weights | f32[100] | After trade simulation |
| prices | f32[100] | Execution prices |
| portfolio_value | f64 | NAV |
| daily_return | f64 | |
| regime_posteriors | f32[4] | |
| aggregate_surprise | f32 | |
| active_head | u8 | R=0, S=1 |
| event_count | u16 | Events this minute |
| inference_latency_us | u32 | Microseconds |

**Daily summary (computed from per-minute log):**

| Metric | Computation |
|---|---|
| Daily return | `(NAV_close - NAV_open) / NAV_open` |
| Daily Sharpe | Return / vol (annualized from daily) |
| Max intraday drawdown | Peak-to-trough within day |
| Turnover | Sum of absolute weight changes |
| Transaction costs | Sum of slippage + commissions |
| Number of trades | Count of threshold-exceeding rebalances |
| Dominant regime | Mode of regime posteriors over the day |
| Surprise frequency | Fraction of minutes with surprise > 2σ |
| Model confidence | Average output head confidence scores |
| Feature importance top-5 | Top VSN gate values |
| Active experts | MoE expert utilization distribution |

**Rolling analytics (computed from daily summaries):**

| Window | Metrics |
|---|---|
| 5-day | Return, Sharpe, max DD, turnover |
| 20-day | Return, Sharpe, Sortino, max DD, beta to SPY |
| 60-day | Return, Sharpe, Sortino, Calmar, max DD, information ratio vs SPY |
| Since inception | All PerformanceMetrics fields from finance-query |

### 13.3 Adaptation Monitoring

During paper trading, the adaptation system's health is tracked in detail:

**Tier 1 (Surprise-gated dynamics):**
- Plot: surprise signal over time, with regime labels
- Alert: if surprise EMA trends upward for 5+ consecutive days → market structure may be changing in a way the model can't handle with Tier 1 alone

**Tier 2 (Normalization adaptation):**
- Plot: RMSNorm parameter drift from checkpoint over time
- Alert: if drift exceeds 20% → parameters approaching instability

**Tier 3 (EWC + LoRA consolidation):**
- Plot: consolidation loss trajectory each night
- Plot: Fisher diagonal magnitude distribution (which parameters are most constrained?)
- Alert: if consolidation loss fails to decrease → data may be too noisy or regimes too mixed

**State health:**
- Plot: GDN hidden state norms across layers over time
- Plot: per-asset state norms (identify assets with unusual state dynamics)
- Alert: monotonically increasing state norms → drift problem (see Risk #1)

**Comparison to frozen model:**
- Run a second Meridian instance with all adaptation disabled
- Track the performance gap: `Sharpe_adaptive - Sharpe_frozen`
- This directly measures the value of the continuous learning system
- Target: adaptive Sharpe > frozen Sharpe by at least 0.1 after 3 months

### 13.4 Graduation Criteria (Paper → Live)

Before deploying real capital, the paper trading system must pass all of the following:

| Criterion | Threshold | Measurement Period |
|---|---|---|
| Cumulative return | > 0% (profitable) | 6 months minimum |
| Annualized Sharpe ratio | > 1.0 | 6 months |
| Max drawdown | < 20% | Since inception |
| Monte Carlo 5th percentile return | > 0% | 10,000 sims on paper data |
| Walk-forward consistency ratio | > 0.6 | Rolling 3-month OOS windows |
| Adaptation improvement | Adaptive Sharpe > Frozen Sharpe | 3 months |
| State stability | No monotonic state growth | 6 months |
| Uptime | > 99.5% (< 1.8 days downtime) | 6 months |
| Inference latency p99 | < 100 ms | Since inception |
| Data pipeline reliability | < 1% missed minutes | 6 months |

**Only after ALL criteria are met should the system be considered for live trading.** Even then, start with minimal capital (e.g., $10K) and scale gradually.

---

## 14. Inference Runtime on Pi 5

Expanded from v2.0 to reflect the richer data pipeline:

### Per-Minute Pipeline

```
Time 0ms: New minute tick (PriceStream)
  │
  ├─ [Core 3, ASYNC] Data pipeline:
  │   ├── Fetch prices for 100 tickers (PriceStream — already subscribed)
  │   ├── Compute 42 technical indicators per asset
  │   ├── Process any new RSS headlines (batch, ~20 articles/5min avg)
  │   ├── Check for new events (insider tx, filings, ratings)
  │   ├── Update rolling statistics (z-scores, EMAs, correlations)
  │   ├── Assemble 100 × 184 per-asset features
  │   ├── Assemble 96 market-wide features
  │   ├── Assemble 201 portfolio state features
  │   └── Duration: 200–500ms (indicator computation + normalization)
  │
  ├─ [Cores 0-2] Model inference:
  │   ├── Input projection + VSN gating: 0.5ms
  │   ├── Regime detection: 0.1ms
  │   ├── RA-GDN backbone (8 blocks × 100 assets): 4ms
  │   ├── FMoE (3 blocks × 100 assets): 1ms
  │   ├── Event token processing (~5 assets with events): 0.3ms
  │   ├── Temporal cascade (when active): 0.2ms
  │   ├── Cross-asset fusion (110 tokens, 2 layers): 0.5ms
  │   ├── FAISS episodic memory: 0.5ms
  │   ├── Output head: 0.05ms
  │   └── TOTAL: ~7ms
  │
  ├─ [Core 2, ASYNC] Post-inference:
  │   ├── Observe actual returns from t-1
  │   ├── Compute surprise signal
  │   ├── Tier 2 norm adaptation (every 60 min): 0.1ms
  │   ├── Log to SQLite: 0.5ms
  │   └── Paper broker simulation: 0.1ms
  │
  └─ Budget: 60,000ms available. Total used: ~510ms. CPU util: ~0.85%
```

---

## 15. Memory Budget

### Static Allocation

| Component | Size | Notes |
|---|---|---|
| Ternary weights (backbone + MoE + cascade) | 18.5 MB | 2-bit packed + FP16 scales |
| INT8 weights (regime detector + MarketReader) | 3.8 MB | |
| FP16 weights (output heads + projections + VSN) | 1.5 MB | Expanded input projections |
| GDN hidden states (100 assets × 8 layers) | 75.5 MB | FP16 |
| Temporal cascade states | 16.4 MB | |
| Conv1d caches | 3.9 MB | |
| Episodic memory buffer (50K entries) | 13.0 MB | |
| FAISS index | 6.4 MB | |
| Rolling statistics + feature caches | 8.0 MB | Expanded: 42 indicators × 100 assets |
| Event state buffer | 2.0 MB | Per-asset event history |
| Text embedding cache (100 assets × 64 dims × recent) | 1.0 MB | |
| HMM parameters | 0.1 MB | |
| **Total static** | **~150 MB** | |

### Dynamic Allocation

| Component | Size |
|---|---|
| Activation buffers | ~3.0 MB |
| Cross-asset attention KV | 0.2 MB |
| Feature assembly buffer | 0.19 MB (100 × 184 × 4 bytes) |
| RSS/news processing queue | 2.0 MB |
| **Total dynamic** | **~5.5 MB** |

### Periodic Allocation (Tier 3)

| Component | Size |
|---|---|
| Fisher diagonal | 60 MB |
| Parameter checkpoint | 23 MB |
| Replay buffer | 15 MB (expanded features) |
| LoRA adapters + gradients | 1.5 MB |
| **Total Tier 3** | **~100 MB** |

### SQLite Logging

| Component | Size per day | Notes |
|---|---|---|
| Per-minute log | ~20 MB/day | 390 rows × ~50 KB/row (compressed) |
| Daily summaries | ~10 KB/day | |
| Event log | ~1 MB/day | |
| NVMe budget | 256 GB | ~3.5 years of logs |

### Grand Total

| Scenario | Memory |
|---|---|
| Normal inference | ~156 MB |
| During Tier 3 consolidation | ~256 MB |
| OS + Rust data pipeline + Python bridge | ~500 MB |
| **Peak usage** | **~756 MB** |
| **Available (8GB Pi 5)** | **8,192 MB** |
| **Headroom** | **~7,436 MB (91%)** |

---

## 16. Development Roadmap

### Phase 0: Data Infrastructure (Weeks 1–4)

- [ ] Build DataBus in Rust using finance-query as library dependency
- [ ] Implement all 42 indicator computations with rolling z-score normalization
- [ ] Implement RSS feed polling across all 31 sources with deduplication
- [ ] Implement event detection (diff-based) for insider tx, ratings, filings
- [ ] Build historical data download pipeline for training corpus
- [ ] Implement SQLite logging schema
- [ ] Build feature assembly pipeline (184 per-asset + 96 market-wide)
- [ ] Verify all finance-query API integrations end-to-end
- [ ] Train BPE tokenizer on financial text corpus
- [ ] Pre-fit 4-state HMM on historical regime data

### Phase 1: Core Model (Weeks 5–10)

- [ ] Implement RA-GDN block in PyTorch (extending FLA Gated DeltaNet)
- [ ] Implement regime conditioning (α bias, β scaling)
- [ ] Implement EventEncoder
- [ ] Implement FMoE with external regime routing
- [ ] Implement temporal cascade
- [ ] Implement cross-asset attention with sector bias
- [ ] Implement VSN with regime-conditioned gating
- [ ] Train MarketReader text encoder (3-tier)
- [ ] Run Phase 1 self-supervised pre-training
- [ ] Validate prediction accuracy on held-out data

### Phase 2: Portfolio Training (Weeks 11–14)

- [ ] Implement portfolio output heads with leverage normalization
- [ ] Implement DSR loss, return loss, volatility auxiliary loss
- [ ] Run Phase 2a (return head) and Phase 2b (Sharpe head)
- [ ] Build MeridianStrategy wrapper for finance-query backtesting
- [ ] Run backtests vs all benchmarks (Section 12.2)
- [ ] Run walk-forward validation (Section 12.3)
- [ ] Run Monte Carlo analysis (Section 12.4)
- [ ] Run regime-segmented analysis (Section 12.5)
- [ ] Iterate: if backtest fails criteria, diagnose and retrain

### Phase 3: Adaptation & QAT (Weeks 15–20)

- [ ] Implement 3-tier continuous learning system
- [ ] Run Phase 3 meta-learning training
- [ ] Run Phase 4 QAT
- [ ] Validate quantized model performance within tolerance
- [ ] Ablation study: adaptation value (adaptive vs frozen backtest comparison)
- [ ] Ablation study: event stream value (with vs without events)
- [ ] Ablation study: text value (with vs without MarketReader)

### Phase 4: Deployment (Weeks 21–26)

- [ ] Implement NEON ternary inference kernel (bitnet.cpp adapted)
- [ ] Implement FAISS episodic memory on ARM
- [ ] Port full inference pipeline to Pi 5
- [ ] Run 10M+ step state drift test
- [ ] Implement paper trading engine
- [ ] Deploy to Pi 5 with monitoring dashboard
- [ ] 72-hour continuous operation stress test
- [ ] Begin 6-month paper trading period

### Phase 5: Paper Trading & Iteration (Weeks 27–52)

- [ ] Monitor all graduation criteria (Section 13.4)
- [ ] Weekly review of adaptation health metrics
- [ ] Monthly comparison to benchmarks
- [ ] Iterate on model if needed (retrain specific phases)
- [ ] After 6 months: evaluate graduation criteria
- [ ] If passed: proceed to minimal live trading

---

## 17. Risk Analysis & Mitigations

All risks from v2.0 (state drift, catastrophic forgetting, regime miscalibration, overfitting, data pipeline latency, insufficient capacity) remain with their mitigations.

**Additional risks from expanded data ingestion:**

### Risk #7: Feature Bloat / Noise Injection

**Risk:** With 184 per-asset features, some features may be pure noise that degrades the signal-to-noise ratio.

**Mitigation:**
1. The VSN dynamically gates features to zero — noisy features will receive near-zero gate weights
2. During training, monitor VSN gate distributions. Features with consistently near-zero gates across all regimes can be removed to simplify the pipeline.
3. L1 regularization on VSN gate logits encourages sparsity
4. Ablation study: train with reduced feature sets and compare. If 184 dims doesn't improve over a smaller set, prune accordingly.

### Risk #8: RSS Feed Reliability

**Risk:** Third-party RSS feeds may go down, change format, or produce duplicate/stale content.

**Mitigation:**
1. finance-query's `feeds::fetch_all()` already deduplicates
2. Staleness detection: if a feed hasn't produced new content in 24h, flag it
3. Graceful degradation: if all RSS fails, the 12 market-wide news features default to their EMA baselines
4. Multiple overlapping sources reduce single-source dependency

### Risk #9: Event Misclassification

**Risk:** Events detected by diffing API snapshots may have incorrect timestamps or magnitudes.

**Mitigation:**
1. Event timestamps are coarsened to the nearest minute, reducing precision requirements
2. Event magnitudes are z-scored against historical distribution, limiting outlier impact
3. The model's event tokens have learned time decay, so even if an event is slightly mistimed, its impact fades naturally
4. Worst case: a misclassified event looks like a noise token and the backbone learns to downweight it

### Risk #10: Options Data Sparsity

**Risk:** Not all 100 assets have liquid options markets. Illiquid options produce unreliable IV and flow signals.

**Mitigation:**
1. The learned 12-dim default embedding per asset class handles assets without any options
2. For assets with illiquid options: apply a confidence weight based on bid-ask spread and volume. If confidence is below threshold, replace with the default embedding.
3. The model's VSN can learn to gate options features to zero for assets where they're unreliable

---

## Appendix A — Parameter Count Breakdown

| Component | Parameters | Active Per Step | Quantization | Size |
|---|---|---|---|---|
| Input projections + VSN (expanded) | 380K | 380K | Ternary | 95 KB |
| EventEncoder | 50K | 50K | FP16 | 100 KB |
| Regime detector (expanded) | 55K | 55K | INT8 | 55 KB |
| RA-GDN blocks (8) | 17.4M | 17.4M | Ternary | 4.3 MB |
| FMoE blocks (3 × 4 experts) | 42.5M | 21.3M | Ternary | 10.6 MB |
| Temporal cascade (3 GDN + projections) | 4.2M | 4.2M | Ternary | 1.1 MB |
| Cross-asset attention (2 layers) | 11.8M | 11.8M | Ternary + FP16 | 3.0 MB |
| Episodic memory projections | 200K | 200K | FP16 | 400 KB |
| Portfolio heads (2) | 394K | 197K | FP16 | 788 KB |
| Auxiliary prediction head | 100K | 100K | FP16 | 200 KB |
| Sector bias matrix | 121 | 121 | FP32 | 484 B |
| MarketReader text encoder | 3.5M | 3.5M (async) | INT8 | 3.5 MB |
| RMSNorm parameters | ~25K | ~25K | FP32 | 100 KB |
| **Total** | **~80.6M** | **~35.2M** | **Mixed** | **~24.2 MB** |

---

## Appendix B — Complete finance-query Integration Map

### Every Minute (via PriceStream + compute)

| API / Computation | Features Generated | Count |
|---|---|---|
| `PriceStream::subscribe(&symbols)` | OHLCV per asset | 500 |
| 42 indicator functions | Technical indicators per asset | 4,200 |
| Rolling return computation | Multi-horizon returns per asset | 500 |
| Portfolio state computation | Weights, P&L, NAV | 201 |
| Calendar computation | Temporal embedding | 20 |
| Cross-asset correlations | Pairwise rolling corr | 1 |

### Every 5 Minutes

| API Call | Purpose |
|---|---|
| `feeds::fetch_all(ALL_31_SOURCES)` | RSS headlines → MarketReader → sentiment features |
| `feeds::fetch(FederalReserve)` | Fed-specific news detection |
| `feeds::fetch(SecPressReleases)` | SEC regulatory news |
| `feeds::fetch(SecFilings(form))` | New filing detection |

### Every 15 Minutes

| API Call | Purpose |
|---|---|
| `ticker.options(nearest_expiry)` × 100 | Options chain → 12 features per asset |
| Via `Tickers::options(date)` batch | Batch options data |

### Every Hour

| API Call | Purpose |
|---|---|
| `finance::indices(Americas)` | US index performance |
| `finance::indices(Europe)` | European index performance |
| `finance::indices(AsiaPacific)` | Asian index performance |
| `crypto::coins("usd", 10)` | Top crypto market data |
| `finance::trending(Region::US)` | Trending ticker detection |
| `finance::market_summary(Region::US)` | Breadth, advance/decline |

### Every 6 Hours

| API Call | Purpose |
|---|---|
| `ticker.insider_transactions()` × 100 | Insider buy/sell detection |
| `ticker.sec_filings()` × 100 | New SEC filing detection |
| `ticker.news()` × 100 | Per-ticker news (deep processing) |

### Daily

| API Call | Purpose |
|---|---|
| `fred::treasury_yields(year)` | Full yield curve (12 maturities) |
| `finance::fear_and_greed()` | Fear & Greed index |
| `finance::sector(Sector::*)` × 11 | All sector performance |
| `ticker.grading_history()` × 100 | Analyst upgrade/downgrade detection |
| `ticker.recommendation_trend()` × 100 | Analyst consensus snapshot |
| `ticker.risk(1d, 1y, "SPY")` × 100 | Risk analytics (VaR, Sharpe, beta) |
| `ticker.dividend_analytics(1y)` × 100 | Dividend trend analysis |
| `ticker.recommendations(5)` × 100 | Similar stock recommendations |
| `finance::screener(screen, 25)` × 15 | All 15 screeners → membership features |
| `finance::custom_screener(query)` | Custom momentum/value screens |

### Weekly

| API Call | Purpose |
|---|---|
| `fred::series("ICSA")` | Initial jobless claims |
| `fred::series("CCSA")` | Continuing claims |
| `ticker.calendar_events()` × 100 | Upcoming earnings/dividend dates |

### Monthly / On Release

| API Call | Purpose |
|---|---|
| `fred::series("FEDFUNDS")` | Fed funds rate |
| `fred::series("UNRATE")` | Unemployment |
| `fred::series("CPIAUCSL")` | CPI |
| `fred::series("CPILFESL")` | Core CPI |
| `fred::series("RSXFS")` | Retail sales |
| `fred::series("INDPRO")` | Industrial production |
| `fred::series("HOUST")` | Housing starts |
| `fred::series("UMCSENT")` | Consumer confidence |
| `fred::series("USSLIND")` | Leading economic index |

### Quarterly

| API Call | Purpose |
|---|---|
| `ticker.financial_data()` × 100 | Revenue, margins, EPS, EBITDA |
| `ticker.summary_detail()` × 100 | PE, yield, market cap |
| `ticker.key_stats()` × 100 | EV, beta, shares outstanding |
| `ticker.earnings_history()` × 100 | Actual vs estimate (surprise) |
| `ticker.earnings_trend()` × 100 | Estimate revisions |
| `ticker.institution_ownership()` × 100 | Institutional holder changes |
| `ticker.fund_ownership()` × 100 | Fund holder changes |
| `ticker.major_holders()` × 100 | Ownership breakdown |
| `ticker.share_purchase_activity()` × 100 | Net insider purchase activity |
| `finance::earnings_transcript(sym, q, y)` × 100 | Full transcript → MarketReader |
| `ticker.financials(Income, Quarterly)` × 100 | Income statement |
| `ticker.financials(Balance, Quarterly)` × 100 | Balance sheet |
| `ticker.financials(CashFlow, Quarterly)` × 100 | Cash flow statement |
| `edgar::company_facts(cik)` × 100 | XBRL data (us-gaap facts) |

### On-Demand / Once

| API Call | Purpose |
|---|---|
| `ticker.asset_profile()` × 100 | Sector, industry, description (static) |
| `ticker.quote_type()` × 100 | Exchange, quote type metadata |
| `ticker.summary_profile()` × 100 | Company address, website |
| `finance::search(query, options)` | Ticker lookup/discovery |
| `finance::lookup(query, options)` | Asset type filtering |
| `finance::exchanges()` | Exchange list |
| `finance::currencies()` | Currency reference data |
| `finance::hours(Region::US)` | Market hours schedule |
| `edgar::resolve_cik(symbol)` × 100 | CIK mapping (cached) |
| `edgar::submissions(cik)` × 100 | Filing history |
| `edgar::search(query, ...)` | Filing search |

### Backtesting Integration

| API Call | Purpose |
|---|---|
| `ticker.chart(interval, range)` | Historical candle data |
| `ticker.dividends(range)` | Dividend history for adjusted returns |
| `ticker.splits(range)` | Split-adjusted prices |
| `Tickers::backtest(interval, range, config, factory)` | Portfolio-level backtesting |
| `BacktestEngine::run_with_benchmark(...)` | Alpha/beta vs SPY |
| `WalkForwardConfig::new(...).run(...)` | Walk-forward validation |
| `MonteCarloConfig::new().run(&result)` | Confidence intervals |
| `BacktestComparison::new().add(...)` | Multi-strategy comparison |
| `GridSearch::new().run(...)` | Hyperparameter optimization |
| `BayesianSearch::new().run(...)` | Bayesian hyperparameter search |

---

## Appendix C — Hyperparameter Reference

### Model Architecture

| Parameter | Value | Notes |
|---|---|---|
| D (model dim) | 768 | Main backbone hidden size |
| Per-asset input dim | 184 | 120 continuous + 64 text |
| Market-wide input dim | 96 | Expanded from 24 |
| RA-GDN blocks | 8 | Core temporal layers |
| FMoE blocks | 3 | After blocks 2, 5, 8 |
| GDN heads / d_head / d_v | 4 / 64 / 192 | |
| MoE experts / active | 4 / 2 | Regime-routed |
| Regime embedding dim | 64 | |
| Regime classes | 4 | Low-vol, high-vol, mean-rev, crisis |
| Event types | 18 | Discrete event categories |
| Event embedding dim | 64 | Per component (type, magnitude, asset) |
| Temporal cascade dims | 512, 512, 384 | Hourly, daily, weekly |
| Cross-asset attention heads/layers | 8 / 2 | |
| Episodic memory slots | 50,000 | |
| MarketReader dim / layers | 256 / 4 | |
| MarketReader vocab | 4,096 | BPE |
| Text embedding output | 64 | Per-asset |
| RSS feed sources | 31 | All finance-query FeedSource variants |
| FRED macro series tracked | 12 | |
| Technical indicators used | 42 | All finance-query indicators |

### Training (same as v2.0)

### Online Adaptation (same as v2.0, plus adaptation feedback loop thresholds)

### Backtesting

| Parameter | Value | Notes |
|---|---|---|
| Initial capital | $1,000,000 | Paper/backtest default |
| Commission | 10 bps | Per trade |
| Slippage | 5 bps | Per trade |
| Min trade threshold | 0.5% | Weight change required to trigger trade |
| Leverage limit | 2.0× | Sum of |weights| |
| Walk-forward in-sample | 1 year | Minute bars |
| Walk-forward out-of-sample | 3 months | |
| Monte Carlo simulations | 10,000 | Block bootstrap |
| Consistency ratio target | > 0.6 | Fraction of OOS windows profitable |
| Sharpe ratio target | > 1.0 | Annualized, OOS |
| Max drawdown target | < 20% | 95% MC CI |

### Paper Trading Graduation Criteria

| Criterion | Threshold | Period |
|---|---|---|
| Cumulative return | > 0% | 6 months |
| Annualized Sharpe | > 1.0 | 6 months |
| Max drawdown | < 20% | Since inception |
| MC 5th pctl return | > 0% | 10K sims |
| Consistency ratio | > 0.6 | Rolling 3-month OOS |
| Adaptive > Frozen Sharpe | > 0.1 | 3 months |
| State stability | No monotonic growth | 6 months |
| Uptime | > 99.5% | 6 months |
| p99 latency | < 100 ms | Since inception |

---

## Appendix D — Event Type Catalog

| Event Type | Source API | Detection Method | Magnitude Encoding | Typical Frequency |
|---|---|---|---|---|
| INSIDER_BUY | `insider_transactions()` | Diff consecutive polls | Dollar value (log) | ~2/week per asset |
| INSIDER_SELL | `insider_transactions()` | Diff | Dollar value (log) | ~5/week per asset |
| ANALYST_UPGRADE | `grading_history()` | Diff | Grade delta (e.g., hold→buy = +1) | ~1/month per asset |
| ANALYST_DOWNGRADE | `grading_history()` | Diff | Grade delta | ~1/month per asset |
| EARNINGS_BEAT | `earnings_history()` | Actual > estimate | Surprise % | Quarterly |
| EARNINGS_MISS | `earnings_history()` | Actual < estimate | Surprise % | Quarterly |
| SEC_FILING_10K | `sec_filings()` / `edgar::submissions()` | New filing detected | Binary | Annual |
| SEC_FILING_10Q | Same | New filing | Binary | Quarterly |
| SEC_FILING_8K | Same | New filing | Binary (high priority — unscheduled) | ~4/year per asset |
| SEC_FILING_OTHER | Same | New filing | Binary | Variable |
| FOMC_DECISION | `feeds::fetch(FederalReserve)` | RSS new entry matching pattern | Rate change bps | 8/year (global) |
| MACRO_RELEASE | FRED series update | New data point for tracked series | Surprise vs consensus | Variable |
| NEWS_HIGH_IMPACT | MarketReader score | Sentiment magnitude > threshold | Sentiment score | ~1/day per asset |
| DIVIDEND_ANNOUNCED | `calendar_events()` | New dividend date appears | Yield change | Quarterly |
| SPLIT_ANNOUNCED | `splits()` | New split detected | Split ratio | Rare |
| OPTIONS_UNUSUAL | Options chain polling | Volume > 2σ AND volume/OI > threshold | Volume ratio | ~1/week per asset |
| TRENDING_ENTRY | `finance::trending()` | Ticker appears in trending list | Binary | Variable |
| RATING_CHANGE | `recommendation_trend()` | Distribution shift | Net change in consensus score | Monthly |

---

*Meridian v3.0 — Harvey, March 2026*
*Built on Gated DeltaNet (ICLR 2025), BitNet b1.58, and the full finance-query ecosystem.*
*"Every signal matters. Every event counts. The market never sleeps, and neither does Meridian."*
