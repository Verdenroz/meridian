# MERIDIAN: Regime-Adaptive State-Space Portfolio Model

## Architectural Specification v2.0 — Final

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Novel Contributions](#3-novel-contributions)
4. [Hardware Target & Constraints](#4-hardware-target--constraints)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Component Specifications](#6-component-specifications)
   - 6.1 [Input Feature Engineering](#61-input-feature-engineering)
   - 6.2 [Financial Calendar Embedding](#62-financial-calendar-embedding)
   - 6.3 [Input Projection Layer](#63-input-projection-layer)
   - 6.4 [Regime Detector](#64-regime-detector)
   - 6.5 [Per-Asset Temporal Backbone (RA-GDN Stack)](#65-per-asset-temporal-backbone-ra-gdn-stack)
   - 6.6 [Regime-Routed Sparse Mixture of Experts](#66-regime-routed-sparse-mixture-of-experts)
   - 6.7 [Temporal Cascade](#67-temporal-cascade)
   - 6.8 [Cross-Asset Fusion Module](#68-cross-asset-fusion-module)
   - 6.9 [Financial Episodic Memory](#69-financial-episodic-memory)
   - 6.10 [Portfolio Output Heads](#610-portfolio-output-heads)
   - 6.11 [Auxiliary Prediction Head](#611-auxiliary-prediction-head)
   - 6.12 [MarketReader Text Encoder](#612-marketreader-text-encoder)
7. [Continuous Learning System](#7-continuous-learning-system)
   - 7.1 [Tier 1 — Surprise-Gated Delta Dynamics](#71-tier-1--surprise-gated-delta-dynamics)
   - 7.2 [Tier 2 — Normalization Adaptation](#72-tier-2--normalization-adaptation)
   - 7.3 [Tier 3 — Periodic Consolidation (EWC + Ternary LoRA)](#73-tier-3--periodic-consolidation-ewc--ternary-lora)
8. [Quantization Strategy](#8-quantization-strategy)
9. [Training Pipeline](#9-training-pipeline)
   - 9.1 [Data Sources & Preprocessing](#91-data-sources--preprocessing)
   - 9.2 [Phase 1 — Self-Supervised Pre-training](#92-phase-1--self-supervised-pre-training)
   - 9.3 [Phase 2 — Portfolio Optimization Training](#93-phase-2--portfolio-optimization-training)
   - 9.4 [Phase 3 — Meta-Learning for Plasticity](#94-phase-3--meta-learning-for-plasticity)
   - 9.5 [Phase 4 — Quantization-Aware Training](#95-phase-4--quantization-aware-training)
10. [Two-Head Strategy (Meridian-R / Meridian-S)](#10-two-head-strategy-meridian-r--meridian-s)
11. [Inference Runtime on Pi 5](#11-inference-runtime-on-pi-5)
12. [Memory Budget](#12-memory-budget)
13. [Development Roadmap](#13-development-roadmap)
14. [Risk Analysis & Mitigations](#14-risk-analysis--mitigations)
15. [Appendix A — Parameter Count Breakdown](#appendix-a--parameter-count-breakdown)
16. [Appendix B — finance-query Integration Map](#appendix-b--finance-query-integration-map)
17. [Appendix C — Hyperparameter Reference](#appendix-c--hyperparameter-reference)

---

## 1. Executive Summary

Meridian is a novel CPU-native neural network architecture designed to manage a portfolio of up to 100 US-listed assets (equities, ETFs, crypto) 24/7 on a Raspberry Pi 5. It produces allocation weights every minute using a shared backbone with two output heads: one optimizing raw returns (Meridian-R), one optimizing risk-adjusted returns via the Differential Sharpe Ratio (Meridian-S).

The backbone is a **Regime-Aware Gated DeltaNet** (RA-GDN) — a novel adaptation of the Gated DeltaNet linear attention architecture (ICLR 2025) with regime-conditioned dynamics, surprise-gated online plasticity, and ternary-native quantization. It is trained on H100 GPUs, quantized to mixed-precision (ternary weights, FP16 hidden states, INT8 activations), and deployed with a custom C/NEON inference kernel.

Key metrics:

| Metric | Value |
|---|---|
| Total parameters | ~82M (31M active per forward pass) |
| Quantized model size | ~19 MB |
| Runtime memory footprint | ~120 MB |
| Inference latency (100 assets) | ~3–8 ms |
| CPU utilization at 1 inf/min | < 0.02% |
| Power draw (sustained) | < 3W board total |
| Adaptation mechanisms | 3-tier (sub-second / hourly / daily) |
| Training compute | ~200–400 H100-hours |

---

## 2. Design Philosophy

Five interlocking principles govern every architectural decision:

**Temporal nativeness.** Financial markets are continuous processes with dynamics operating across sub-minute microstructure, daily momentum, weekly mean-reversion, and monthly macro cycles. The backbone models time as a first-class citizen through a recurrent state that theoretically captures infinite history, updates in O(1) per step at inference, and requires no re-processing of context. This is critical for a model running 24/7 without restarts.

**Regime awareness.** Markets change. A model trained on 2020–2025 data will encounter dynamics in 2026 that violate its training distribution. Meridian addresses this at three architectural levels: (1) the recurrent dynamics themselves adapt based on detected market regime via conditioned gating, (2) a mixture-of-experts routes to specialized sub-networks based on regime posterior probabilities, and (3) a continuous learning system updates parameters during live inference using prediction errors as signal.

**Extreme efficiency.** Every operation must justify its compute cost on a Cortex-A76. No quadratic attention over long sequences, no dense floating-point weight matrices where ternary suffices, no redundant recomputation. The model is designed for ternary weights from inception — not quantized as an afterthought — following the BitNet b1.58 methodology of training with quantization constraints from step zero.

**Financial domain encoding.** Unlike general-purpose LLMs that learn everything from token co-occurrence, Meridian bakes in financial structure: time-of-day effects, sector relationships, asset-class-specific feature encoders, calendar-aware temporal embeddings, and variable selection gating that dynamically weights input features based on regime. This lets a 82M-parameter model outperform a larger generic one within its domain.

**Separation of understanding from strategy.** The backbone learns to understand market state. The output heads learn allocation strategy. This separation enables the shared-backbone-two-heads architecture for Meridian-R and Meridian-S, and makes future extensions (new objectives, explainability heads, risk parity head) modular additions rather than architectural rewrites.

---

## 3. Novel Contributions

Seven architectural innovations distinguish Meridian from existing work:

### 3.1 Regime-Aware Gated DeltaNet (RA-GDN)

Standard Gated DeltaNet combines adaptive forgetting (gated decay) with error-correcting writes (delta rule) into a single recurrence:

```
S_t = α_t · S_{t-1} + β_t · k_t ⊗ (v_t − α_t · S_{t-1}^T k_t)
```

This is mathematically equivalent to recursive least squares — the same algorithm quantitative finance has relied on for decades. Meridian adds a second conditioning path: the **regime embedding**. A small regime detector network ingests market-wide features (realized volatility, trend slope, cross-asset correlation dispersion, yield curve shape, VIX level) and produces a compact regime vector. This vector modulates the GDN dynamics at a level deeper than input conditioning:

- **α (forgetting gate)** receives an additive bias from the regime embedding. This controls the state's "metabolic rate." In high-volatility regimes, α decreases toward 0, causing faster forgetting and weighting recent observations more heavily. In stable regimes, α increases toward 1, preserving longer memory. This is the core mechanism by which the model's effective memory horizon adapts to market conditions without any weight changes.

- **β (update strength)** receives a multiplicative scale from the regime embedding. During regime transitions (high regime-change velocity), β is amplified, allowing the state to absorb new dynamics more aggressively. During stable regimes, β is dampened.

This dual conditioning (input-dependent gating + regime-dependent dynamics) is architecturally novel. Input gating handles token-level "what to absorb" decisions; regime modulation handles system-level "how to process" decisions.

### 3.2 Surprise-Gated Plasticity (SGP)

Continuous learning on a CPU requires a mechanism that is (a) computationally cheap, (b) resistant to catastrophic forgetting, and (c) self-regulating. SGP draws from neuromodulatory plasticity in biological neural networks and the Titans framework (Google Research, 2025).

The model maintains an auxiliary prediction task (forecasting next-minute returns, volatility, and correlation change per asset). After each minute, when actuals are observed, the prediction error is computed and compared against an exponentially-weighted running baseline to produce a **surprise signal** — how much worse the prediction was than the model's recent average.

The critical insight: the delta rule's error term `v_t − S_{t-1}^T k_t` IS already a prediction error. By modulating the update strength β_t by prediction surprise, we get surprise-gated plasticity for free within the existing recurrence:

```
β_t = β_base × sigmoid(γ × surprise_t)
```

When surprise is low, the model's understanding is adequate — minimal updates. When surprise is high, the market has done something unexpected — accelerated learning. This adds approximately 50 FLOPs per step — negligible.

### 3.3 Temporal Cascade with Adaptive Speed

The model processes information at four temporal resolutions (minute, hourly, daily, weekly) using separate GDN stacks connected bidirectionally:

- **Upward aggregation**: lower-resolution GDNs receive pooled representations from the level below at their natural cadence.
- **Downward context**: higher-resolution GDNs receive projected context from the level above, added as a bias to their input.

The novel element is **adaptive cascade speed**. During high-surprise periods, the downward context injection is amplified (the daily context has more influence on minute-level decisions). During low-surprise periods, the downward injection is dampened. The amplification factor is a direct function of the surprise signal from SGP, creating a feedback loop between the plasticity system and the temporal architecture.

### 3.4 Financial Episodic Memory

A fixed-size FAISS-backed k-NN buffer (10,000–50,000 slots) storing compressed representations of the most "surprising" market states encountered during live operation. Each slot contains: the cross-asset representation at the time of surprise, the surprise magnitude, and a tag indicating what happened next (recovery within N minutes, volatility expansion, correlation spike).

During cross-asset fusion, the top-k most similar historical episodes are retrieved and their outcomes concatenated as additional context for the policy network. A two-tier write gate prevents storing repetitive high-volatility events while capturing genuinely novel patterns. This is architecturally similar to retrieval-augmented generation but entirely internal and financially specialized.

### 3.5 External Regime-Routed Sparse MoE

Standard MoE routes via a learned dense gate network, which is fragile under ternary quantization because the limited weight resolution produces noisy routing decisions. Meridian uses **external regime-based routing**: an HMM/GMM regime detector's posterior probabilities directly determine expert activation weights. During training, experts naturally specialize in different market conditions because they consistently receive data from those conditions.

This eliminates learned-routing instability under quantization, provides interpretability (you can inspect which expert handles which regime), and leverages decades of validated regime-detection methodology in quantitative finance. The router is "free" — it is the existing regime detector.

### 3.6 Financial Calendar Embedding

Rather than generic positional encoding, Meridian uses a multi-component temporal embedding that encodes market-specific temporal structure: minute within trading session, day of week, proximity to FOMC meetings, options expiration, earnings dates, and market-open status. These are concatenated into a time vector and projected to the model's hidden dimension, then added to every asset token at every timestep. This eliminates the need for the model to learn calendar effects from data alone.

### 3.7 Ternary-Native Architecture with Precision Highway

The entire architecture is designed from the ground up for BitNet b1.58 ternary quantization. Weight matrices use absmean ternary quantization with straight-through estimators during training. The critical innovation is the **Precision Highway**: the recurrent hidden state S is maintained in FP16, with periodic RMSNorm stabilization, while all linear projections are ternary. This prevents the unbounded error accumulation that plagues ternary recurrent models during continuous operation, validated by the Precision Highway principle (Park et al., 2018) and Q-Mamba findings showing state caches consume 79.6% of memory in naively quantized SSMs.

---

## 4. Hardware Target & Constraints

### Primary Target: Raspberry Pi 5 (8GB)

| Component | Specification |
|---|---|
| SoC | BCM2712, quad-core Cortex-A76 @ 2.4 GHz |
| SIMD | Dual 128-bit NEON pipelines per core |
| RAM | 8 GB LPDDR4X @ 4267 MT/s |
| Memory bandwidth | ~8.5 GB/s theoretical, ~6 GB/s practical |
| INT8 throughput | ~307 GOPS (4 cores, dual-issue NEON) |
| Ternary throughput (optimized) | ~400–600 effective GOPS (bitwise kernels) |
| FP16 throughput | ~76 GFLOPS (4 cores, NEON FMLA) |
| FP32 throughput | ~38 GFLOPS |
| Storage | NVMe SSD via PCIe 2.0 x1 |
| Power | 5V/5A USB-C, ~3W idle, ~8W peak |
| Thermal | Official active cooler, sustained 59–72°C under load |
| OS | Raspberry Pi OS Lite 64-bit (headless) |

### Budget Allocation

| Resource | Budget | Meridian Usage |
|---|---|---|
| RAM | 8 GB total, ~5.5 GB available | ~120 MB model + runtime |
| CPU | 4 cores | Cores 0–2 inference, Core 3 OS/data pipeline |
| Storage | 256 GB NVMe | Model (19 MB), episodic memory (25 MB), logs |
| Network | Ethernet/WiFi | finance-query data ingestion, continuous |
| Power | Sustained 24/7 | < 3W under inference workload |

### Training Target: H100 Cloud

| Resource | Specification |
|---|---|
| GPU | NVIDIA H100 80GB SXM |
| Budget | Up to 500 GPU-hours |
| Framework | PyTorch 2.x + Flash Linear Attention (FLA) library |
| Precision | BF16 training with ternary QAT in final phase |
| Parallelism | Data parallel across 4–8 GPUs for large batch phases |

---

## 5. High-Level Architecture

The data flow processes one minute-step for all 100 assets:

```
INPUTS (per minute)
├── Per-Asset Features (100 assets × 112 dims each)
│   ├── Price & Volume: OHLCV normalized + multi-horizon returns (10)
│   ├── Technical Indicators: trend, momentum, volatility, volume, composite (20)
│   ├── Options-Derived: IV, P/C ratio, skew, term structure, OI change (5)
│   ├── Fundamental: PE, growth, margins, sector embedding (10)
│   ├── Text: MarketReader 64-dim embedding (64)
│   └── Asset-Specific Calendar: days to earnings, ex-div (3)
├── Market-Wide Features (24 dims)
│   ├── VIX level + change (2)
│   ├── Yield curve: spreads, level, slope (4)
│   ├── Fear & Greed, breadth, sector dispersion (4)
│   ├── FRED macro: fed funds, unemployment, CPI, claims (4)
│   ├── Cross-asset: DXY, oil, gold, BTC (4)
│   ├── Market volatility ratio, aggregate P/C, HY spread, net new H/L (4)
│   └── Rolling cross-asset correlation (1), padding (1)
├── Portfolio State (201 dims)
│   ├── Current weights per asset (100)
│   ├── Unrealized P&L per asset (100)
│   └── Total portfolio value, log-scaled (1)
└── Financial Calendar Embedding (20 global dims)

         │
         ▼
┌─────────────────────────────────────────────────┐
│            INPUT PROJECTION LAYER               │
│                                                 │
│  Per-asset: VSN-Gated Linear(112 → D) + RMSNorm│
│  Market:    Linear(24 → D) + RMSNorm           │
│  Portfolio: Linear(201 → D) + RMSNorm          │
│  Calendar:  Linear(20 → D)                     │
│  All summed per asset token                     │
│  D = 768                                        │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│       REGIME DETECTOR (shared across assets)    │
│                                                 │
│  Input: market-wide features (24) + portfolio   │
│         summary stats + rolling regime state    │
│  Architecture: 2-layer MLP (64 → 128 → 64)     │
│  Outputs:                                       │
│    regime_embedding ∈ ℝ^64                      │
│    regime_posteriors ∈ ℝ^4 (softmax)            │
│    regime_change_velocity ∈ ℝ                   │
│  Also fed to HMM for posterior refinement       │
└─────────────────────────────────────────────────┘
         │ regime_embedding, regime_posteriors
         ▼
┌─────────────────────────────────────────────────┐
│      PER-ASSET TEMPORAL BACKBONE (RA-GDN)       │
│   (shared weights, separate state per asset)    │
│                                                 │
│  ┌── RA-GDN Block ──────────────────────────┐   │
│  │  Token Shift → RMSNorm                   │   │
│  │  Q,K projections → L2-Norm → Conv1d(k=4) │   │
│  │  V projection → Conv1d(k=4) → SiLU       │   │
│  │  α = sigmoid(gate_proj + regime_bias)     │   │
│  │  β = sigmoid(update_proj × regime_scale)  │   │
│  │  S = α·S + β·k⊗(v − α·Sᵀk)             │   │
│  │  Output = (qᵀ·S) ⊙ output_gate          │   │
│  │  → Residual + RMSNorm                    │   │
│  └──────────────────────────────────────────┘   │
│  × 8 layers                                     │
│                                                 │
│  Regime-Routed FMoE after layers 3, 5, 7        │
│  4 experts, 2 active, regime-posterior-routed   │
│                                                 │
│  Output: 100 asset representations (D=768 each) │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│           TEMPORAL CASCADE                      │
│                                                 │
│  Hourly GDN  (1 layer, D=512, updates q60 min) │
│  Daily GDN   (1 layer, D=512, updates q390 min)│
│  Weekly GDN  (1 layer, D=384, updates q5 days) │
│                                                 │
│  ↑ Mean-pooled inputs from minute backbone      │
│  ↓ Projected context injection back to backbone │
│    with surprise-gated amplification            │
│                                                 │
│  Fusion: [minute; hourly; daily; weekly]        │
│  → Linear(D+512+512+384 → D) per asset         │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│        CROSS-ASSET FUSION MODULE                │
│                                                 │
│  Token set:                                     │
│    100 asset tokens (D=768)                     │
│    1 market-wide token (D=768)                  │
│    1 portfolio state token (D=768)              │
│    + k episodic memory retrievals (D=768)       │
│    = 102+k tokens (k=8 default)                 │
│                                                 │
│  2 layers of multi-head attention               │
│  (8 heads, 96 dims/head, sector-bias prior)     │
│  + SwiGLU FFN(D → 2D → D) per layer            │
│                                                 │
│  Output: 100 refined asset representations      │
└─────────────────────────────────────────────────┘
         │
         ├──────────────────┬────────────────────┐
         ▼                  ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ RETURN HEAD  │   │ SHARPE HEAD  │   │ AUXILIARY HEAD   │
│ (Meridian-R) │   │ (Meridian-S) │   │ (for surprise)   │
│              │   │              │   │                  │
│ Per-asset:   │   │ Per-asset:   │   │ Per-asset:       │
│ MLP(D→256→2)│   │ MLP(D→256→2)│   │ MLP(D→128→3)    │
│ →(weight,    │   │ →(weight,    │   │ →(ret_pred,      │
│   confidence)│   │   confidence)│   │   vol_pred,      │
│              │   │              │   │   corr_pred)     │
│ Normalization│   │ Normalization│   │                  │
│ → leverage   │   │ → leverage   │   │ Used ONLY for    │
│   constrained│   │   constrained│   │ surprise signal  │
│   weights    │   │   weights    │   │ & plasticity     │
└──────────────┘   └──────────────┘   └──────────────────┘
```

---

## 6. Component Specifications

### 6.1 Input Feature Engineering

Every minute, 100 asset feature vectors are assembled. Each vector contains 112 scalar features, derived from finance-query APIs during live operation and from historical datasets during training.

#### Price & Volume Block (10 features per asset)

OHLCV from `chart(Interval::OneMinute, ...)` — 5 raw values, normalized:

| Feature | Computation | Source |
|---|---|---|
| Open (normalized) | `ln(open / prev_close)` | `chart()` |
| High (normalized) | `ln(high / open)` | `chart()` |
| Low (normalized) | `ln(low / open)` | `chart()` |
| Close (normalized) | `ln(close / prev_close)` | `chart()` |
| Volume (normalized) | `ln(volume / SMA_20(volume))` | `chart()` |
| Return 1m | `ln(close / close_{t-1})` | Computed |
| Return 5m | `ln(close / close_{t-5})` | Computed |
| Return 15m | `ln(close / close_{t-15})` | Computed |
| Return 60m | `ln(close / close_{t-60})` | Computed |
| Return 1d | `ln(close / close_{t-390})` | Computed |

All log returns provide explicit multi-scale momentum information without requiring the model to learn the computation.

#### Technical Indicator Block (20 features per asset)

Computed via finance-query's indicator API. Selected for maximum information density with minimum redundancy:

| Category | Features | Source |
|---|---|---|
| Trend (4) | SMA(20) z-score, SMA(50) z-score, SMA(200) z-score, ADX(14) | `indicators::sma()`, `indicators::adx()` |
| Momentum (4) | RSI(14), MACD histogram / ATR, Stochastic %K(14,3,3), ROC(10) | `indicators::rsi()`, `indicators::macd()`, `indicators::stochastic()`, `indicators::roc()` |
| Volatility (4) | ATR(14)/price, Bollinger %B(20,2), Keltner position, RV ratio (5d/20d) | `indicators::atr()`, `indicators::bollinger_bands()`, `indicators::keltner_channels()` |
| Volume (4) | OBV 5-period slope, CMF(20), MFI(14), VWAP deviation | `indicators::obv()`, `indicators::cmf()`, `indicators::mfi()`, `indicators::vwap()` |
| Composite (4) | SuperTrend direction (±1), Ichimoku cloud position (-1/0/+1), Parabolic SAR direction (±1), Choppiness(14) | `indicators::supertrend()`, `indicators::ichimoku()`, `indicators::parabolic_sar()` |

All indicators are normalized to approximately zero-mean, unit-variance using **rolling z-scores** (lookback 252 trading days) maintained as running statistics during inference. The rolling statistics are part of the data pipeline, not the model.

#### Options-Derived Block (5 features per asset)

From `options(date)`:

| Feature | Computation |
|---|---|
| Aggregate 30-day IV | Weighted average of near-the-money call/put IVs |
| Put/Call volume ratio | `total_put_volume / total_call_volume` |
| 25-delta skew | 25-delta put IV minus 25-delta call IV |
| IV term structure slope | Front month IV / back month IV |
| Open interest change | Daily change in total OI, normalized |

For assets without options (some ETFs, crypto), these 5 features are filled with a **learned default embedding** — a trainable 5-dim vector per asset class (equity, ETF, crypto) that the model learns to interpret as "no options signal available."

#### Fundamental Block (10 features per asset)

Updated quarterly via `financial_data()`, `summary_detail()`, `key_stats()`, `asset_profile()`:

| Feature | Normalization | Source |
|---|---|---|
| Trailing PE | Cross-sectional percentile rank [0,1] | `summary_detail()` |
| Forward PE | Cross-sectional percentile rank [0,1] | `summary_detail()` |
| Dividend yield | Raw (already bounded) | `summary_detail()` |
| Market cap | `ln(market_cap)` then percentile rank | `summary_detail()` |
| Beta | Raw | `key_stats()` |
| Earnings growth rate | Winsorized at ±100% | `financial_data()` |
| Revenue growth rate | Winsorized at ±100% | `financial_data()` |
| Profit margin | Raw (already bounded [-1,1]) | `financial_data()` |
| Debt-to-equity | Cross-sectional percentile rank | `financial_data()` |
| Sector embedding | Learned scalar per GICS sector (11 → 1) | `asset_profile()` |

The sector embedding is a single learned scalar per sector (not a one-hot), trained end-to-end. This provides sector identity in a single feature dimension. A full sector relationship matrix is encoded in the cross-asset attention's sector bias prior (Section 6.8).

#### Text Block (64 features per asset)

From `MarketReader` (Section 6.12) processing news from `feeds::fetch_all()`, `news()`, and `finance::earnings_transcript()`. When no new text is available for an asset, the last encoded text representation is used with a temporal decay factor:

```
text_embedding_t = text_embedding_last × exp(-λ × minutes_since_last_text / 1440)
```

Where λ = 1.0 gives a half-life of approximately 24 hours, reflecting decreasing relevance of old news.

#### Asset-Specific Calendar Features (3 per asset)

| Feature | Encoding |
|---|---|
| Days to next earnings | `1 / (1 + days)` — approaches 1 as earnings approach |
| Days to next ex-dividend | `1 / (1 + days)` |
| Is earnings week | Binary flag |

Sourced from `calendar_events()`.

**Total per asset: 10 + 20 + 5 + 10 + 64 + 3 = 112 features.**

---

### 6.2 Financial Calendar Embedding

A multi-component temporal embedding shared across all assets, encoding market-specific temporal structure:

| Component | Encoding | Dims |
|---|---|---|
| Minute of trading session (0–389) | sin/cos pair | 2 |
| Day of week (1–5) | sin/cos pair | 2 |
| Day of month (1–31) | sin/cos pair | 2 |
| Month of year (1–12) | sin/cos pair | 2 |
| Market open flag | Binary | 1 |
| Pre-market flag | Binary | 1 |
| Post-market flag | Binary | 1 |
| Minutes to market open | `1 / (1 + min)`, 0 if open | 1 |
| Minutes to market close | `1 / (1 + min)`, 0 if closed | 1 |
| Minutes since last trade | Log-scaled | 1 |
| Days to next FOMC meeting | `1 / (1 + days)` | 1 |
| Days to next quad witching | `1 / (1 + days)` | 1 |
| Is options expiration week | Binary | 1 |
| Is first 30 min of session | Binary | 1 |
| Is last 30 min of session | Binary | 1 |
| Weekend flag | Binary | 1 |
| **Total** | | **20** |

FOMC dates and quad-witching dates are sourced from a static calendar file updated annually. All cyclical features use sin/cos encoding to preserve continuity (e.g., Monday and Friday are close in the weekly cycle embedding).

Projected: `Linear(20 → D)`, then added to every asset token.

---

### 6.3 Input Projection Layer

Each input modality is projected to the model's hidden dimension D=768 independently, then fused:

```
Per-Asset:
  x_asset = VSN(features_112) → Linear(112 → 768) → RMSNorm
  
Market:
  x_market = Linear(24 → 768) → RMSNorm
  
Portfolio:
  x_portfolio = Linear(201 → 768) → RMSNorm
  
Calendar:
  x_calendar = Linear(20 → 768)

Per-asset token:
  token_i = x_asset_i + x_market + x_portfolio + x_calendar + calendar_per_asset_i
```

**Variable Selection Network (VSN):** Before the per-asset linear projection, a gating mechanism (adapted from the Temporal Fusion Transformer) dynamically weights the 112 input features. The VSN computes per-feature importance weights conditioned on the regime embedding:

```
gates = softmax(Linear_gate(regime_embedding)) ∈ ℝ^112
x_selected = gates ⊙ features_112
```

This allows the model to, for example, downweight momentum indicators during choppy regimes and upweight volatility indicators during crisis regimes. The VSN adds 112 × 64 + 64 × 112 = ~14K parameters — negligible.

**Parameters:** ~265K total (112×768 + 24×768 + 201×768 + 20×768 + VSN).

---

### 6.4 Regime Detector

A lightweight network producing a regime embedding that conditions the backbone's dynamics and routes MoE experts.

**Input:** Concatenation of:
- Market-wide features (24 dims)
- Portfolio summary: total value, aggregate P&L, current leverage (3 dims)
- Rolling regime state: EMA of last 60 regime embeddings (64 dims)
- **Total: 91 dims**

**Architecture:**

```
Linear(91 → 128) → SiLU → Dropout(0.1)
Linear(128 → 128) → SiLU → Dropout(0.1)
├── Linear(128 → 64) → regime_embedding
├── Linear(128 → 4) → softmax → regime_posteriors  
└── regime_change_velocity = ||regime_embedding_t − regime_embedding_{t-1}||_2
```

**Regime posterior interpretation:** The 4 regime classes are:

| Regime | Characterized By |
|---|---|
| Low Volatility / Trending | ADX > 25, VIX < 20, positive momentum |
| High Volatility / Trending | ADX > 25, VIX > 25, strong directional moves |
| Mean-Reverting / Range-Bound | ADX < 20, low VIX, oscillating prices |
| Crisis / Dislocation | VIX > 35, correlation spike, drawdown > 5% |

These are initialized from a pre-fit HMM on historical data, then fine-tuned end-to-end. During inference, the HMM posterior is blended with the neural regime posterior for routing robustness:

```
final_posterior = 0.7 × neural_posterior + 0.3 × hmm_posterior
```

The HMM is fit offline on rolling 2-year windows and costs negligible compute at inference (a single matrix multiply per step).

**Parameters:** ~25K

---

### 6.5 Per-Asset Temporal Backbone (RA-GDN Stack)

The heart of the model. Processes each asset's temporal feature stream through 8 Regime-Aware Gated DeltaNet blocks with 3 interleaved FMoE blocks.

**Shared weights, separate states.** All 100 assets pass through the same weight matrices, but each maintains its own GDN hidden state S and convolutional cache. The model learns universal patterns; the per-asset state captures unique temporal context.

#### RA-GDN Block Internals (one block)

```
Input: x ∈ ℝ^D  (D=768)

1. Token shift: x_shifted = lerp(x, x_{t-1}, λ_shift)
   λ_shift is a learnable per-channel vector ∈ [0,1]
   Provides short-range temporal mixing before the recurrence.

2. Pre-norm: x_normed = RMSNorm(x_shifted)

3. Q/K/V projections:
   Q = L2Norm(Linear_Q(x_normed))  ∈ ℝ^{n_heads × d_head}  [n_heads=4, d_head=64]
   K = L2Norm(Linear_K(x_normed))  ∈ ℝ^{n_heads × d_head}
   V = Linear_V(x_normed)           ∈ ℝ^{n_heads × d_v}     [d_v=192]
   
   L2 normalization on Q/K eliminates softmax's large dynamic range,
   making the model ternary-quantization-friendly.

4. Short convolutions (for local context at inference):
   Q = SiLU(Conv1d(Q, kernel=4))  — cached buffer of 3 prior Q values
   K = SiLU(Conv1d(K, kernel=4))
   V = SiLU(Conv1d(V, kernel=4))

5. Gating with regime conditioning:
   α_raw = Linear_α(x_normed)                    ∈ ℝ^{n_heads}
   α_regime_bias = Linear_α_regime(regime_emb)    ∈ ℝ^{n_heads}
   α = sigmoid(α_raw + α_regime_bias)             — forgetting gate
   
   β_raw = Linear_β(x_normed)                    ∈ ℝ^{n_heads}
   β_regime_scale = sigmoid(Linear_β_regime(regime_emb)) ∈ ℝ^{n_heads}
   β = sigmoid(β_raw) × β_regime_scale            — update strength

6. Delta rule state update (per head):
   prediction_error = v_t − α · S_{t-1}^T k_t    ∈ ℝ^{d_v}
   S_t = α · S_{t-1} + β · k_t ⊗ prediction_error   ∈ ℝ^{d_head × d_v}
   
   State dimensions: 4 heads × 64 × 192 = 49,152 floats per asset per layer
   At FP16: 98,304 bytes = 96 KB per asset per layer

7. Output:
   o_t = Q_t^T · S_t                              ∈ ℝ^{n_heads × d_v}
   output_gate = sigmoid(Linear_gate(x_normed))    ∈ ℝ^{n_heads × d_v}
   y = o_t ⊙ output_gate
   y = Linear_out(concat_heads(y))                 ∈ ℝ^D

8. Residual + post-norm:
   output = RMSNorm(x + y)
```

#### Block Architecture Summary

| Layer Index | Block Type | Notes |
|---|---|---|
| 0 | RA-GDN | |
| 1 | RA-GDN | + Hourly context injection |
| 2 | RA-GDN | |
| 3 | **FMoE** | 4 experts, regime-routed |
| 4 | RA-GDN | + Daily context injection |
| 5 | RA-GDN | |
| 6 | **FMoE** | 4 experts, regime-routed |
| 7 | RA-GDN | + Weekly context injection |
| 8 | RA-GDN | |
| 9 | **FMoE** | 4 experts, regime-routed |
| 10 | RA-GDN | Final representation |

Total: 8 RA-GDN blocks + 3 FMoE blocks = 11 effective layers.

#### Per-Block Parameter Count

**RA-GDN block:**

| Component | Computation | Parameters |
|---|---|---|
| Token shift λ | D | 768 |
| RMSNorm (pre) | D | 768 |
| Q projection | D × (n_heads × d_head) = 768 × 256 | 196,608 |
| K projection | 768 × 256 | 196,608 |
| V projection | 768 × 768 (n_heads × d_v = 4 × 192) | 589,824 |
| α gate projection | 768 × 4 | 3,072 |
| α regime bias | 64 × 4 | 256 |
| β gate projection | 768 × 4 | 3,072 |
| β regime scale | 64 × 4 | 256 |
| Output gate | 768 × 768 | 589,824 |
| Output projection | 768 × 768 | 589,824 |
| RMSNorm (post) | D | 768 |
| **Total per block** | | **~2.17M** |

**8 RA-GDN blocks: ~17.4M parameters.**

---

### 6.6 Regime-Routed Sparse Mixture of Experts

Inserted after RA-GDN layers 2, 5, and 8 (3 FMoE blocks total). Each FMoE block:

**Architecture:**
```
Input: x ∈ ℝ^D

1. Pre-norm: x_normed = RMSNorm(x)

2. Routing (external, not learned):
   routing_weights = final_regime_posteriors ∈ ℝ^4
   top_2_indices = argtop2(routing_weights)
   weights = renormalize(routing_weights[top_2_indices])  — sum to 1

3. Expert computation (only 2 of 4 fire):
   For each active expert i:
     e_i = SwiGLU(x_normed)
         = (Linear_up_i(x_normed) ⊙ SiLU(Linear_gate_i(x_normed))) 
           → Linear_down_i
     Expert dims: D → 2D → D  (768 → 1536 → 768)

4. Combine:
   y = Σ_i (weights[i] × e_i)

5. Residual:
   output = RMSNorm(x + y)
```

**SwiGLU** is used instead of ReLU/SiLU because it is better behaved under ternary quantization (smoother activation landscape) and has become standard in modern architectures.

**Expert specialization:** During training, the routing mechanism ensures each expert sees predominantly one regime's data. Over time, experts naturally specialize:

| Expert | Specialization |
|---|---|
| Expert 0 | Low volatility / trending: momentum strategies, trend following |
| Expert 1 | High volatility / trending: breakout detection, volatility scaling |
| Expert 2 | Mean-reverting / range-bound: mean reversion, pair relationships |
| Expert 3 | Crisis / dislocation: risk reduction, correlation spike handling |

**Load balancing:** During training, an auxiliary loss encourages balanced utilization:

```
L_balance = 4 × Σ_i (f_i × P_i)
```

Where f_i is the fraction of tokens routed to expert i and P_i is the average routing probability for expert i. This prevents expert collapse.

**Per-FMoE block parameters:**

| Component | Parameters |
|---|---|
| RMSNorm | 768 |
| 4 experts × (Linear_up + Linear_gate + Linear_down) | 4 × (768×1536 + 768×1536 + 1536×768) = 4 × 3,538,944 = 14,155,776 |
| RMSNorm (post) | 768 |
| **Total per block** | **~14.2M** |

**3 FMoE blocks: ~42.5M parameters** (but only ~21.3M active per forward pass, since 2 of 4 experts fire).

---

### 6.7 Temporal Cascade

Three additional GDN layers operate at coarser temporal granularities, each maintaining per-asset state:

#### Hourly GDN (1 RA-GDN block, D=512)

- **Update cadence:** Every 60 minutes
- **Input:** Mean pool of last 60 minute-level backbone outputs per asset (after layer 1)
- **State:** 4 heads × 64 × 128 = 32,768 FP16 per asset = 64 KB per asset
- **Parameters:** ~1.0M (same block structure, scaled to D=512)
- **Captures:** Intraday trends, session dynamics, momentum shifts

#### Daily GDN (1 RA-GDN block, D=512)

- **Update cadence:** Every 390 minutes (one US trading session); 1440 for crypto
- **Input:** Mean pool of last ~7 hourly outputs (one day's worth)
- **State:** Same as hourly = 64 KB per asset
- **Parameters:** ~1.0M
- **Captures:** Multi-day trends, earnings reactions, sector rotation

#### Weekly GDN (1 RA-GDN block, D=384)

- **Update cadence:** Every 5 daily updates
- **Input:** Mean pool of last 5 daily outputs
- **State:** 4 heads × 48 × 96 = 18,432 FP16 per asset = 36 KB per asset
- **Parameters:** ~0.6M
- **Captures:** Weekly/monthly macro trends, rate cycle effects, seasonal patterns

#### Bidirectional Cascade

**Upward:** Natural aggregation via mean pooling at each level's cadence.

**Downward:** Each higher level's output is projected and added to the minute backbone's residual stream at specific injection points:

| Cascade Level | Injection Point | Projection |
|---|---|---|
| Hourly → Minute | After RA-GDN layer 1 | Linear(512 → 768) |
| Daily → Minute | After RA-GDN layer 4 | Linear(512 → 768) |
| Weekly → Minute | After RA-GDN layer 7 | Linear(384 → 768) |

The injection magnitude is scaled by a **surprise-sensitive gate**:

```
gate_cascade = sigmoid(w_gate × surprise_ema + b_gate)
injected_context = gate_cascade × projected_cascade_output
```

High surprise → gate opens → macro context floods into minute processing.
Low surprise → gate partially closes → minute processing relies more on local signals.

#### Temporal Representation Fusion

Before entering the cross-asset module, each asset's representation is formed by concatenating outputs from all four temporal levels:

```
asset_repr = Linear([minute_out; hourly_out; daily_out; weekly_out])
           = Linear(768 + 512 + 512 + 384 → 768) per asset
```

**Total cascade parameters:** ~2.6M (blocks) + ~1.6M (projections) = ~4.2M.

---

### 6.8 Cross-Asset Fusion Module

After per-asset temporal processing, the 100 asset representations must interact to capture cross-asset dynamics: sector correlations, risk-on/risk-off flows, pair relationships, contagion effects.

**Token set:**
- 100 asset tokens (D=768 each)
- 1 market-wide token (market feature vector projected to D)
- 1 portfolio state token (portfolio vector projected to D)
- k episodic memory retrieval tokens (default k=8, from FAISS)
- **Total: 110 tokens**

**Architecture:** 2 layers, each containing:

```
1. Multi-Head Attention:
   - 8 heads, d_head = 96
   - Q, K, V projections: Linear(768 → 768) each
   - Standard scaled dot-product attention (NOT causal — all assets attend to all)
   - Sector bias prior added to attention logits (see below)
   - Output projection: Linear(768 → 768)
   - Residual + RMSNorm

2. Feed-Forward Network:
   - SwiGLU: Linear(768 → 1536), SiLU gate, Linear(1536 → 768)
   - Residual + RMSNorm
```

Note: Full quadratic attention is used here because 110 tokens is computationally trivial (110² = 12,100 attention scores × 8 heads = ~97K multiplies). This takes < 0.1 ms on CPU.

**Sector bias prior:** A trainable 11×11 sector affinity matrix S (symmetric, initialized from historical GICS sector return correlations) is used to bias attention:

```
attention_logits[i,j] += S[sector_i, sector_j]
```

This encodes prior knowledge that, e.g., financials attend more to each other than to utilities. The bias is learned end-to-end and can deviate from the initialization.

**Per-layer parameters:**

| Component | Parameters |
|---|---|
| Q, K, V projections | 3 × 768 × 768 = 1,769,472 |
| Output projection | 768 × 768 = 589,824 |
| RMSNorm | 768 |
| SwiGLU FFN | 768 × 1536 × 2 + 1536 × 768 = 3,538,944 |
| RMSNorm | 768 |
| **Total per layer** | **~5.9M** |

**2 layers: ~11.8M parameters.**

---

### 6.9 Financial Episodic Memory

A FAISS-backed k-NN buffer providing non-parametric pattern matching for rare market events.

**Buffer Structure:**

| Field | Type | Size |
|---|---|---|
| market_embedding | float16[128] | 256 bytes |
| surprise_score | float32 | 4 bytes |
| outcome_tag | int8[5] — recovery time, vol change, corr change, drawdown, regime after | 5 bytes |
| timestamp | uint32 (minutes since epoch) | 4 bytes |
| **Total per entry** | | **269 bytes** |

**Buffer size:** 50,000 entries × 269 bytes = **~13 MB**.

**FAISS index:** `IndexFlatL2` over 128-dim float16 vectors. At 50,000 entries, exact search takes < 1 ms on Pi 5 (validated on Pi 4 at lower core counts).

**Two-tier write gate:**

```
Tier 1: Is prediction error > mean + 2σ of recent errors?
  → No: discard (not surprising enough)
  → Yes: proceed to Tier 2

Tier 2: Query FAISS for 5 nearest neighbors.
  Is min_distance > novelty_threshold?
  → No: discard (this type of surprise has been seen before)
  → Yes: STORE — this is a genuinely novel market event
```

**At inference:**

```
1. Encode current cross-asset state → 128-dim query vector
   (via Linear(768 → 128) from the cross-asset module's market token)

2. FAISS search: retrieve top-k=8 most similar historical episodes

3. For each retrieved episode:
   - Project market_embedding back to D=768 via Linear(128 → 768)
   - Append outcome_tag as an additional conditioning vector
   
4. Concatenate as additional tokens in cross-asset attention
```

**Priority-based eviction:**

```
priority = surprise_score × recency_decay × (1 + retrieval_count × 0.1)
```

where `recency_decay = exp(-age_minutes / (30 × 24 × 60))` gives a half-life of 30 days. When the buffer is full, the lowest-priority entry is evicted.

**Parameters:** ~200K (projection layers). **Runtime overhead:** < 3 ms per inference step.

---

### 6.10 Portfolio Output Heads

Two heads share the backbone but learn different allocation strategies:

#### Meridian-R (Return Maximization Head)

```
Per asset i:
  h_i = refined_asset_repr_i  ∈ ℝ^768

  hidden = SiLU(Linear(768 → 256))
  weight_raw = Linear(256 → 1)      — unbounded raw weight
  confidence = sigmoid(Linear(256 → 1))   — [0, 1]

  weight_i = weight_raw × confidence

Portfolio normalization (long-short with leverage constraint):
  leverage_limit = 2.0  (sum of |weights| ≤ 2.0)
  
  if Σ|weight_i| > leverage_limit:
    weights = weights × (leverage_limit / Σ|weight_i|)
  
  Final: weights ∈ ℝ^100, Σ|w_i| ≤ 2.0
```

**Parameters:** 768×256 + 256 + 256×1 + 256×1 = ~197K per head.

#### Meridian-S (Sharpe Maximization Head)

Identical architecture to Meridian-R with separate learned weights. The difference is in the training objective (Section 10).

**Total head parameters:** ~394K (both heads combined).

---

### 6.11 Auxiliary Prediction Head

Used exclusively for computing the surprise signal that drives plasticity (Tier 1 adaptation) and episodic memory storage. Not used for portfolio decisions.

```
Per asset i:
  h_i = refined_asset_repr_i  ∈ ℝ^768

  hidden = SiLU(Linear(768 → 128))
  
  ret_pred = Linear(128 → 1)    — next-minute return prediction
  vol_pred = softplus(Linear(128 → 1))  — next-minute volatility prediction
  corr_pred = tanh(Linear(128 → 1))     — correlation change prediction
```

**Surprise computation (per asset):**

```
error_ret = |actual_ret - ret_pred|
error_vol = |actual_vol - vol_pred|
error_corr = |actual_corr_change - corr_pred|

composite_error = 0.5 × error_ret/σ_ret + 0.3 × error_vol/σ_vol + 0.2 × error_corr/σ_corr
  (normalized by running standard deviations)

surprise_per_asset = max(0, composite_error - ema_error)
  (ema_error is exponentially-weighted mean of recent composite errors, span=60)

aggregate_surprise = mean(surprise_per_asset)
```

**Parameters:** ~100K.

---

### 6.12 MarketReader Text Encoder

A lightweight GDN-based text encoder that converts raw financial text into 64-dimensional dense vectors. Runs asynchronously on the Pi, processing new text items as they arrive from RSS feeds and news APIs, outside the critical minute-by-minute inference path.

**Tokenizer:** Custom byte-pair encoding with a 4,096-token vocabulary trained on a financial text corpus (SEC filings, earnings transcripts, financial news from finance-query RSS feeds). The small vocabulary is deliberate — financial text has concentrated terminology, and a smaller vocabulary means smaller embedding tables and faster tokenization.

**Architecture:**

```
Embedding:    4,096 × 256 (token) + 512 × 256 (positional, max 512 tokens)
              = ~1.18M parameters

4 GDN Blocks: D=256, n_heads=4, d_head=32, d_v=64
              (standard Gated DeltaNet, no regime conditioning)
              = ~0.54M per block × 4 = ~2.16M parameters

Pooling:      Learned attention-over-positions (single head, 256-dim)
              Linear(256 → 1) → softmax over sequence → weighted average
              = ~256 parameters

Output:       Linear(256 → 64)
              = ~16K parameters

Total:        ~3.4M parameters
```

**Quantization:** INT8 (not ternary — text processing benefits from higher precision for embedding lookups). Quantized size: ~3.4 MB.

**Inference time:** ~2 ms per article (256 tokens) on Pi 5.

**Training:** Pre-trained on:
1. Financial sentiment classification (Financial PhraseBank, SEC sentiment datasets)
2. Financial topic modeling (EDGAR filing types, news categories)
3. Contrastive learning: articles about the same event should produce similar embeddings

The 64-dim output space is shaped such that cosine similarity reflects semantic similarity, and direction encodes sentiment polarity (positive returns → positive dot product with a learned "bullish" direction vector).

**MarketReader's weights are entirely in the "frozen" parameter partition.** Text understanding is considered stable — the meaning of "Fed raises rates" doesn't change; the market's reaction does, which is captured by the main backbone's plasticity.

---

## 7. Continuous Learning System

Financial markets are non-stationary. Meridian adapts to regime changes through a three-tier system operating at different timescales, without catastrophic forgetting.

### 7.1 Tier 1 — Surprise-Gated Delta Dynamics (Sub-Second)

**Mechanism:** The Gated DeltaNet's delta rule already performs error-correcting updates at each step. The term `v_t − S_{t-1}^T k_t` is literally a prediction error — the difference between the new value and what the current state would predict. By modulating the update strength β_t by the surprise signal from the auxiliary head, we get adaptive plasticity for free:

```
β_effective = β_base × (1 + γ × clamp(surprise_t, 0, 5))
```

Where:
- β_base is the standard gated update strength from the RA-GDN block
- γ = 0.2 (a learned scalar, initialized to 0.2)
- surprise_t is the aggregate surprise from the auxiliary head
- Clamped to prevent runaway updates from extreme outliers

**When surprise is low:** β_effective ≈ β_base. The state evolves normally.
**When surprise is high:** β_effective is amplified up to 2×. The state absorbs new information more aggressively.

**Cost:** ~50 FLOPs per step (one multiply + one add). Effectively zero.

**This is the fastest adaptation path** — it operates at every inference step and requires no gradient computation. It is purely within the forward pass.

### 7.2 Tier 2 — Normalization Adaptation (Hourly)

**Mechanism:** Update only the RMSNorm affine parameters (scale γ, bias β if used) across all layers via entropy minimization on the model's own predictions.

```
For each RMSNorm layer with scale parameter γ ∈ ℝ^D:
  
  L_entropy = -Σ p_i log(p_i)
    where p is the softmax of the model's confidence outputs
  
  L_drift = λ_drift × ||γ - γ_checkpoint||²
    (quadratic penalty anchoring to last stable values)
  
  γ ← γ - η_norm × ∇_γ (L_entropy + L_drift)
```

**Number of adaptable parameters:** (8 RA-GDN blocks × 2 norms + 3 FMoE blocks × 2 norms + 2 attention layers × 3 norms + cascades) × D ≈ 28 × 768 = 21,504 parameters.

**Update cadence:** Every 60 minutes (hourly), using the last 60 minutes of prediction data.

**Cost:** One backward pass through norm parameters only. With 21K parameters, this is < 50K FLOPs — negligible.

**Uncertainty fallback:** If the entropy loss increases for 3 consecutive updates (indicating the adaptation is making things worse), revert to the checkpoint values and skip adaptation until the next cycle.

### 7.3 Tier 3 — Periodic Consolidation (EWC + Ternary LoRA)

**Mechanism:** During low-activity windows (US market close, 4:00 PM – 8:00 PM ET, or crypto low-volume hours), perform targeted fine-tuning of the backbone using accumulated recent data.

**LoRA configuration:**
- Apply low-rank adapters to Q, K, V projections in all 8 RA-GDN blocks
- Rank r = 8 per adapter
- Total adapter parameters: 8 blocks × 3 projections × 2 × 768 × 8 = ~295K
- At INT8: ~295 KB

**Update procedure:**
```
1. Accumulate replay buffer: last 4 hours of (input, actual_return) pairs
   Buffer size: 240 steps × 100 assets × 112 features × 4 bytes ≈ 10.8 MB

2. Compute auxiliary prediction loss on replay buffer:
   L_pred = MSE(ret_pred, actual_ret) + MSE(vol_pred, actual_vol)

3. EWC regularization:
   L_ewc = λ_ewc × Σ_i F_i × (θ_i - θ*_i)²
   where F_i is the diagonal Fisher Information estimate
   and θ*_i is the parameter value at the last stable checkpoint

4. Combined loss:
   L = L_pred + L_ewc

5. Update LoRA adapters only:
   adapter_params ← adapter_params - η_lora × ∇_adapter L

6. Merge adapters back into base weights using ternary LoRA (LoTA-QAF):
   Ternary signed gradient descent ensures merged weights remain in {-1, 0, 1}
   
7. Update EWC checkpoint:
   θ*_i ← θ_i (current params become new anchor)
   F_i ← 0.9 × F_i + 0.1 × (∇L)² (running Fisher estimate)
```

**Memory overhead:**
- LoRA adapters (INT8): ~295 KB
- Fisher diagonal estimate (FP32): 8 blocks × 3 × 768 × 768 × 4 bytes ≈ 56 MB
- Old parameter snapshot (ternary): same as model ≈ 19 MB
- Replay buffer: ~11 MB
- **Total Tier 3 overhead: ~86 MB** (only allocated during consolidation window)

**Update cadence:** Once per day during market close. Duration: ~5–15 minutes on Pi 5.

---

## 8. Quantization Strategy

Meridian is designed for ternary quantization from inception, not quantized as an afterthought. The strategy follows the BitNet b1.58 methodology with adaptations for the recurrent architecture.

### Precision Allocation

| Component | Precision | Rationale |
|---|---|---|
| Linear projections (Q, K, V, gates, FFN, output) | **Ternary {-1, 0, 1}** | ~90% of parameters. Multiplies → adds on ARM NEON. |
| GDN hidden state S | **FP16** | Precision Highway: recurrent state accumulates errors at lower precision. |
| Conv1d caches | **FP16** | Only 3 cached values per layer — negligible memory. |
| Activations (intermediate) | **INT8** | Per-token absmax quantization as in BitNet b1.58. |
| RMSNorm parameters | **FP32** | Tier 2 adaptation target; must be high precision. Few parameters. |
| Regime detector | **INT8** | Small MLP; needs more resolution than ternary but runs infrequently. |
| MarketReader text encoder | **INT8** | Embedding lookups benefit from higher precision. |
| Episodic memory embeddings | **FP16** | FAISS similarity computation requires float. |
| Output head weights | **FP16** | Critical for allocation precision. Small parameter count. |

### Ternary Quantization Method (BitNet b1.58)

For each weight matrix W:

```
Training (forward pass):
  α = mean(|W|)                    — absmean scaling factor
  W_ternary = sign(W) ⊙ (|W| > α/2)   — threshold at α/2
  W_quantized = α × W_ternary      — scaled ternary
  
Training (backward pass):
  Straight-Through Estimator (STE): ∂L/∂W ≈ ∂L/∂W_quantized
  Full-precision shadow weights maintained for gradient accumulation

Inference:
  Store W_ternary as 2-bit packed array (00=0, 01=+1, 10=-1)
  Store α as a single FP16 scalar per output channel
  GEMV: accumulate additions/subtractions, multiply by α at the end
```

### ARM NEON Ternary Kernel

The inference kernel follows the TAB (Ternary Approximation with BITMASK) framework:

```c
// Pseudo-code for ternary GEMV on NEON
// W stored as two bitmasks per 128 values: mask_pos, mask_neg
// x is INT8 activation vector

int32x4_t acc = vdupq_n_s32(0);
for (int j = 0; j < D; j += 128) {
    // Load 128 ternary weights as two 128-bit bitmasks
    uint64x2_t pos_mask = vld1q_u64(&W_pos[j/64]);
    uint64x2_t neg_mask = vld1q_u64(&W_neg[j/64]);
    
    // Load INT8 activations
    int8x16_t x_vec = vld1q_s8(&x[j]);
    
    // Accumulate: where pos_mask bit is set, add x; where neg, subtract
    // Uses NEON bitwise select + horizontal add
    acc = vadd_where_set(acc, x_vec, pos_mask);
    acc = vsub_where_set(acc, x_vec, neg_mask);
}
// Final: multiply accumulated sum by per-channel α (FP16)
float result = vaddvq_s32(acc) * alpha;
```

This achieves 13.4× speedup over standard INT8 GEMV on ARM (per TAB benchmarks) because:
1. No multiplications — just conditional additions
2. 128 weights packed into a single NEON register pair
3. Memory bandwidth reduced by ~4× vs INT8 weight storage

### State Drift Mitigation

The recurrent state S is maintained in FP16. To prevent unbounded drift during 24/7 operation:

1. **Periodic RMSNorm:** Every 60 steps (once per hour in minute-level processing), apply RMSNorm to each head's state matrix S:
   ```
   S_normalized = S / RMS(S) × γ_state
   ```
   Where γ_state is a learned scaling parameter.

2. **State magnitude monitoring:** Track `||S||_F` per head. If it exceeds 2× the training-time average, trigger an emergency normalization.

3. **Soft state reset:** At daily boundaries (market close), apply a learned decay:
   ```
   S_new_day = δ × S_end_of_day + (1 - δ) × S_checkpoint
   ```
   Where δ ∈ [0.9, 0.99] is a learned parameter and S_checkpoint is the state from the model's initialization (capturing the "default" market understanding).

---

## 9. Training Pipeline

### 9.1 Data Sources & Preprocessing

#### Historical Data (5 years: 2020-01-01 to 2025-06-30)

| Data Type | Source | Granularity |
|---|---|---|
| Minute OHLCV (US equities, ETFs) | Polygon.io / FirstRate Data | 1-minute |
| Minute OHLCV (crypto) | Binance / CoinGecko historical | 1-minute |
| Technical indicators | Computed from OHLCV via finance-query | 1-minute |
| Options chains | CBOE historical / OptionMetrics | Daily snapshots |
| Fundamentals | finance-query `financials()` / EDGAR XBRL | Quarterly |
| News headlines + summaries | finance-query RSS historical / GDELT | Event-driven |
| Earnings transcripts | finance-query `earnings_transcript()` | Quarterly |
| FRED macro series | finance-query `fred::series()` | Various (daily to monthly) |
| Treasury yield curve | finance-query `fred::treasury_yields()` | Daily |
| VIX | finance-query `Ticker::new("^VIX")` | 1-minute |
| Fear & Greed | finance-query `finance::fear_and_greed()` | Daily |
| Sector performance | finance-query `finance::sector()` | Daily |

#### Data Scale

- ~490,000 minutes of US market data per asset (5 years × 252 days × 390 minutes)
- ~2,600,000 minutes of crypto data (24/7)
- 100 assets × 112 features = 11,200 features per minute
- Total training samples: ~490K minute-steps (each with 100 assets)
- Uncompressed: ~20 GB (FP32) → ~5 GB (FP16) → stored as memory-mapped files

#### Preprocessing Pipeline

1. **OHLCV normalization:** Log-returns for prices, log-ratio to moving average for volume.
2. **Indicator computation:** All 20 indicators computed using finance-query's indicator API, then z-scored with rolling 252-day statistics.
3. **Options features:** Aggregated from raw chains, missing data filled with asset-class defaults.
4. **Fundamentals:** Percentile-ranked cross-sectionally, carried forward between quarters.
5. **Text encoding:** Pre-trained MarketReader applied to historical news corpus, embeddings stored.
6. **Calendar features:** Computed from historical FOMC calendar, options expiration schedules, earnings dates.
7. **Train/validation/test split:**
   - Train: 2020-01 to 2024-06 (4.5 years)
   - Validation: 2024-07 to 2024-12 (6 months)
   - Test: 2025-01 to 2025-06 (6 months, never seen during any training phase)

### 9.2 Phase 1 — Self-Supervised Pre-training

**Objective:** Teach the backbone to understand market dynamics without any portfolio optimization signal.

**Tasks (multi-task):**
1. **Next-step return prediction** (per asset): MSE between predicted and actual 1-minute returns. Weight: 0.5.
2. **Volatility prediction** (per asset): MSE between predicted and actual 5-minute realized volatility. Weight: 0.3.
3. **Cross-asset correlation prediction:** MSE between predicted and actual rolling 20-minute pairwise correlation changes. Weight: 0.2.

**Training details:**

| Hyperparameter | Value |
|---|---|
| Sequence length | 1,024 minutes (~2.6 trading days) |
| Batch size | 32 sequences (processed in parallel on H100) |
| Optimizer | AdamW (β1=0.9, β2=0.95, weight_decay=0.1) |
| Learning rate | 3e-4, cosine schedule with 2,000 warmup steps |
| Precision | BF16 mixed precision |
| Gradient clipping | 1.0 |
| Training steps | ~100,000 (multiple epochs over training data) |
| Estimated time | ~80 H100-hours |

**Note:** During this phase, training uses full-precision weights. No quantization yet. The GDN blocks operate in their parallel (chunked) mode during training (processing full 1,024-step sequences in parallel), not in the recurrent mode used at inference.

The **Flash Linear Attention (FLA)** library provides production-ready Triton kernels for the Gated DeltaNet's parallel training mode, enabling efficient H100 utilization with chunked sequences.

### 9.3 Phase 2 — Portfolio Optimization Training

**Objective:** Train the output heads to produce allocation weights that maximize portfolio performance.

**Two sub-phases run sequentially, sharing the backbone:**

#### Phase 2a — Meridian-R (Return Head)

**Loss function:**
```
L_R = -Σ_t log(1 + r_portfolio_t) + λ_turnover × Σ_t |w_t - w_{t-1}|₁
```

Where:
- `r_portfolio_t = Σ_i w_{i,t} × r_{i,t+1}` (portfolio return at time t)
- λ_turnover = 0.001 (penalizes excessive trading)

**Training:** The backbone is fine-tuned end-to-end along with the return head.

#### Phase 2b — Meridian-S (Sharpe Head)

**Loss function:** Differential Sharpe Ratio (DSR):

```
A_t = A_{t-1} + η × (r_t - A_{t-1})         — EMA of returns
B_t = B_{t-1} + η × (r_t² - B_{t-1})        — EMA of squared returns

DSR_t = (B_{t-1} × ΔA_t - 0.5 × A_{t-1} × ΔB_t) / (B_{t-1} - A_{t-1}²)^(3/2)

L_S = -Σ_t DSR_t + λ_turnover × Σ_t |w_t - w_{t-1}|₁
```

Where η = 0.01 (EMA decay). The DSR decomposes the Sharpe ratio into per-timestep contributions, enabling online optimization without batch-level ratio instability.

**Training:** Only the Sharpe head is trained. The backbone is frozen from Phase 2a. This prevents the Sharpe objective from degrading the backbone's return-prediction capability.

**Additional auxiliary loss (both heads):**

```
L_volatility = MSE(predicted_portfolio_vol, actual_portfolio_vol)
```

This volatility auxiliary task has been shown to significantly boost portfolio Sharpe ratio (Ong & Herremans, 2023).

**Combined loss for Phase 2a:**
```
L = L_R + 0.3 × L_volatility + 0.1 × L_balance (MoE load balancing)
```

| Hyperparameter | Value |
|---|---|
| Sequence length | 2,048 minutes (~5.3 trading days) |
| Batch size | 16 sequences |
| Optimizer | AdamW (β1=0.9, β2=0.95, weight_decay=0.01) |
| Learning rate | 1e-4, cosine schedule with 500 warmup steps |
| Phase 2a steps | ~50,000 |
| Phase 2b steps | ~30,000 |
| Estimated time | ~60 H100-hours (both sub-phases) |

### 9.4 Phase 3 — Meta-Learning for Plasticity

**Objective:** Train the model to adapt quickly to regime changes, optimizing the plasticity mechanisms themselves.

**Method:** MAML-style (Model-Agnostic Meta-Learning) episodic training:

```
For each meta-training episode:
  1. Sample a "regime transition" from training data:
     - 500 steps of regime A (e.g., bull market)
     - followed by 500 steps of regime B (e.g., crisis)
  
  2. Inner loop (simulate online adaptation):
     - Run inference on regime A, accumulating state
     - At regime transition, begin tracking surprise and plasticity updates
     - Run 100 steps of regime B with Tier 1 + Tier 2 adaptation active
  
  3. Outer loop (optimize adaptation speed):
     - Evaluate portfolio performance on the LAST 200 steps of regime B
     - Backpropagate through the inner loop to optimize:
       - γ (surprise sensitivity in SGP)
       - η_norm (normalization adaptation learning rate)
       - α/β regime conditioning weights
       - surprise_ema decay rate
  
  4. The meta-gradient tells us: "if the model had adapted THIS way during
     the regime transition, it would have performed THAT much better."
```

**Regime transition sampling:** Identify regime boundaries in historical data using the pre-fit HMM. Sample transitions between all 4 regime pairs (12 ordered pairs × multiple instances each).

| Hyperparameter | Value |
|---|---|
| Episode length | 1,000 minutes |
| Inner loop steps | 100 (simulated adaptation) |
| Outer loop optimization | Adam, lr=1e-5 |
| Meta-training episodes | ~10,000 |
| Estimated time | ~40 H100-hours |

This phase is the most computationally expensive per step (due to second-order gradients through the inner loop) but has the fewest steps.

### 9.5 Phase 4 — Quantization-Aware Training (QAT)

**Objective:** Fine-tune the full-precision model with ternary weight constraints to minimize quantization loss.

**Method:** BitNet b1.58 QAT:

```
For each training step:
  1. Forward pass with quantized weights:
     W_ternary = absmean_quantize(W)
     activations quantized to INT8 per-token
  
  2. Compute loss (same as Phase 2, using both heads)
  
  3. Backward pass with Straight-Through Estimator:
     Gradients pass through quantization as if it weren't there
     Full-precision shadow weights updated by optimizer
```

**Key:** The hidden state S remains in FP16 throughout — only the weight matrices are ternarized.

**Starting point:** Best checkpoint from Phase 3.

| Hyperparameter | Value |
|---|---|
| Sequence length | 1,024 minutes |
| Batch size | 32 |
| Optimizer | AdamW (β1=0.9, β2=0.95, weight_decay=0.01) |
| Learning rate | 1e-5 (much smaller — fine-tuning) |
| Training steps | ~20,000 |
| Estimated time | ~20 H100-hours |

**Validation criteria:** On the validation set, the QAT model's portfolio return must be within 5% of the full-precision model's, and the Sharpe ratio within 10%.

### Total Training Budget

| Phase | H100-Hours | Purpose |
|---|---|---|
| Phase 1 | ~80 | Self-supervised pre-training |
| Phase 2 | ~60 | Portfolio optimization (both heads) |
| Phase 3 | ~40 | Meta-learning for plasticity |
| Phase 4 | ~20 | Quantization-aware training |
| MarketReader | ~10 | Text encoder pre-training |
| HMM fitting + data prep | ~5 | Regime detector initialization |
| **Total** | **~215** | Well within 500-hour budget |

---

## 10. Two-Head Strategy (Meridian-R / Meridian-S)

The shared backbone feeds two output heads with different optimization objectives:

| Aspect | Meridian-R | Meridian-S |
|---|---|---|
| Objective | Maximize cumulative return | Maximize risk-adjusted return |
| Loss function | Negative log return + turnover penalty | Negative DSR + turnover penalty |
| Training | End-to-end with backbone (Phase 2a) | Sharpe head only, backbone frozen (Phase 2b) |
| Expected behavior | Aggressive in trending markets, larger drawdowns | More conservative, balanced, lower drawdowns |
| Use case | High-conviction directional bets | Steady portfolio growth |
| Auxiliary task | Volatility prediction (shared) | Volatility prediction (shared) |

**At deployment, only ONE head is active at any time.** The operator configures which head to use, or can implement a simple meta-strategy:

```
if current_regime == Crisis:
    use Meridian-S (risk-adjusted, defensive)
else:
    use Meridian-R (return-maximizing, aggressive)
```

Or more sophisticated: blend the two heads' outputs with regime-dependent weights:

```
final_weights = (1 - risk_aversion) × weights_R + risk_aversion × weights_S
risk_aversion = f(VIX, drawdown_depth, regime_posterior[Crisis])
```

---

## 11. Inference Runtime on Pi 5

### Per-Minute Inference Pipeline

```
Time 0ms: New minute tick arrives (from finance-query PriceStream)
  │
  ├─ [ASYNC, Core 3] Data pipeline:
  │   Fetch OHLCV, compute indicators, update rolling stats
  │   Fetch news/RSS (if available), run MarketReader
  │   Duration: ~50–200ms (network-bound, not on critical path)
  │
  ├─ [Core 0-2] Model inference (begins when features ready):
  │
  │   Step 1: Feature assembly (0.1ms)
  │     Assemble 100 × 112 feature vectors
  │     Assemble market (24), portfolio (201), calendar (20) vectors
  │
  │   Step 2: Input projection + regime detection (0.3ms)
  │     VSN gating, linear projections, regime MLP
  │
  │   Step 3: Per-asset RA-GDN backbone (3–5ms)
  │     8 GDN blocks × 100 assets, single-step recurrent mode
  │     3 FMoE blocks with regime routing
  │     Ternary GEMV via NEON bitwise kernels
  │
  │   Step 4: Temporal cascade (0.1ms, when active)
  │     Hourly/daily/weekly GDN updates (only at their cadence)
  │     Context injection into backbone
  │
  │   Step 5: Cross-asset fusion (0.5ms)
  │     110-token attention, 2 layers
  │     FAISS episodic memory retrieval
  │
  │   Step 6: Output heads (0.05ms)
  │     Active head produces 100 allocation weights
  │
  │   TOTAL INFERENCE: ~4–6ms
  │
  ├─ [ASYNC, Core 2] Plasticity update (5–10ms)
  │   Observe actual returns from t-1
  │   Compute surprise signal
  │   Tier 1: β modulation (already in forward pass)
  │   Tier 2: Norm adaptation (every 60 steps)
  │
  └─ Output: 100 allocation weights → agentic layer (abstracted)
```

### Compute Budget Analysis

| Component | Operations | Ternary GOPS | Time @ 150 GOPS |
|---|---|---|---|
| Input projection (100 assets) | ~100 × 112 × 768 = 8.6M | Yes | 0.06ms |
| RA-GDN backbone (100 assets × 8 layers) | ~100 × 8 × 3.5M = 2.8B | Yes | 1.9ms |
| FMoE (100 assets × 3 blocks × 2 experts) | ~100 × 3 × 2 × 1.8M = 1.1B | Yes | 0.7ms |
| Temporal cascade (when active) | ~100 × 3 × 1.2M = 360M | Yes | 0.2ms |
| Cross-asset attention (110 tokens × 2 layers) | ~2 × 110² × 768 + 2 × 110 × 768² = 132M | INT8/FP16 | 0.4ms |
| Episodic memory (FAISS) | 50K × 128 float16 comparisons | FP16 | 0.5ms |
| Output heads | ~100 × 768 × 256 × 2 = 39M | FP16 | 0.1ms |
| **Total** | **~4.4B** | | **~3.9ms** |

**Conservative estimate with memory access overhead: ~6ms total.**

At 1 inference per minute (60,000ms budget), CPU utilization is **0.01%**. The Pi is essentially idle between inference steps.

---

## 12. Memory Budget

### Static Allocation (always in memory)

| Component | Size | Notes |
|---|---|---|
| Ternary weights (backbone + MoE + cascade) | 16.4 MB | 2-bit packed + FP16 scales |
| INT8 weights (regime detector + MarketReader) | 3.7 MB | |
| FP16 weights (output heads + projections) | 1.1 MB | |
| **Total model weights** | **~21.2 MB** | |
| GDN hidden states (100 assets × 8 layers × 4 heads × 64 × 192 × 2 bytes) | 75.5 MB | FP16, per-asset per-layer |
| Temporal cascade states (100 assets × 3 levels) | 16.4 MB | FP16 |
| Conv1d caches (100 assets × 11 layers × 3 × D × 2 bytes) | 3.9 MB | FP16, 3 cached values per layer |
| Episodic memory buffer (50K entries) | 13.0 MB | Embeddings + metadata |
| FAISS index | 6.4 MB | IndexFlatL2 over 128-dim FP16 |
| Rolling statistics (z-scores, EMAs) | 2.0 MB | Per-asset normalization state |
| HMM parameters | 0.1 MB | 4-state Gaussian HMM |
| **Total static** | **~138.5 MB** | |

### Dynamic Allocation (during inference step)

| Component | Size | Notes |
|---|---|---|
| Activation buffers | ~2.0 MB | 100 assets × D per layer, INT8 |
| Cross-asset attention KV | ~0.2 MB | 110 tokens × 768 × 2 |
| Feature assembly buffer | ~0.1 MB | 100 × 112 × 4 bytes |
| **Total dynamic** | **~2.3 MB** | |

### Periodic Allocation (Tier 3 consolidation only)

| Component | Size | Notes |
|---|---|---|
| Fisher diagonal | 56 MB | Per-parameter FP32 |
| Parameter checkpoint | 21 MB | Copy of model weights |
| Replay buffer | 11 MB | 4 hours of features |
| LoRA adapters + gradients | 1.2 MB | Rank-8 INT8 |
| **Total Tier 3** | **~89 MB** | Allocated during market close only |

### Grand Total

| Scenario | Memory |
|---|---|
| Normal inference | ~141 MB |
| During Tier 3 consolidation | ~230 MB |
| OS + Python + data pipeline | ~500 MB |
| **Peak usage** | **~730 MB** |
| **Available (8GB Pi 5)** | **8,192 MB** |
| **Headroom** | **~7,462 MB (91%)** |

---

## 13. Development Roadmap

### Phase 0: Foundation (Weeks 1–4)

- [ ] Set up training infrastructure: H100 cloud instance, PyTorch + FLA library
- [ ] Implement data pipeline: historical data download, preprocessing, memory-mapped storage
- [ ] Implement feature engineering from finance-query (all 112 per-asset features)
- [ ] Train and validate MarketReader text encoder
- [ ] Fit 4-state HMM on historical regime data
- [ ] Build custom BPE tokenizer for MarketReader

### Phase 1: Core Model (Weeks 5–10)

- [ ] Implement RA-GDN block in PyTorch (extending FLA's Gated DeltaNet)
- [ ] Implement regime conditioning (α bias, β scaling)
- [ ] Implement FMoE with external regime routing
- [ ] Implement temporal cascade (hourly, daily, weekly GDNs)
- [ ] Implement cross-asset attention with sector bias
- [ ] Implement VSN input gating
- [ ] Run Phase 1 training (self-supervised pre-training)
- [ ] Validate: prediction accuracy, state stability, regime detection quality

### Phase 2: Portfolio Training (Weeks 11–14)

- [ ] Implement portfolio output heads with leverage normalization
- [ ] Implement DSR loss function
- [ ] Implement turnover penalty and volatility auxiliary task
- [ ] Run Phase 2a (return head) and Phase 2b (Sharpe head)
- [ ] Backtest on validation set using finance-query's backtesting engine
- [ ] Compare against baselines: buy-and-hold SPY, equal-weight, risk parity

### Phase 3: Adaptation (Weeks 15–18)

- [ ] Implement surprise-gated plasticity (Tier 1)
- [ ] Implement normalization adaptation (Tier 2)
- [ ] Implement EWC + LoRA consolidation (Tier 3)
- [ ] Run Phase 3 meta-learning training
- [ ] Validate: measure adaptation speed on held-out regime transitions
- [ ] Stress test: simulate market crash, flash crash, Fed surprise scenarios

### Phase 4: Quantization & Deployment (Weeks 19–24)

- [ ] Implement BitNet b1.58 QAT
- [ ] Run Phase 4 QAT training
- [ ] Validate: quantized model performance within tolerance of full-precision
- [ ] Implement NEON ternary inference kernel in C (based on bitnet.cpp / TAB)
- [ ] Implement FAISS episodic memory on ARM
- [ ] Implement full inference pipeline on Pi 5
- [ ] Run 10M+ step state drift test (Section 14, Risk #1)
- [ ] Implement data pipeline (finance-query integration, async feature assembly)
- [ ] End-to-end integration test: live data → inference → allocation weights
- [ ] Thermal and power testing: 72-hour continuous operation

### Phase 5: Live Operation (Weeks 25+)

- [ ] Paper trading with both heads
- [ ] Monitor state drift, surprise distribution, regime detection accuracy
- [ ] Tune Tier 2/3 adaptation hyperparameters based on live performance
- [ ] Build monitoring dashboard (state norms, surprise levels, allocation drift)
- [ ] Implement alerting for anomalous model behavior

---

## 14. Risk Analysis & Mitigations

### Risk #1: Ternary State Drift (HIGH PRIORITY)

**Risk:** FP16 hidden state S accumulates quantization errors during 24/7 operation. After millions of steps, the state may drift into meaningless regions.

**Mitigation:**
1. Periodic RMSNorm every 60 steps (Section 8)
2. Magnitude monitoring with emergency normalization
3. Daily soft reset toward checkpoint state
4. **Prototype test:** Before full model training, run a standalone ternary GDN block for 10M+ recurrent steps with synthetic data. Measure state drift magnitude, prediction accuracy over time, and identify the failure horizon. If drift exceeds tolerance within 24 hours, increase state precision to FP32 (costs ~37 MB additional memory — still within budget).

### Risk #2: Catastrophic Forgetting in Tier 3

**Risk:** EWC + LoRA consolidation may not preserve knowledge of rare regimes (e.g., 2020 COVID crash dynamics) after months of adaptation to benign markets.

**Mitigation:**
1. EWC's Fisher diagonal anchors parameters near their trained values
2. Maintain a "crisis replay buffer" — a permanent set of 500 crisis-period samples that are always included in Tier 3 consolidation batches
3. Monitor per-regime auxiliary prediction accuracy; if any regime's accuracy drops below a threshold, increase its replay weight

### Risk #3: Regime Detector Miscalibration

**Risk:** The HMM/neural regime detector may misclassify market conditions, routing to the wrong MoE expert.

**Mitigation:**
1. Blended routing (70% neural, 30% HMM) provides redundancy
2. Use top-2 experts with soft weights, not hard top-1 routing — even if the top expert is wrong, the second expert provides partial coverage
3. During sustained misrouting (detected by elevated surprise levels with no adaptation), fall back to equal-weight all experts (effectively a dense FFN)

### Risk #4: Overfitting to Historical Regimes

**Risk:** Training on 5 years of data may not capture all future market dynamics.

**Mitigation:**
1. Meta-learning (Phase 3) explicitly optimizes for rapid adaptation, not memorization
2. Continuous learning system provides real-time adaptation
3. Ternary quantization acts as a regularizer (reduced model capacity prevents overfitting)
4. Data augmentation: time-stretch, amplitude-scale, and synthetic regime transitions during training

### Risk #5: Data Pipeline Latency

**Risk:** finance-query API calls may occasionally exceed the 60-second minute budget, causing missed inference steps.

**Mitigation:**
1. Use `PriceStream` for real-time prices (sub-second latency)
2. Cache all features with staleness tracking; use stale features if fresh ones aren't available within 5 seconds
3. Indicators are computed locally from cached OHLCV — no API dependency after initial data load
4. Fundamentals and options data update infrequently (daily/quarterly) and are pre-cached

### Risk #6: Insufficient Model Capacity

**Risk:** 82M parameters (31M active) may not be sufficient to model 100 assets across 4 regimes with the required fidelity.

**Mitigation:**
1. The memory budget analysis shows 91% headroom — the model can be scaled to ~300M parameters (with proportionally more MoE experts) while staying within the Pi 5's memory budget
2. Active parameter count matters more than total: 31M active with ternary weights is capacity-equivalent to a ~10M full-precision model, which is comparable to successful financial ML models in literature
3. If capacity is insufficient, first increase MoE experts (4→8) before widening the backbone — this adds capacity without increasing per-step compute

---

## Appendix A — Parameter Count Breakdown

| Component | Parameters | Active Per Step | Quantization | Size |
|---|---|---|---|---|
| Input projections + VSN | 265K | 265K | Ternary | 66 KB |
| Regime detector | 25K | 25K | INT8 | 25 KB |
| RA-GDN blocks (8) | 17.4M | 17.4M | Ternary | 4.3 MB |
| FMoE blocks (3 × 4 experts) | 42.5M | 21.3M (2/4 active) | Ternary | 10.6 MB |
| Temporal cascade (3 GDN + projections) | 4.2M | 4.2M (when active) | Ternary | 1.1 MB |
| Cross-asset attention (2 layers) | 11.8M | 11.8M | Ternary + FP16 attn | 3.0 MB |
| Episodic memory projections | 200K | 200K | FP16 | 400 KB |
| Portfolio heads (2) | 394K | 197K (one active) | FP16 | 788 KB |
| Auxiliary prediction head | 100K | 100K | FP16 | 200 KB |
| Sector bias matrix | 121 | 121 | FP32 | 484 B |
| MarketReader text encoder | 3.4M | 3.4M (async) | INT8 | 3.4 MB |
| RMSNorm parameters (all layers) | ~22K | ~22K | FP32 | 88 KB |
| **Total** | **~80.3M** | **~31.0M** | **Mixed** | **~24.0 MB** |

Note: "Active per step" counts only parameters involved in a single forward pass (one head active, 2 of 4 MoE experts, MarketReader async). Temporal cascade parameters are active only at their respective cadences.

---

## Appendix B — finance-query Integration Map

This maps every finance-query API to its role in Meridian's data pipeline.

### Real-Time Data Path (every minute)

| API Call | Meridian Component | Features Generated |
|---|---|---|
| `PriceStream::subscribe(&symbols)` | Price & Volume Block | OHLCV (5) + returns (5) |
| `chart(Interval::OneMinute, ...)` | Fallback to PriceStream | Same as above |
| `indicators::rsi(&closes, 14)` | Technical Indicator Block | RSI (1) |
| `indicators::macd(&closes, 12, 26, 9)` | Technical Indicator Block | MACD histogram (1) |
| `indicators::bollinger_bands(&closes, 20, 2.0)` | Technical Indicator Block | BB %B (1) |
| `indicators::atr(&highs, &lows, &closes, 14)` | Technical Indicator Block | ATR normalized (1) |
| `indicators::stochastic(h, l, c, 14, 3, 3)` | Technical Indicator Block | Stochastic %K (1) |
| `indicators::sma(&closes, 20/50/200)` | Technical Indicator Block | SMA z-scores (3) |
| `indicators::adx(&highs, &lows, &closes, 14)` | Technical Indicator Block | ADX (1) |
| `indicators::roc(&closes, 10)` | Technical Indicator Block | ROC (1) |
| `indicators::obv(&closes, &volumes)` | Technical Indicator Block | OBV slope (1) |
| `indicators::cmf(&highs, &lows, &closes, &volumes, 20)` | Technical Indicator Block | CMF (1) |
| `indicators::mfi(&highs, &lows, &closes, &volumes, 14)` | Technical Indicator Block | MFI (1) |
| `indicators::vwap(&highs, &lows, &closes, &volumes)` | Technical Indicator Block | VWAP deviation (1) |
| `indicators::supertrend(&highs, &lows, &closes, ...)` | Technical Indicator Block | Direction (1) |
| `indicators::ichimoku(h, l, c, 9, 26, 26, 26)` | Technical Indicator Block | Cloud position (1) |
| `indicators::parabolic_sar(&highs, &lows, ...)` | Technical Indicator Block | SAR direction (1) |
| `indicators::keltner_channels(&highs, &lows, &closes, ...)` | Technical Indicator Block | Keltner position (1) |
| `Ticker::new("^VIX").price()` | Market-Wide Features | VIX level + change (2) |
| `finance::fear_and_greed()` | Market-Wide Features | F&G index (1) |

### Periodic Data Path

| API Call | Cadence | Meridian Component |
|---|---|---|
| `ticker.options(date)` | Daily | Options-Derived Block |
| `ticker.financial_data()` | Quarterly | Fundamental Block |
| `ticker.summary_detail()` | Quarterly | Fundamental Block |
| `ticker.key_stats()` | Quarterly | Fundamental Block |
| `ticker.asset_profile()` | Once (static) | Sector embedding |
| `ticker.calendar_events()` | Weekly | Calendar features |
| `fred::treasury_yields(year)` | Daily | Yield curve features |
| `fred::series("FEDFUNDS")` | On release | FRED macro (fed funds) |
| `fred::series("UNRATE")` | On release | FRED macro (unemployment) |
| `fred::series("CPIAUCSL")` | On release | FRED macro (CPI) |
| `fred::series("ICSA")` | On release | FRED macro (claims) |
| `finance::sector(type)` | Daily | Sector dispersion |
| `finance::market_summary(Region::US)` | Daily | Market breadth |
| `feeds::fetch_all(sources)` | Continuous | MarketReader input |
| `ticker.news()` | Continuous | MarketReader input |
| `finance::earnings_transcript(sym, q, y)` | Quarterly | MarketReader input |
| `crypto::coin(id, "usd")` | Hourly | Cross-asset features (BTC) |

### Backtesting Integration

During training data preparation and validation:

| API Call | Purpose |
|---|---|
| `ticker.chart(interval, range)` | Historical OHLCV for training data |
| `ticker.dividends(range)` | Dividend-adjusted returns |
| `ticker.splits(range)` | Split-adjusted prices |
| `ticker.indicators(interval, range)` | Batch indicator computation |
| `ticker.risk(interval, range, benchmark)` | Baseline risk metrics |
| `Tickers::backtest(...)` | Strategy comparison baselines |
| `BacktestEngine::run_with_benchmark(...)` | Alpha/beta vs benchmark |
| `MonteCarloConfig::new().run(&result)` | Confidence intervals |
| `WalkForwardConfig::new(...).run(...)` | Walk-forward validation |

---

## Appendix C — Hyperparameter Reference

### Model Architecture

| Parameter | Symbol | Value | Notes |
|---|---|---|---|
| Model dimension | D | 768 | Main backbone hidden size |
| Number of RA-GDN blocks | L | 8 | Core temporal layers |
| Number of FMoE blocks | L_moe | 3 | After blocks 2, 5, 8 |
| GDN heads | n_heads | 4 | Per RA-GDN block |
| GDN head dimension (key) | d_head | 64 | Q/K dimension per head |
| GDN head dimension (value) | d_v | 192 | V dimension per head |
| GDN conv kernel | k_conv | 4 | Short-range temporal context |
| MoE experts | n_experts | 4 | Per FMoE block |
| MoE active experts | k_active | 2 | Top-2 routing |
| MoE FFN expansion | r_ffn | 2× | D → 2D → D per expert |
| Regime embedding dim | d_regime | 64 | |
| Regime classes | n_regimes | 4 | Low-vol, high-vol, mean-rev, crisis |
| Temporal cascade dims | D_hourly, D_daily, D_weekly | 512, 512, 384 | |
| Cross-asset attention heads | n_attn_heads | 8 | d_head = 96 |
| Cross-asset attention layers | L_attn | 2 | |
| Episodic memory slots | n_memory | 50,000 | FAISS buffer |
| Episodic memory dim | d_memory | 128 | float16 |
| Episodic retrieval count | k_retrieve | 8 | Appended to attention |
| MarketReader dim | D_text | 256 | |
| MarketReader layers | L_text | 4 | |
| MarketReader vocab | V_text | 4,096 | BPE |
| MarketReader output dim | d_text_out | 64 | Per-asset text embedding |
| Max tickers | N_assets | 100 | US equities + ETFs + crypto |

### Training

| Parameter | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW | Adam | AdamW |
| Learning rate | 3e-4 | 1e-4 | 1e-5 | 1e-5 |
| Weight decay | 0.1 | 0.01 | 0 | 0.01 |
| β1, β2 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.99 | 0.9, 0.95 |
| Batch size | 32 | 16 | 8 | 32 |
| Sequence length | 1,024 | 2,048 | 1,000 | 1,024 |
| Training steps | 100K | 80K | 10K | 20K |
| Grad clip | 1.0 | 1.0 | 0.5 | 1.0 |
| Precision | BF16 | BF16 | BF16 | QAT |

### Online Adaptation

| Parameter | Value | Notes |
|---|---|---|
| Surprise EMA span | 60 minutes | Running baseline for surprise |
| Surprise sensitivity (γ) | 0.2 (learned) | Modulates β in SGP |
| Surprise clamp | [0, 5] | Prevents runaway updates |
| Norm adaptation LR (η_norm) | 1e-3 | Tier 2 |
| Norm adaptation cadence | 60 minutes | Tier 2 |
| Norm drift penalty (λ_drift) | 0.1 | Tier 2 |
| LoRA rank | 8 | Tier 3 |
| LoRA adaptation LR | 1e-4 | Tier 3 |
| EWC penalty (λ_ewc) | 1000 | Tier 3 |
| Fisher EMA decay | 0.9 | Tier 3 |
| Consolidation cadence | Daily (market close) | Tier 3 |
| Replay buffer duration | 4 hours | Tier 3 |
| Crisis replay buffer | 500 samples | Permanent, Tier 3 |
| State RMSNorm cadence | 60 steps | Drift mitigation |
| Daily state decay (δ) | 0.95 | Soft reset toward checkpoint |

### Portfolio

| Parameter | Value | Notes |
|---|---|---|
| Leverage limit | 2.0 | Sum of |weights| ≤ 2.0 |
| Turnover penalty (λ_turnover) | 0.001 | In loss function |
| DSR EMA decay (η) | 0.01 | Differential Sharpe Ratio |
| Volatility aux weight | 0.3 | Multi-task loss |
| MoE balance loss weight | 0.1 | Load balancing |
| HMM/neural blend | 0.7/0.3 | Regime routing |

---

*Meridian v2.0 — Harvey, March 2026*
*Built on Gated DeltaNet (ICLR 2025), BitNet b1.58, and finance-query.*
