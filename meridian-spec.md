# MERIDIAN: Regime-Adaptive State-Space Portfolio Model

## Architectural Specification v4.0 — Data-Constrained Full Integration

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
   - 6.6 [Portfolio State Vector (2·n_assets + 1 dims)](#66-portfolio-state-vector-2n_assets--1-dims)
   - 6.7 [Financial Calendar Embedding (20 dims)](#67-financial-calendar-embedding-20-dims)
7. [Model Components](#7-model-components)
   - 7.1 [Input Projection Layer with Variable Selection](#71-input-projection-layer-with-variable-selection)
   - 7.2 [Regime Detector](#72-regime-detector)
   - 7.3 [Per-Asset Temporal Backbone (Hybrid RA-GDN + SWA Stack)](#73-per-asset-temporal-backbone-hybrid-ra-gdn--swa-stack)
   - 7.4 [Regime-Routed Sparse Mixture of Experts](#74-regime-routed-sparse-mixture-of-experts)
   - 7.5 [Temporal Cascade](#75-temporal-cascade)
   - 7.6 [Cross-Asset Fusion Module](#76-cross-asset-fusion-module)
   - 7.7 [Financial Episodic Memory](#77-financial-episodic-memory)
   - 7.8 [Portfolio Output Heads](#78-portfolio-output-heads)
   - 7.9 [Auxiliary Prediction Head](#79-auxiliary-prediction-head)
   - 7.10 [MarketReader Text Encoder (distilled, optional)](#710-marketreader-text-encoder-distilled-optional)
8. [Financial Diffusion Augmentation (FDA)](#8-financial-diffusion-augmentation-fda)
   - 8.1 [Motivation & Scope](#81-motivation--scope)
   - 8.2 [FDA Architecture](#82-fda-architecture)
   - 8.3 [Stylized Fact Validation Gate](#83-stylized-fact-validation-gate)
   - 8.4 [Regime-Conditioned Generation](#84-regime-conditioned-generation)
   - 8.5 [Cross-Asset Correlation Preservation](#85-cross-asset-correlation-preservation)
   - 8.6 [Limitations & Honest Caveats](#86-limitations--honest-caveats)
9. [Continuous Learning System](#9-continuous-learning-system)
   - 9.1 [Tier 1 — Surprise-Gated Delta Dynamics](#91-tier-1--surprise-gated-delta-dynamics)
   - 9.2 [Tier 2 — Normalization Adaptation](#92-tier-2--normalization-adaptation)
   - 9.3 [Tier 3 — Periodic Consolidation (EWC + Ternary LoRA)](#93-tier-3--periodic-consolidation-ewc--ternary-lora)
   - 9.4 [Adaptation Feedback Loop](#94-adaptation-feedback-loop)
10. [Quantization Strategy](#10-quantization-strategy)
11. [Training Pipeline](#11-training-pipeline)
    - 11.1 [Data Sources & Honest Row Counts](#111-data-sources--honest-row-counts)
    - 11.2 [Phase 0 — FDA Pre-Training (Diffusion Model)](#112-phase-0--fda-pre-training-diffusion-model)
    - 11.3 [Phase 1 — Contrastive Self-Supervised Pre-training](#113-phase-1--contrastive-self-supervised-pre-training)
    - 11.4 [Phase 2 — Portfolio Optimization with Augmentation](#114-phase-2--portfolio-optimization-with-augmentation)
    - 11.5 [Phase 3 — Meta-Learning for Plasticity](#115-phase-3--meta-learning-for-plasticity)
    - 11.6 [Phase 4 — Quantization-Aware Training](#116-phase-4--quantization-aware-training)
12. [Known Limitations & Expected Performance Bounds](#12-known-limitations--expected-performance-bounds)
13. [Two-Head Strategy (Meridian-R / Meridian-S)](#13-two-head-strategy-meridian-r--meridian-s)
14. [Historical Backtesting Framework](#14-historical-backtesting-framework)
    - 14.1 [Meridian as a Backtesting Strategy](#141-meridian-as-a-backtesting-strategy)
    - 14.2 [Benchmark Comparisons](#142-benchmark-comparisons)
    - 14.3 [Walk-Forward Validation](#143-walk-forward-validation)
    - 14.4 [Monte Carlo Confidence Intervals](#144-monte-carlo-confidence-intervals)
    - 14.5 [Regime-Segmented Analysis](#145-regime-segmented-analysis)
15. [Paper Trading System](#15-paper-trading-system)
    - 15.1 [Architecture](#151-architecture)
    - 15.2 [Performance Tracking](#152-performance-tracking)
    - 15.3 [Adaptation Monitoring](#153-adaptation-monitoring)
    - 15.4 [Graduation Criteria (Paper → Live)](#154-graduation-criteria-paper--live)
16. [Inference Runtime](#16-inference-runtime)
17. [Memory Budget](#17-memory-budget)
18. [Development Roadmap](#18-development-roadmap)
19. [Risk Analysis & Mitigations](#19-risk-analysis--mitigations)
20. [Appendix A — Parameter Count Breakdown](#appendix-a--parameter-count-breakdown)
21. [Appendix B — Complete finance-query Integration Map](#appendix-b--complete-finance-query-integration-map)
22. [Appendix C — Hyperparameter Reference](#appendix-c--hyperparameter-reference)
23. [Appendix D — Event Type Catalog](#appendix-d--event-type-catalog)
24. [Appendix E — FDA Stylized Fact Checklist](#appendix-e--fda-stylized-fact-checklist)

---

## 1. Executive Summary

Meridian is a neural network architecture designed to manage a portfolio of US-listed assets (equities, ETFs, crypto) 24/7 on a single small x86-64 host. The reference deployment target is a **Framework Desktop (AMD Ryzen AI Max+ 395, 128 GB unified memory)**; consistent with the project's quality-over-efficiency priority, the model runs in full-precision **FP32** (FP16 optional) with no quantization. The tradable universe is **user-controlled and dynamic**: a fixed maximum of `n_assets` slots (default 100) is the static, trained asset-axis dimension, and the live universe may be any subset of size 1..`n_assets`. Users add/remove symbols by editing `portfolio_symbols.txt` (hot-reloaded live); padded slots are masked out of cross-asset attention and the portfolio softmax, and the model is trained on random universe subsets so it generalizes to arbitrary membership. It ingests every available data stream from finance-query — price, volume, 42 technical indicators, 30+ RSS/news feeds, full options chains, EDGAR filings, XBRL fundamentals, earnings transcripts, analyst ratings, insider transactions, institutional ownership, macro series, yield curves, sector/industry data, Fear & Greed, world indices, trending tickers, and crypto market data — producing allocation weights every minute through a shared backbone with two output heads.

**v4.0 addresses a fundamental constraint identified during v3.0 implementation:** training and inference must use the same data source (finance-query) to avoid distribution mismatch. This eliminates paid minute-bar providers (Polygon.io, FirstRate Data) as training data sources. With finance-query's free tier, the available training corpus is approximately **1,695 daily bars per equity asset** — roughly 290× less than the spec's original ~490K minute-bar assumption. At a ~95M-parameter target, this corresponds to ~56,000:1 in the 100-asset sequence view (1,695 joint samples) and ~560:1 in the asset-day view (169,500 observations), which drives early saturation and limits OOS Sharpe to approximately 0.57 in the current baseline.

**v4.0 addresses this through three research-backed mechanisms that require no external data:**

1. **Financial Diffusion Augmentation (FDA)** — a regime-conditioned score-based diffusion model trained on real finance-query data that generates statistically valid synthetic training sequences, increasing effective training volume by up to 30× while preserving all stylized facts (fat tails, volatility clustering, cross-asset correlations).

2. **Contrastive Self-Supervised Pretraining** — replaces the next-step return prediction objective in Phase 1 with a temporal contrastive objective that extracts up to 100× more training signal from the same data by treating temporally adjacent windows as positive pairs across the full 100-asset universe.

3. **Portfolio-Theoretic Augmentation (PTA)** — applies label-preserving symmetry transformations during portfolio head training (asset permutations preserving covariance structure, sign-coupled return flips) that multiply effective portfolio training samples by 10–50× without introducing distributional drift.

Additionally, the **RA-GDN backbone is augmented with interleaved Sparse Sliding Window Attention (SWA)** layers (Samba-style hybrid), giving the model precise short-sequence recall capability that reduces the data volume needed to achieve generalization.

**Empirically validated baseline (v3.0 with daily data only):**

| Metric | Result | Spec Target | Pass? |
|---|---|---|---|
| Sharpe (OOS 2024+) | 0.57 | > 1.0 | No |
| Total Return | +15.18% | Beat passive | Yes (vs −9.21%) |
| Max Drawdown | 13.05% | < 25% | Yes |
| Win Rate | 67.25% | — | — |
| MC Pass Rate | 73/100 | p5 > 0, Sharpe p25 > 0.5 | Partial |

**v4.0 expected performance range (projected, not guaranteed):**

| Metric | v3.0 Actual | v4.0 Target | Mechanism |
|---|---|---|---|
| Sharpe (OOS) | 0.57 | 0.82–1.24 (projected) | FDA + PTA + contrastive |
| Max Drawdown | 13.05% | < 20% | Improved generalization |
| Win Rate | 67.25% | 68–72% | Better backbone recall |

The Sharpe > 1.0 target remains achievable but is not guaranteed with daily data alone. Section 12 (Known Limitations) documents the honest performance bounds.

Key metrics:

| Metric | Value |
|---|---|
| Total parameters | ~95–96M target architecture (~35M active); ~60.9M currently implemented in codebase |
| Input features per asset | 120 scalar + 64 text embedding = 184 |
| Market-wide features | 96 scalar |
| Event types tracked | 18 discrete event categories |
| Text sources ingested | 30+ RSS feeds + per-ticker news + transcripts |
| FDA synthetic multiplier | Up to 30× generated windows before effective-sample discount (planned) |
| Model size | ~240 MB (FP32, default) / ~120 MB (FP16 option) — not quantized |
| Runtime memory footprint | ~1.5 GB process (FP32, measured) — trivial against 128 GB unified |
| Inference latency (100 assets) | ~145 ms/step (FP32 CPU) / ~35 ms (FP16 iGPU), measured — ≪ 60,000 ms per-minute budget |
| Inference precision | FP32 (quality-first default); no quantization |
| Data pipeline latency | ~200–500 ms (network-bound) |
| Adaptation mechanisms | 3-tier (sub-second / hourly / daily) |
| Training compute | ~180–280 H100-hours |
| Training data source | finance-query exclusively |

### 1.1 Validation and feasibility status (April 2026)

| Area | Validation verdict | Feasibility status |
|---|---|---|
| Data-source fidelity (train = inference source) | **Validated** (internally consistent and code-aligned) | **Feasible now** — daily finance-query flows already exist in collection/backtesting paths |
| Dataset arithmetic (rows/windows/ratios) | **Validated with corrections** (ratios harmonized in this spec) | **Feasible now** for daily-data training; does not remove small-sample risk |
| FDA (diffusion augmentation) | **Research-grounded but unverified in this repo** | **Feasible with additional implementation** (new model + validation gate pipeline required) |
| TCP (contrastive pretraining) | **Methodologically plausible** | **Feasible with additional implementation** (current pretraining still uses predictive objectives) |
| PTA (portfolio-theoretic augmentation) | **Methodologically plausible** | **Feasible with additional implementation** (current portfolio trainer lacks these transforms) |
| Hybrid RA-GDN + SWA | **Architecturally plausible** | **Partially feasible** — RA-GDN exists; SWA layers/caches are not yet present in code |
| Performance targets | **Projected only** (not guaranteed) | Requires FDA/TCP/PTA/SWA implementation plus clean OOS evaluation |

---

## 2. Design Philosophy

Seven interlocking principles govern every architectural decision in v4.0. **Quality over efficiency (below) is the tie-breaker whenever two principles conflict.**

**Information maximalism.** The model ingests every available signal from finance-query — not just price and volume, but the full constellation of fundamental, sentiment, ownership, regulatory, macro, and cross-asset data that professional portfolio managers use. The Variable Selection Network ensures irrelevant signals are gated to zero rather than adding noise.

**Temporal nativeness.** Financial markets are continuous processes with multi-scale dynamics. The hybrid RA-GDN + SWA backbone models time at two complementary scales: the recurrent state captures infinite-horizon compressed history in O(1) per step, while sliding window attention captures precise short-range episodic recall. Both are necessary — recurrence alone cannot recall exact events from even 20 steps ago without sufficient training data; attention alone cannot scale to infinite history.

**Event-driven proactivity.** Markets move on discrete events: an insider purchase, an analyst upgrade, a Fed statement, an earnings miss. Meridian processes events as first-class inputs — not by waiting for their price impact, but by incorporating the event signal immediately when it arrives. The model learns the typical price dynamics following each event type and can position proactively.

**Regime awareness.** Markets change. Meridian addresses this at three levels: recurrent dynamics adapt based on detected regime, MoE routes to specialized sub-networks, and continuous learning updates parameters using prediction errors. The FDA generates synthetic data conditioned on all four regimes, preventing the regime imbalance problem (crisis regimes are rare in 1,695 rows).

**Data source fidelity.** Training data must come from the same source as inference data. No paid third-party price provider is used for training. All price and indicator data is sourced from finance-query, ensuring z-score baselines, volume normalizations, and indicator values are computed consistently between training and live inference. This constraint is architecturally enforced: the FDA is trained on finance-query data and generates sequences in the finance-query feature space.

**Quality over efficiency.** This is the project's governing priority: predictive and risk-adjusted quality is maximized first, and compute/memory cost is spent freely in its service. The reference hardware (Framework Desktop, 128 GB unified) makes the former Pi-era budget non-binding — the full 100-asset backbone runs in ~145 ms/step (FP32 CPU) / ~35 ms/step (FP16 iGPU), ≪ the 60,000 ms per-minute budget — so every prior quality-vs-budget tradeoff resolves toward quality. Concretely: **FP32 inference** (no quantization rounding), **full teacher text models** run in-process (no distillation loss), **exact k-NN episodic memory**, and **spending inference compute** on blended heads / always-on temporal cascade / test-time augmentation. The efficiency-motivated primitives that *also* aid quality are retained on their own merit — O(1) recurrence and sliding window attention (W=64) capture complementary time scales — but no accuracy is ever traded for size or speed. Quantization (ternary/INT8) and the distilled-student encoder are **not used**; they survive only as a documented fallback for hypothetical resource-constrained deployments.

**Separation of understanding from strategy.** The backbone learns market state. The output heads learn allocation strategy. This enables modular head swapping and backtesting flexibility. The FDA augments both backbone pretraining and portfolio head training independently, with different augmentation strategies appropriate to each phase.

---

## 3. Novel Contributions

Contributions 1–9 are carried forward from v3.0. v4.0 adds four new contributions that directly address the data scarcity constraint:

**Carried forward from v3.0:**

1. **Regime-Aware Gated DeltaNet (RA-GDN)** — regime embedding modulates forgetting gate α and update strength β
2. **Surprise-Gated Plasticity (SGP)** — delta rule's prediction error drives adaptive learning rates
3. **Temporal Cascade with Adaptive Speed** — multi-resolution GDN stacks with surprise-gated downward injection
4. **Financial Episodic Memory** — FAISS k-NN buffer of surprising market states
5. **External Regime-Routed Sparse MoE** — HMM/neural posterior drives expert selection
6. **Financial Calendar Embedding** — market-specific temporal encoding
7. **Full-precision inference (quality-first)** — the model runs in **FP32** by default; no quantization. The "precision highway" (FP16 state + ternary/INT8 weights) is retained only as a documented fallback for constrained deployments and is not part of the reference path.
8. **Multi-Modal Event Fusion** — discrete financial events encoded as typed event tokens
9. **Full teacher text models (quality-first)** — Block-8 text features come from the **full harrier-oss embedder + FinBERT sentiment teachers run in-process** at inference, for maximum fidelity. The distilled 3.5M-param MarketReader student exists only as a fallback for resource-constrained hosts where the teachers cannot run; it is not used on the reference target.

**New in v4.0:**

10. **Hybrid RA-GDN + Sparse Sliding Window Attention (Samba-Finance)** — interleaved SWA layers at positions 3 and 7 of the 8-block backbone give the model precise episodic recall for recent market events without quadratic scaling. Window size W=64 steps (64 trading days at daily resolution; 64 minutes at minute resolution). This directly reduces the data volume required for the backbone to generalize, as precise short-term recall no longer needs to be learned through recurrent compression alone.

11. **Financial Diffusion Augmentation (FDA)** — a regime-conditioned score-based diffusion model (FinDiff-style) trained exclusively on finance-query daily data that generates statistically valid synthetic price + feature sequences. Synthetic data passes a 9-point stylized fact validation gate before entering training. Increases effective training volume by up to 30×. All synthetic data is in the finance-query feature space, preserving train/inference distribution alignment.

12. **Temporal Contrastive Pretraining (TCP)** — replaces next-step return prediction in Phase 1 with a temporal contrastive objective operating over the full 100-asset universe. Adjacent windows (±5 steps) form positive pairs; distant windows (>50 steps apart) and different-asset windows in different regimes form negative pairs. Extracts ~100× more gradient signal from the same data volume compared to single-step prediction.

13. **Portfolio-Theoretic Augmentation (PTA)** — applies label-preserving symmetry transformations to portfolio training batches following Ziyin et al. (2022). Valid transformations include: asset permutations that preserve the covariance matrix block structure, simultaneous sign flips of return labels paired with weight sign flips, and scale-invariant portfolio rebalancing paths. Multiplies effective portfolio training samples by 10–50× without introducing distributional drift.

---

## 4. Hardware Target & Constraints

**Reference inference target (v4.0, revised):** Framework Desktop — AMD Ryzen AI Max+ 395 (Strix Halo): 16-core Zen 5, x86-64 with AVX-512; Radeon 8060S iGPU (RDNA 3.5, 40 CUs); XDNA 2 NPU (~50 TOPS); up to **128 GB unified LPDDR5x** shared across CPU/GPU/NPU.

**Development / validation target:** Framework Laptop 13 (AMD Ryzen AI 9 HX 370) — same Zen 5 / RDNA 3.5 / XDNA 2 architecture family with a smaller Radeon 890M iGPU and 128 GB RAM. Anything validated here transfers directly to the Desktop (same ISA, same NPU, same iGPU architecture); the Desktop only adds iGPU CUs.

**Inference precision/backend (quality-first):** **FP32 is the default** — quality is prioritized over speed/size, and at these latencies there is no reason to accept half-precision rounding. FP16 (CPU or ROCm iGPU) remains available purely as a latency option, not a default. Measured on the Framework Laptop 13 (Radeon 890M, the smaller validation iGPU), full 100-asset backbone, batch=1: FP32 CPU ~144 ms/step, FP32 iGPU ~122 ms/step, FP16 iGPU ~35 ms/step — all far inside the 60,000 ms per-minute budget, against 128 GB of memory. The Desktop's 8060S (40 CUs vs 16) is faster still. ROCm requires `HSA_OVERRIDE_GFX_VERSION=11.0.0` for these RDNA 3.5 APUs (gfx1150/gfx1151). Quantization is **not used**: INT8/ternary and the XDNA 2 NPU path would trade predictive quality for size/power that this hardware does not need.

Training on H100 cloud (500-hour budget) is unchanged.

> **Superseded:** earlier v3.0/v4.0 drafts targeted a Raspberry Pi 5 (8 GB, BCM2712 Cortex-A76, NEON SIMD). The ARM/NEON kernels, ternary quantization, hard memory budget (§17), and the distilled-only MarketReader requirement all stemmed from that 8 GB ARM constraint and no longer bind. Those sections are retained below as a constrained-deployment fallback, not the primary path.

**v4.0 additions:**
- FDA model is trained offline and runs only during training data generation, not at inference. No inference-time memory impact.
- SWA layers add ~1.5 ms to inference latency. Negligible against the per-minute budget.
- FDA training: ~40–60 H100-hours (separate from main model training budget).

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
├── Yield Curve (daily)                    SOURCE CONSTRAINT (v4.0):
├── Sector/Industry Performance (daily)    All training + inference data from
├── Earnings Transcripts (quarterly)       finance-query only. No external
├── Analyst Ratings Snapshot (daily)       price providers. Train/inference
├── Institutional Ownership (quarterly)    distributions are identical by
├── Insider Holdings Snapshot (daily)      construction.
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
         │                              │
         │                    ┌─────────▼──────────────────────────┐
         │                    │  FINANCIAL DIFFUSION AUGMENTATION  │
         │                    │  (TRAINING ONLY — offline)         │
         │                    │  Generates synthetic sequences in  │
         │                    │  finance-query feature space       │
         │                    │  Regime-conditioned generation     │
         │                    │  9-point stylized fact gate        │
         │                    │  Up to 30× real row multiplier     │
         │                    └────────────────────────────────────┘
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
│  PER-ASSET TEMPORAL BACKBONE (Hybrid RA-GDN + SWA)   │
│  Block 1: RA-GDN                                      │
│  Block 2: RA-GDN + FMoE                               │
│  Block 3: RA-GDN → SWA(W=64) [NEW]                   │
│  Block 4: RA-GDN                                      │
│  Block 5: RA-GDN + FMoE                               │
│  Block 6: RA-GDN                                      │
│  Block 7: RA-GDN → SWA(W=64) [NEW]                   │
│  Block 8: RA-GDN + FMoE                               │
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

Unchanged from v3.0. The full finance-query API surface is ingested as specified. Section 6 content is identical to v3.0 Sections 6.1–6.7, reproduced here for completeness.

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
│    options anomaly detector → options_event  │
│    trending diff → trending_event            │
└─────────────────────────────────────────────┘
```

### 6.2 Per-Asset Feature Assembly (120 dims)

Every minute, each of 100 assets gets a 120-dimensional continuous feature vector plus a 64-dimensional text embedding, for 184 effective dims total.

#### Block 1: Price & Volume (10 features)

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

Each indicator is z-scored with rolling 252-day statistics. Z-score baselines are computed from finance-query data and stored as rolling statistics — the same pipeline runs at training and inference time, ensuring no distributional mismatch.

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

Sourced from `ticker.options(date)` polled every 15 minutes.

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

For assets without listed options: filled with a **learned 12-dim default embedding** per asset class.

#### Block 4: Fundamental & Ownership (30 features)

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
| Institutional ownership % | `major_holders()` → institutional percentage | [0, 1] |
| Institutional Δ (quarterly) | `institution_ownership()` → change in top-10 total shares | Z-scored |

**Asset Identity (2 features):**

| Feature | Source | Encoding |
|---|---|---|
| Sector embedding | `asset_profile()` → GICS sector | Learned scalar per sector (11→1) |
| Market cap log-rank | `summary_detail().market_cap` | `ln(mktcap)` percentile [0, 1] |

#### Block 5: Risk Analytics (8 features)

Sourced from `ticker.risk(Interval::OneDay, TimeRange::OneYear, Some("SPY"))`, refreshed daily.

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

Each feature represents a decayed signal from the most recent event of that type.

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

| Feature | Computation | Live source | Historical source |
|---|---|---|---|
| Relative strength (20d) | Asset return − SPY return (20d) | `store.prices` (SPY price buffer) | Asset + SPY chart series |
| Sector relative strength | Asset 1d return − sector 1d return | `store.sector.returns[sector_idx]` | Asset + sector ETF chart series (20d) |
| Correlation to SPY (60d) | Rolling Pearson correlation | `store.prices` (60d window) | Asset + SPY chart series |
| Correlation change (Δ20d) | Corr today − corr 20d ago | `store.prices` offset window | Asset + SPY chart series |
| Is trending ticker | `finance::trending(Region::US)` | `store.trending_symbols` (hourly poll) | Snapshot blended with vol z-score proxy |
| Screener membership count | Fraction of screeners containing ticker | `store.screener_membership` (hourly poll) | Snapshot blended with positive-return-ratio proxy |
| Recommendation overlap | Strong buy + buy fraction from analyst data | `store.analyst.recommendation_period` | Snapshot blended with correlation magnitude proxy |
| Dividend analytics trend | Dividend payout direction | `store.fundamentals.dividend_yield` presence | `dividend_analytics()` history comparison |

**Note on sector relative strength:** Live mode uses the daily sector return from `finance::sector()` as the benchmark; historical mode uses a 20-day sector ETF (XLK/XLF/…/XLU) chart lookback. Both are meaningful but differ in horizon. The GICS mapping lives in `src/features/sector_map.rs`.

#### Block 8: Text Embedding (64 features)

The 64-dim Block-8 vector is produced by the **MarketReader deep head**
(Section 7.10), the on-device encoder (or the full teachers run in-process on the reference target).

**Distillation teacher (training-time only).** The deep head is distilled against
`microsoft/harrier-oss-v1-0.6b` — a qwen3-based embedder, **1024-dim**, 32K-token
context, last-token pooling, served locally over an OpenAI-compatible
`/v1/embeddings` endpoint. Each text snippet (EDGAR primary-document text ∪ FNSPID
news) in the trailing 90-day window is embedded, pooled with
`weight = exp(-Δdays / 30)`, and L2-normalized. Each input is capped to ~80K chars
(under harrier's 32K-token context) and the embedder character-budgets each
request (`src/historical/embedder.rs`) so large filings never OOM the CUDA decode
— the cause of the intermittent HTTP 500s in early collection runs. harrier-oss is **not** Matryoshka,
so the 1024-dim pooled vector is reduced by a learned **PCA(64)** basis (NIPALS,
fit once over the corpus and persisted; `src/historical/text_pca.rs`) and
re-normalized — these PCA(64) vectors are the deep-head regression targets. This
keeps Block 8 at its 64-dim budget, so the per-asset vector stays **184 effective
dims**. (Until the student is trained, the collector uses these teacher PCA(64)
vectors directly — see the §7.10 implementation-status note.)

**Total per-asset features: 10 + 42 + 12 + 30 + 8 + 10 + 8 = 120 continuous + 64 text = 184 effective dims.**

---

### 6.3 Market-Wide Feature Assembly (96 dims)

#### Volatility & Sentiment (12 features)

Sub-block layout — absolute dims 184–195:

| Dim (market-relative) | Feature | Live source | Historical source | Dims |
|---|---|---|---|---|
| 0 | VIX level (÷ 100, clamped 0–2) | `Ticker::new("^VIX").quote()` | `Ticker::new("^VIX").chart()` | 1 |
| 1–5 | VIX regime one-hot (extreme-fear/fear/neutral/greed/extreme-greed) | Thresholds: ≥35 / ≥28 / ≥20 / ≥14 / <14 | Same thresholds applied to VIX chart close | 5 |
| 6 | VIX level (÷ 20, clamped 0–5) | Same VIX quote | Same VIX chart | 1 |
| 7 | VIX 1d log-return | `(vix_now / vix_prev).ln()` | Day-over-day log-return from VIX chart | 1 |
| 8 | VIX term structure proxy | Stub (0.0) — no live VIX3M feed | Stub (0.0) | 1 |
| 9 | Fear & Greed index (÷ 100) | `finance::fear_and_greed()` | `finance::fear_and_greed()` historical | 1 |
| 10 | DXY proxy return | UUP return from `world_indices` poller | UUP log-return from chart | 1 |
| 11 | Gold–oil spread proxy | `(gold_ret − oil_ret) × 0.5` clamped ±2 | Same formula from GLD/USO chart | 1 |

> **Note:** Both live and historical modes now use VIX-based dims [0–8]. Fear & Greed is at dim 9 as a macro-sentiment proxy in both modes. The VIX term structure (dim 8) remains a stub — no live or historical VIX3M data source is available via finance-query.

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

#### Sector & Industry Performance (16 features)

Sector order: Tech(0), Financials(1), ConsumerDisc(2), CommServices(3), Healthcare(4), Industrials(5), ConsumerStaples(6), Energy(7), Materials(8), RealEstate(9), Utilities(10).

| Feature | Dims | Live source | Historical source |
|---|---|---|---|
| Sector 1d returns (11 sectors) | 11 | `finance::sector()` daily returns | Sector ETF chart data (XLK, XLF, XLY, XLC, XLV, XLI, XLP, XLE, XLB, XLRE, XLU) |
| Sector dispersion | 1 | Std dev of 11 sector returns | Same, computed from ETF returns |
| Best sector ID | 1 | Index of top-performing sector | Same |
| Worst sector ID | 1 | Index of worst-performing sector | Same |
| Sector rotation velocity | 1 | 20d change in sector ranking correlation | Same, using ETF return windows |
| Breadth (% above SMA200) | 1 | `finance::market_summary()` | `finance::market_summary()` historical |

#### World Indices & Cross-Asset (20 features)

| Feature | Live source | Historical source | Dims |
|---|---|---|---|
| S&P 500 return (1d) | `finance::indices(Americas)` — `^GSPC` | `Ticker::new("^GSPC").chart()` | 1 |
| NASDAQ return (1d) | `finance::indices(Americas)` — `^IXIC` | `Ticker::new("^IXIC").chart()` | 1 |
| Russell 2000 return (1d) | `finance::indices(Americas)` — `^RUT` | `Ticker::new("^RUT").chart()` | 1 |
| Dow Jones return (1d) | `finance::indices(Americas)` — `^DJI` | `Ticker::new("^DJI").chart()` | 1 |
| FTSE 100 return (1d) | `finance::indices(Europe)` — `^FTSE` | `Ticker::new("^FTSE").chart()` | 1 |
| DAX return (1d) | `finance::indices(Europe)` — `^GDAXI` | `Ticker::new("^GDAXI").chart()` | 1 |
| Nikkei 225 return (1d) | `finance::indices(AsiaPacific)` — `^N225` | `Ticker::new("^N225").chart()` | 1 |
| Hang Seng return (1d) | `finance::indices(AsiaPacific)` — `^HSI` | `Ticker::new("^HSI").chart()` | 1 |
| Shanghai Composite return | `finance::indices(AsiaPacific)` — `000001.SS` | `Ticker::new("000001.SS").chart()` | 1 |
| DXY proxy (UUP return) | From `world_indices` poller | `Ticker::new("UUP").chart()` | 1 |
| EUR/USD return | `world_indices` poller | `Ticker::new("EURUSD=X").chart()` | 1 |
| USD/JPY return | `world_indices` poller | `Ticker::new("JPY=X").chart()` | 1 |
| Gold return | `world_indices` poller (GLD proxy) | `Ticker::new("GLD").chart()` | 1 |
| Oil return | `world_indices` poller (USO proxy) | `Ticker::new("USO").chart()` | 1 |
| BTC price (normalized) + 1d return | `finance::crypto_market()` | `Ticker::new("BTC-USD").chart()` | 2 |
| ETH price (normalized) + 1d return | `finance::crypto_market()` | `Ticker::new("ETH-USD").chart()` | 2 |
| Crypto total market cap (log-normalized) | `finance::crypto_market()` | `finance::crypto_market()` historical | 1 |
| BTC dominance | BTC mkt cap / total | Same | 1 |

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
| Market hours state | `finance::hours(Region::US)` → pre/open/post/closed | 1 |
| Minutes since market event | Minutes since last open/close transition | 1 |

#### RSS & News Aggregate Signals (12 features)

| Feature | Computation | Dims |
|---|---|---|
| Aggregate news sentiment (1h) | Mean FinBERT sentiment across all articles in last hour | 1 |
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

### 6.3.1 Train/Inference Alignment Notes

The following documents any residual divergences between live-mode (`src/features/`) and historical-mode (`src/historical/batch.rs`) feature computation. All listed items are intentional approximations, not bugs.

| Block | Dim(s) | Live mode | Historical mode | Impact |
|---|---|---|---|---|
| Vol/Sentiment | 8 (term structure) | 0.0 stub | 0.0 stub | None — both modes stub equally |
| Vol/Sentiment | 9 (F&G) | `finance::fear_and_greed()` real-time | Historical daily F&G values | Minimal — same series, different latency |
| Cross-Asset | 113 (sector RS) | 1d return vs sector daily (live `finance::sector()`) | 20d return vs sector ETF chart (e.g., XLK) | Horizon mismatch (1d live vs 20d historical) — training sees stronger relative-return signals |
| Cross-Asset | 116 (trending) | Real-time `trending_symbols` set updated hourly | Historical trending computed from volume anomaly | Source mismatch; both are binary signals |
| Cross-Asset | 117 (screener) | Hourly screener scan, normalized count | Historical screener scan from that day | Both use `Screener::all()` iteraton; latency differs |
| Breadth/Flow | 82 (market hours) | UTC-approximate, EST/EDT not adjusted | UTC-approximate, same code path | Both slightly inaccurate during DST transitions |

**All 280 feature dimensions are populated in both live and historical modes.** No block is zero-filled by design.

---

### 6.4 Event Stream Architecture

Unchanged from v3.0. 18 event categories, typed event tokens with learned embeddings, time-decayed injection into the backbone. See Appendix D for full catalog.

---

### 6.5 Text Corpus Pipeline

The collector polls 31 RSS feeds. Each in-window headline (∪ EDGAR primary-document text in the historical path) feeds the MarketReader encoder (Section 7.10): the deep head decay-pools them into the 64-dim Block 8 vector and the headline head produces the sentiment slots. MarketReader's deep/sentiment heads are distilled from the harrier-oss→PCA(64) and FinBERT teachers respectively.

---

### 6.6 Portfolio State Vector (2·n_assets + 1 dims)

The portfolio vector is sized to the **maximum** universe (`n_assets` slots), with
inactive slots zero-filled — so its width is `2·n_assets + 1` (default
2·100 + 1 = **201**). Padded slots carry zero weight and zero P&L, so they
contribute nothing to portfolio return, turnover, or P&L.

| Feature | Dims |
|---|---|
| Current weight per asset (zero for inactive slots) | n_assets (100) |
| Unrealized P&L per asset (%) (zero for inactive slots) | n_assets (100) |
| Total portfolio value (log-scaled) | 1 |

> The width is derived from one place (`MeridianSpec.n_assets`, see Appendix C):
> `portfolio_state_dim = 2·n_assets + 1`. Growing the cap is a retrain (it is a
> trained `portfolio_proj` input dimension) and requires widening the Rust
> portfolio-vector assembly to match.

---

### 6.7 Financial Calendar Embedding (20 dims)

Unchanged. Encodes minute of session, day of week, month, market-open status, proximity to FOMC, options expiration, etc.

---

## 7. Model Components

### 7.1 Input Projection Layer with Variable Selection

Unchanged from v3.0:

```
Per-Asset (184 dims → D):
  Step 1: Variable Selection Network
    gates = softmax(Linear(regime_emb, 184)) ∈ ℝ^184
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
```

**Parameters:** ~245K

### 7.2 Regime Detector

Unchanged from v3.0. Input: 96 market-wide + 3 portfolio summary + 64 regime state = 163 dims.

```
Linear(163 → 256) → SiLU → Dropout(0.1)
Linear(256 → 128) → SiLU → Dropout(0.1)
├── Linear(128 → 64)  → regime_embedding
├── Linear(128 → 4)   → softmax → regime_posteriors
└── regime_change_velocity = ||regime_t − regime_{t-1}||_2
```

**Parameters:** ~55K

---

### 7.3 Per-Asset Temporal Backbone (Hybrid RA-GDN + SWA Stack)

**v4.0 change: Blocks 3 and 7 are now Hybrid blocks — an RA-GDN layer followed by a Sparse Sliding Window Attention (SWA) layer.** All other blocks remain pure RA-GDN. This is the Samba-Finance architecture (inspired by Samba, NeurIPS 2024).

**Motivation:** With 1,695 training rows per asset, the recurrent state cannot reliably learn to recall exact events from even 20 steps ago because there are too few sequences that share similar event-followed-by-consequence patterns. SWA provides **exact short-range recall** from the last W=64 steps without requiring learned recurrent compression. This directly addresses the mechanism behind the Sharpe 0.57 limitation: the model was missing recent events that the recurrent state had begun to compress away.

#### 7.3.1 RA-GDN Block (Blocks 1, 2, 4, 5, 6, 8)

Unchanged from v3.0. For each asset independently:

```
Input: x_t ∈ ℝ^D, state S_t ∈ ℝ^(d_head × d_v)

Regime modulation:
  α_t = sigmoid(W_α × regime_emb + b_α)   ∈ (0, 1)   — forgetting rate
  β_t = sigmoid(W_β × regime_emb + b_β)   ∈ (0, 1)   — update strength

DeltaNet update rule:
  k_t = W_k × x_t                          ∈ ℝ^d_head
  v_t = W_v × x_t                          ∈ ℝ^d_v
  q_t = W_q × x_t                          ∈ ℝ^d_head
  
  prediction error: δ_t = v_t − k_t^T × S_{t-1}
  S_t = (1 − α_t) × S_{t-1} + β_t × k_t ⊗ δ_t
  
  y_t = q_t^T × S_t
  output = W_o × y_t + x_t                 (residual connection)
```

#### 7.3.2 Sparse Sliding Window Attention (SWA) Layer (Blocks 3, 7)

The SWA layer operates **only on the last W=64 timesteps** of the current sequence window, giving the model precise short-range memory.

```
SWA parameters:
  W_q_swa, W_k_swa, W_v_swa: Linear(D → D)
  W_o_swa: Linear(D → D)
  n_heads_swa = 4, d_head_swa = 192
  window_size W = 64

Forward pass (training, sequence mode):
  For timestep t with context window [t-W, t]:
    Q = W_q_swa × X[t-W:t]   ∈ ℝ^(W × D)
    K = W_k_swa × X[t-W:t]   ∈ ℝ^(W × D)
    V = W_v_swa × X[t-W:t]   ∈ ℝ^(W × D)
    
    attn = softmax(QK^T / √d_head_swa) × V
    output = W_o_swa × attn + X[t]   (residual at current step only)

Forward pass (inference, recurrent mode):
  Maintain rolling KV cache of last W=64 steps per asset
  Cache size: 64 × D × 2 (keys + values) × FP16
           = 64 × 768 × 2 × 2 bytes = 196 KB per asset
           × 100 assets = 19.2 MB total KV cache
  
  At inference step t:
    Append (k_t, v_t) to cache, evict oldest
    Compute attention over all W cached pairs
    Latency: O(W × D) = O(64 × 768) per asset, per SWA layer
```

**Critical design decision — why W=64:**
- At daily resolution: 64 days ≈ 3 months of trading history. Captures the most recent earnings cycle, any recent analyst changes, and recent insider activity precisely.
- At inference: O(W) not O(sequence_length). Fixed cost regardless of how long the system has been running.
- At training: the RA-GDN handles >64 step dependencies. SWA handles ≤64. The combination covers all timescales without quadratic cost.

**Quantization:** SWA attention weights are FP16 (not ternary) because attention logits require precision. Keys, values, and queries use INT8. This adds ~2.4 MB to the model size.

**Hybrid block structure (complete backbone):**

```
Block 1:  RA-GDN
Block 2:  RA-GDN → FMoE
Block 3:  RA-GDN → SWA(W=64)       [Hybrid — precise short-term recall]
Block 4:  RA-GDN
Block 5:  RA-GDN → FMoE
Block 6:  RA-GDN
Block 7:  RA-GDN → SWA(W=64)       [Hybrid — pre-fusion recall]
Block 8:  RA-GDN → FMoE
```

SWA at block 3: provides precise short-term context before the first temporal cascade injection point. SWA at block 7: provides precise recall immediately before cross-asset fusion, so the fusion module sees both long-range compressed state and exact short-range context.

**Additional parameters from SWA layers:**
Two SWA layers × (3 × D × D projection + D × D output) ≈ 2 × (4 × 768²) ≈ 4.7M parameters. These are FP16 / INT8.

**Additional KV cache at inference:** 100 assets × 2 SWA layers × 64 × 768 × 2 (FP16) = 37.7 MB.

---

### 7.4 Regime-Routed Sparse Mixture of Experts

Unchanged from v3.0. 3 FMoE blocks (after RA-GDN in blocks 2, 5, 8), 4 experts each, 2 active per step via regime posterior routing.

**Parameters:** ~42.5M (21.3M active per forward pass).

---

### 7.5 Temporal Cascade

Unchanged from v3.0. Three GDN stacks (hourly/daily/weekly resolution), bidirectional surprise-gated injection.

**Parameters:** ~4.2M.

---

### 7.6 Cross-Asset Fusion Module

Unchanged from v3.0. 110+ token multi-head attention over all assets + market + portfolio + episodic memory retrievals. Sector bias prior matrix. 2 layers.

**v4.0 note:** Because SWA at block 7 feeds precise short-range state into this module, cross-asset fusion now operates on representations that include exact recent event context, not just recurrently compressed approximations. This is expected to improve the cross-asset correlation objective.

**Dynamic-universe masking.** Fusion accepts an `active_mask` `(n_assets,)`. Padded
asset slots are excluded as attention *keys* (their columns in the additive bias
are set to −∞), so no live token ever attends to a padded slot; the always-present
market/portfolio/episodic tokens keep every query row finite (no softmax NaN). The
sector-bias prior is sector-indexed (11×11), already universe-size-agnostic.
Verified: perturbing padded-slot contents leaves live outputs bit-identical
(`python/tests/test_dynamic_universe.py`).

**Parameters:** ~11.8M.

---

### 7.7 Financial Episodic Memory

Unchanged from v3.0. FAISS k-NN buffer of surprising market states, populated during training on both real and FDA-generated sequences.

**Quality-first (revised):** on the reference target, retrieval uses **exact k-NN (`IndexFlatL2`/`IndexFlatIP`)** rather than an approximate index — at this buffer size and per-minute cadence, exact search is sub-millisecond and removes ANN recall error entirely. The buffer cap (formerly 50,000, sized to the Pi memory budget) can be raised substantially; with 128 GB available the binding limit is coverage usefulness, not memory.

**v4.0 note:** FDA-generated crisis sequences increase the representation of rare regimes in the episodic memory buffer. Without FDA, crisis-regime episodes are underrepresented (real data: ~2008, ~2020 = ~2 crisis years in 10yr history). With FDA: 30× multiplier with regime conditioning generates proportionally more crisis sequences, giving the buffer better coverage of tail scenarios.

**Parameters:** ~200K projection weights + FAISS index.

---

### 7.8 Portfolio Output Heads

Unchanged from v3.0. Two heads (return-maximizing Meridian-R, Sharpe-maximizing Meridian-S), shared backbone.

**v4.0 note:** Portfolio-Theoretic Augmentation (Section 11.4) is applied during Phase 2 head training. The heads themselves are architecturally identical; PTA operates at the data level.

**Dynamic-universe masking.** Both heads accept an `active_mask`: inactive slots
get logit −∞ before the (temperature-scaled) softmax, so they receive exactly zero
weight and the allocation renormalizes over the live universe (Σ wᵢ = 1 over active
slots). With `mask=None` the heads behave exactly as v3.0. During Phase 2, each
training step draws a random live subset (`dynamic_universe_*` in
`PortfolioTrainConfig`) so the heads learn to allocate over arbitrary user-chosen
universes; eval/backtest always run the full universe for honest metrics.

**Parameters:** ~394K (197K per active head).

---

### 7.9 Auxiliary Prediction Head

Unchanged from v3.0. Predicts next-step return and volatility for computing surprise signal.

**Parameters:** ~100K.

---

### 7.10 MarketReader Text Encoder (distilled, optional)

The distilled on-device text encoder is the custom **3.5M-parameter MarketReader** (BPE-4096
tokenizer + 4 GDN blocks, D=256), runnable in-process with no external servers.

On the reference target (§4, 128 GB unified memory) the distilled student is **not used (quality-first)**: the
full teacher models (harrier-oss embedder + FinBERT sentiment) are run in-process at
inference for maximum-fidelity text features. The distilled student is retained only for resource-constrained
hosts where running the teachers is not feasible — it was originally a hard requirement under the
8 GB Pi 5 budget, which no longer binds.

```
Shared backbone:
  BPE Tokenizer: 4,096 vocab (financial-domain pre-tokenization)
  Token Embedding: 4,096 × 256   Position Embedding: 512 × 256
  4 GDN Blocks: D=256, n_heads=4, d_head=32, d_v=64

Tier-specific output heads:
  Headline:  attention_pool → Linear(256 → 4)   → 4-dim sentiment
  Summary:   attention_pool → Linear(256 → 32)  → 32-dim topic+sentiment
  Deep:      attention_pool → Linear(256 → 64)  → 64-dim Block-8 embedding

Total parameters: ~3.5M (INT8 quantized via Phase-4 QAT: ~3.5 MB)
Scaffold: python/market_data/marketreader.py ; tokenizer: python/train/train_tokenizer.py
```

**How it is trained — distillation from two teachers (training-time only).** The
student is distilled offline (GPU box), never re-deriving the teachers at
inference:

- **Deep head (Block 8, 64-dim)** is distilled against the
  `microsoft/harrier-oss-v1-0.6b` teacher (qwen3-based, 596M, 1024-dim,
  non-Matryoshka): per-symbol in-window headlines (∪ EDGAR primary-doc text) are
  embedded, decay-pooled with `weight = exp(-Δdays/30)`, L2-normalized, then
  reduced by a learned **PCA(64)** basis (NIPALS; `src/historical/text_pca.rs`).
  Those PCA(64) vectors are the deep-head regression targets (cosine + MSE).
- **Headline head (sentiment)** is distilled against **FinBERT**
  (`ProsusAI/finbert`, signed `P(pos)−P(neg)`; `src/sentiment/mod.rs`).

Once trained, historical Block-8/sentiment features are **regenerated with the
student** so training features equal what the Pi produces (train/inference
parity). The teachers (harrier-oss via llama-swap; FinBERT via the
`services/finbert` sidecar) are used only as label generators and are never
deployed. If text is unavailable at inference, Block 8 falls back to honest
zeros — never a fabricated or stale signal (live twins:
`src/poller/text_embed.rs`, `src/poller/news_sentiment.rs`).

> **Implementation status (2026-06):** the MarketReader scaffold, tokenizer
> trainer, both teachers, and the **distillation trainer**
> (`python/train/train_marketreader.py` — `dump-corpus` / `build-targets` /
> `distill` / `smoke`, wired as `make marketreader`) now exist; the distill loop
> is smoke-tested end-to-end. Remaining: run `build-targets`/`distill` on the real
> corpus (needs the eGPU + teachers up), **feature re-anchoring** to the student,
> and the **Rust on-device forward pass**. Until those land, the collector uses
> the teachers directly (harrier-oss→PCA(64) + FinBERT). The measured scaffold is
> ~2.0M params (the original "~3.5M" was an overestimate — even more comfortably
> compact). Distillation quality of the student vs a 596M teacher is unproven —
> the teacher path stays available for A/B until the student passes a held-out
> deep-head cosine threshold.

---

## 8. Financial Diffusion Augmentation (FDA)

This is the central new system in v4.0. The FDA is a standalone model — trained separately, run offline, and never deployed to the inference host. Its sole purpose is to produce additional training sequences for the main Meridian model.

### 8.1 Motivation & Scope

The core training problem: 1,695 daily rows per asset × 100 assets = 169,500 asset-day observations. For the backbone (pretrained with shared weights across assets), the effective training set is 169,500 rows (about a 560:1 parameter-to-observation ratio at 95M parameters). For the portfolio head (trained on 100-asset joint sequences), the effective training set is 1,695 sequences (about a 56,000:1 parameter-to-sequence ratio). This is why the model reaches training saturation before it has seen enough diversity to generalize.

FDA addresses this by generating synthetic sequences that:
1. Are statistically indistinguishable from real finance-query data on 9 measurable stylized facts
2. Are generated in the finance-query feature space (184-dim per-asset + 96-dim market-wide), ensuring no train/inference distribution mismatch
3. Are conditioned on regime labels, enabling controlled generation of underrepresented regimes (crisis, high-volatility)
4. Preserve cross-asset correlation structure, which is critical for portfolio head training

**FDA is used only for training data generation. At inference, the host runs only the main Meridian model.**

### 8.2 FDA Architecture

The FDA is a **score-based diffusion model** (DDPM with linear noise schedule) operating on financial time series windows. It is inspired by FinDiff (2023) and TSDiff (2023), adapted for the Meridian feature space.

```
FDA Model Architecture:

Input space:
  X ∈ ℝ^(T × N × F)  where:
    T = 64 (window length in trading days)
    N = 100 (assets)
    F = 184 (per-asset features) + 96 (market-wide, broadcast)
  
  In practice: work in the return/indicator space (log-returns + 
  normalized indicators), not raw prices, to improve stationarity.

Diffusion process:
  Forward: q(x_τ | x_0) = N(√ᾱ_τ × x_0, (1 − ᾱ_τ) × I)
    τ = noise level from 1 to T_diff = 1000
    ᾱ_τ = cumulative noise schedule (cosine schedule)
  
  Reverse (learned): p_θ(x_{τ-1} | x_τ, c)
    c = conditioning vector (regime label + market context)

Score network architecture:
  Backbone: 4-layer U-Net operating on (T, N, F) tensor
    Each level: (Conv1D along T) + (Linear across N) + RMSNorm
    Skip connections at each resolution
    
  Conditioning:
    Regime one-hot (4) → Linear(4 → 64) → γ, β for AdaRMSNorm
    Market-wide context (96) → Linear(96 → 64) → AdaRMSNorm
    Noise level τ → sinusoidal embedding(64) → AdaRMSNorm
  
  Total FDA parameters: ~8M (FP32 during training, not deployed)
  FDA training: ~40–60 H100-hours
  FDA inference: generate 50,000 windows in ~2 hours on H100
```

**Training the FDA:**

```
Training data: 100 assets × 1,695 days → windowed into
  100 × (1695 - 64) = 100 × 1631 = 163,100 windows of length T=64
  
  Each window: X_t ∈ ℝ^(64 × 280)  (184 per-asset + 96 market-wide)
  
Regime labels: assigned by pre-fit 4-state HMM on SPY returns
  Low-vol trending: ~55% of windows
  High-vol trending: ~20% of windows
  Mean-reverting: ~15% of windows
  Crisis: ~10% of windows

Loss: standard DDPM loss on ε-prediction
  L = ||ε − ε_θ(x_τ, τ, c)||²

Training: 200 epochs, AdamW lr=3e-4, batch=64 windows
```

### 8.3 Stylized Fact Validation Gate

Every batch of synthetic sequences passes a 9-point gate before being admitted to the training pool. Sequences that fail any check are discarded. This ensures the model never trains on unrealistic synthetic data.

| Check | Test | Threshold |
|---|---|---|
| **Fat tails** | Excess kurtosis of return distribution | > 1.0 (normal = 0) |
| **Volatility clustering** | Autocorrelation of squared returns at lag 1 | > 0.10 |
| **Negative skewness** | Return distribution skewness | < -0.05 |
| **No autocorrelation in returns** | Ljung-Box test on raw returns | p > 0.05 |
| **Leverage effect** | Correlation between past returns and future volatility | < -0.05 |
| **Volume-volatility correlation** | Correlation of log-volume and absolute return | > 0.15 |
| **Cross-asset correlations** | Mean pairwise correlation vs real data mean | Within ±0.10 |
| **Tail co-movement** | Fraction of joint extreme events (>2σ same direction) | Within ±0.05 of real |
| **Indicator consistency** | RSI computed from synthetic OHLCV matches synthetic RSI feature | Error < 0.01 |

The last check — indicator consistency — ensures the FDA generates internally consistent feature vectors (RSI, MACD, etc. match what would be computed from the generated OHLCV values). This prevents the main model from training on contradictory signals.

**Expected pass rate:** Based on FinDiff literature benchmarks, ~70–85% of generated sequences pass all 9 checks. The FDA is run until 50,000 valid windows are collected.

### 8.4 Regime-Conditioned Generation

A critical advantage of FDA over simple bootstrap resampling: **controlled generation of underrepresented regimes.**

In 1,695 real days of SPY data, crisis regimes (defined as VIX > 30 sustained for 2+ weeks) occupy approximately 10–12% of trading days. In a walk-forward backtest, the OOS period may contain a higher or lower proportion of crisis days. The model trained only on real data has seen very few examples of crisis sequences for rare assets.

FDA regime conditioning allows explicit oversampling:

```
Generation budget: 50,000 valid windows
  Low-vol trending:    30% (15,000 windows)   ← underweight vs real 55%
  High-vol trending:   25% (12,500 windows)
  Mean-reverting:      25% (12,500 windows)
  Crisis:              20% (10,000 windows)   ← overweight vs real 10%
  
Rationale: 
  - Crisis regime is most important for drawdown control
  - Main model has seen ~170 crisis windows in real data
  - FDA adds ~10,000 crisis windows (59× more)
  - Expected improvement: better calibrated crisis behavior
```

### 8.5 Cross-Asset Correlation Preservation

Portfolio head training requires realistic joint distributions across 100 assets. Simple per-asset augmentation destroys cross-asset structure.

The FDA generates joint windows (all 100 assets simultaneously) via the 2D score network operating on the (T × N) tensor. Cross-asset attention is implicit in the U-Net's spatial (N-axis) linear layers.

**Validation:** Before the FDA training data is used for portfolio head training, an additional cross-asset check is performed:

```python
def validate_cross_asset_structure(synthetic_batch, real_batch):
    """Checks that synthetic data preserves portfolio-relevant statistics."""
    
    # 1. Sector correlation blocks
    for sector in GICS_SECTORS:
        real_intra = mean_corr(real_batch, sector_assets[sector])
        synth_intra = mean_corr(synthetic_batch, sector_assets[sector])
        assert abs(real_intra - synth_intra) < 0.15
    
    # 2. Market beta distribution
    real_betas = compute_betas(real_batch, market_index=SPY_idx)
    synth_betas = compute_betas(synthetic_batch, market_index=SPY_idx)
    assert ks_statistic(real_betas, synth_betas) < 0.20
    
    # 3. Minimum variance portfolio weights
    real_mvp = min_variance_weights(real_batch)
    synth_mvp = min_variance_weights(synthetic_batch)
    assert mean_abs_difference(real_mvp, synth_mvp) < 0.05
```

If a batch of 1,000 synthetic windows fails this check, it is discarded and regenerated.

### 8.6 Limitations & Honest Caveats

The FDA is a powerful tool with real limitations that must be acknowledged:

**Limitation 1: Cannot generate truly novel market regimes.** The FDA is trained on 10 years of daily data (~2014–2024). It can generate variations of what it saw — 2008-style crises are absent from the training window. If the next decade features a new type of shock (e.g., digital asset contagion at scale, geopolitical financial warfare), the FDA-generated crisis sequences may be poor approximations.

**Limitation 2: Stylized fact preservation ≠ predictive signal preservation.** Generating sequences that match statistical properties does not guarantee the sequences contain the same predictive relationships between features and future returns. The FDA learns the marginal distribution well; it may learn the conditional distribution (given feature X, what is next-period return Y?) less well.

**Limitation 3: 30× multiplier is not 30× real data.** Effective sample size increase is sub-linear. FDA-generated sequences are not independent from the real data they were generated from — they share statistical properties by design. The actual effective sample size increase is likely 5–10× for purposes of parameter estimation, not 30×.

**Limitation 4: Indicator consistency check may miss edge cases.** The 9-point gate validates RSI consistency but does not validate every pairwise indicator relationship. A synthetic sequence where MACD and RSI are internally inconsistent might pass the gate.

These limitations are why the v4.0 expected Sharpe range is 0.82–1.24 (projected), not 2.0+. The FDA helps; it does not solve the data problem.

---

## 9. Continuous Learning System

### 9.1–9.3: Tiers 1–3

Identical to v3.0 Sections 8.1–8.3. Surprise-gated delta dynamics, normalization adaptation, EWC + ternary LoRA consolidation.

### 9.4 Adaptation Feedback Loop

Identical to v3.0 Section 8.4, with one addition:

**v4.0 addition — FDA quality monitoring during paper trading:**

As the paper trading system accumulates real inference-time data, FDA quality can be retrospectively evaluated:

```
Monthly FDA quality check:
  1. Collect 20 trading days of real finance-query feature vectors
  2. Run stylized fact checks on real data vs. FDA-generated data
  3. Compute distributional distance (MMD) between real and synthetic
  
  If MMD > 0.3:
    - Flag: FDA may have drifted from current market conditions
    - Retrain FDA on updated real data (including last 20 days)
    - Regenerate synthetic training pool
    - Trigger Phase 1 and Phase 2 retraining
  
  This ensures the FDA stays calibrated to current market structure,
  not just the 2014-2024 training window.
```

**Adaptation health dashboard (same as v3.0, plus):**

| Metric | What It Measures | Healthy Range |
|---|---|---|
| FDA-real MMD (monthly) | Distributional distance of synthetic vs real data | < 0.3 |
| Stylized fact pass rate (current) | Fraction of current synthetic windows passing gate | > 0.70 |
| SWA cache hit rate | Fraction of attention queries hitting informative keys | > 0.40 |

---

## 10. Quantization Strategy

> **Status: not used (quality-first).** The reference deployment (§4) runs **FP32** and is never quantized — quantization trades predictive quality for size/power this hardware does not need. This section is retained only as a documented fallback for a hypothetical resource-constrained deployment (e.g. an 8 GB ARM SBC) or NPU (INT8) execution. None of Phase 4 QAT (§11.6), ternary packing, or the precision-highway weight formats are part of the primary path.

When quantization *is* applied (constrained fallback only), the v3.0 scheme holds, with one addition:

**SWA layers:** SWA projection weights (W_q, W_k, W_v, W_o) are INT8 quantized. Attention logits and softmax are computed in FP16. KV cache stored in FP16. Consistent with the broader precision highway strategy.

The FDA model is never deployed to the inference host and is trained and run in FP32 on the H100 without quantization constraints.

---

## 11. Training Pipeline

### 11.1 Data Sources & Honest Row Counts

**v4.0 removes all paid external data providers.** Training data is sourced exclusively from finance-query to maintain train/inference distribution alignment.

| Data Type | Source | Granularity | Rows Per Asset | Notes |
|---|---|---|---|---|
| OHLCV (equities, ETFs) | `ticker.chart(1d, 10y)` via finance-query | Daily | ~1,695 | Hard constraint — Yahoo Finance free tier |
| OHLCV (crypto) | `crypto::coin()` historical | Daily | ~3,285 | Crypto trades 7 days/week |
| Technical indicators (42) | Computed from above OHLCV | Daily | ~1,695 | Same source, same z-score baselines |
| Options chains | `ticker.options(date)` historical snapshots | Daily | ~252 per year | Limited history availability |
| Fundamentals (XBRL) | `edgar::company_facts()` + finance-query | Quarterly | ~40 quarters | 10-year history |
| Earnings actuals vs estimates | `earnings_history()` | Quarterly | ~40 quarters | |
| Earnings estimate revisions | `earnings_trend()` | Monthly | ~36 months | ~3-year history |
| Analyst ratings history | `recommendation_trend()` | Monthly | ~36 months | |
| Analyst upgrades/downgrades | `grading_history()` | Event-driven | Variable | |
| Insider transactions | `insider_transactions()` | Event-driven | Variable | |
| Institutional ownership | `institution_ownership()` | Quarterly | ~20 quarters | |
| News headlines | finance-query RSS historical | Event-driven | ~500K total headlines | Across all assets over 3 years |
| Earnings transcripts | `earnings_transcript()` | Quarterly | ~40 per asset | |
| Treasury yield curve | `fred::treasury_yields()` | Daily | ~1,695 rows | |
| FRED macro (12 series) | `fred::series()` | Various | ~120 monthly observations | |
| Fear & Greed | Historical | Daily | ~1,695 | |
| VIX | `Ticker::new("^VIX").chart()` | Daily | ~1,695 | |
| World indices | `finance::indices()` historical | Daily | ~1,695 | |
| Sector performance | `finance::sector()` historical | Daily | ~1,695 | |

**FDA synthetic data (generated from the above):**

| Data Type | Method | Generated Rows | Valid After Gate |
|---|---|---|---|
| Synthetic OHLCV + feature sequences | FDA diffusion model | ~70,000 windows × 64 days | ~50,000 windows |
| Crisis-regime oversampled sequences | FDA regime conditioning | ~14,000 windows × 64 days | ~10,000 windows |
| **Total augmented effective rows** | **Real + synthetic** | **~51,695 per asset (est.)** | **Param ratio at 95M target: ~1,838:1 per-asset view (~18.4:1 pooled across 100 assets)** |

**Honest assessment of the augmented dataset:**
The 51,695 figure assumes the FDA generates fully independent samples, which overstates the effective sample count. A conservative estimate accounting for synthetic data dependency gives an effective independent sample count of ~8,000–15,000 rows per asset — a 5–9× improvement over the raw 1,695. This is meaningful but not the 290× gap needed to fully match the original spec's 490K assumption. The Sharpe > 1.0 target remains a stretch goal, not a guarantee.

**Validation crisis-coverage limitation (regime imbalance).**
The chronological split (train ≤ 2022, validation = 2023, test ≥ 2024) is strictly no-look-ahead, which means the **2023 validation slice contains zero crisis-regime days** (2023 never sustained VIX > 30; train-derived crisis labels: bull ~71% / hi-vol ~3% / mean-rev ~21% / crisis ~5%, val crisis = 0). We deliberately **do not** fabricate crisis rows in validation, nor carve a crisis eval window out of 2020/2022 — those crises sit inside the training window, and removing them would deplete the already-scarce ~170 real crisis windows. Instead: (1) **training** crisis coverage is supplied by FDA crisis oversampling (~10k synthetic crisis windows) plus regime-balanced sampling; (2) **crisis-regime evaluation** is performed on the test region (2024+, which can contain crisis days) and on held-out synthetic crisis windows; (3) the val-crisis gap is **reported explicitly** rather than masked. Consequence: 2023 validation metrics characterize normal/elevated-vol behavior only and must not be read as evidence of crisis robustness.

---

### 11.2 Phase 0 — FDA Pre-Training (Diffusion Model)

**Duration:** ~40–60 H100-hours  
**Output:** Trained FDA model capable of generating validated synthetic sequences

```
Step 1: Data preparation
  - Download all 100 assets' daily data via finance-query
  - Compute all 42 indicators per asset
  - Assemble 184-dim per-asset feature vectors for all 1,695 days
  - Compute 96-dim market-wide feature vectors for all 1,695 days
  - Normalize all features with rolling 252-day z-scores
  - Run HMM on SPY returns to assign regime labels to all days
  - Create (T=64, N=100, F=280) windows: 163,100 total
  
Step 2: FDA training
  - Train score network for 200 epochs (see Section 8.2)
  - Validation: held-out 20% of windows
  - Monitor: FID-style score on stylized facts, not raw loss
  
Step 3: Synthetic data generation
  - Generate 70,000 candidate windows
  - Apply 9-point stylized fact validation gate
  - Apply cross-asset correlation validation
  - Collect 50,000 valid windows → write to training shards
  
Step 4: FDA quality verification
  - Compute MMD between real and synthetic on held-out real windows
  - Verify regime distribution matches target (30/25/25/20%)
  - Verify no individual-asset autocorrelation leakage
  - Sign off before proceeding to Phase 1
```

---

### 11.3 Phase 1 — Contrastive Self-Supervised Pre-training

**Duration:** ~60–80 H100-hours  
**Input:** Real data (1,695 rows × 100 assets) + FDA synthetic (50,000 windows × 100 assets)  
**v4.0 change:** Replaces next-step prediction with Temporal Contrastive Pretraining (TCP)

**Why TCP instead of next-step prediction:**

Next-step return prediction generates one gradient signal per timestep. With 1,695 real rows and 100 assets, that's 169,500 gradient signals. TCP generates N² pairwise signals from the same data — temporally adjacent windows are positive pairs, distant or different-regime windows are negative pairs. Over the 100-asset universe, this yields approximately 100× more gradient signal from the same data, giving the backbone dramatically richer representation learning on limited data.

```
Contrastive objective:

For a batch of B sequence windows {W_1, ..., W_B}:
  
  1. Augmentation:
     For each window W_i, create two augmented views:
       W_i^a: original window [t-64, t]
       W_i^b: temporally adjacent window [t-60, t+4] (±4 day offset)
     These are "positive pairs" — same market event, slightly shifted.
     
     Negative pairs: any window W_j^a or W_j^b where j ≠ i
     Hard negatives: windows from same asset but different regime (HMM label differs)
  
  2. Encoder:
     z_i^a = MLP(pool(backbone(W_i^a)))   ∈ ℝ^128
     z_i^b = MLP(pool(backbone(W_i^b)))   ∈ ℝ^128
     Normalize: z = z / ||z||
  
  3. NT-Xent loss (InfoNCE):
     L_contrastive = -log( exp(z_i^a · z_i^b / τ) / Σ_j exp(z_i^a · z_j^b / τ) )
     τ = 0.1 (temperature)
  
  4. Regime supervised-contrastive term (v4.0, SupCon — Khosla et al. 2020):
     Within each batch, ALL windows sharing an HMM regime label are positives.
     L_supcon = mean over anchors i of
        -1/|P(i)| Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )
     P(i) = {windows with the same regime label as i}, taken over both views.

     Rationale (audit, 2026-06-22): the instance-contrastive objective (step 3)
     is solved trivially by instance discrimination — adjacent-window positives
     beat in-batch negatives at ~100% accuracy without learning any regime
     geometry. The multi-asset positive construction below was meant to supply
     regime structure but is empirically inert: at B=128 only ~0.009 cross-asset
     positives per anchor survive the corr>0.6 gate (zero in half of all
     batches), so cluster silhouette stays ~0. SupCon supplies a dense
     (~30–60 positives/anchor) regime-grouping gradient. It groups by *regime*,
     not *asset*, so it does not conflict with the asset-adversarial head.

  5. Auxiliary prediction head (kept from v3.0):
     L_pred = MSE(predict_return(backbone(W_i^a)), actual_return_{t+1})
  
  Total Phase 1 loss:
     L = λ_c × L_contrastive + λ_r × L_supcon + λ_p × L_pred
     λ_c = 0.6, λ_r = 0.2, λ_p = 0.2  (instance-contrastive still leads)

Multi-asset positive pairs (secondary, v4.0 relaxed):
  Additional positive pairs: (window of asset i at time t, window of 
  asset j at time t) when both assets are in the same sector AND same
  regime AND their price correlation over the window is > 0.3.
  
  This teaches the backbone that AAPL and MSFT in a bull tech regime
  should have similar representations — improving cross-asset fusion.
  v4.0 note: the correlation threshold is lowered 0.6 → 0.3 because 64-day
  daily-return correlation rarely clears 0.6 even within a sector, which left
  this pool empty. Regime structure now comes primarily from SupCon (step 4);
  these pairs are a secondary cross-asset signal.
```

**Training setup:**

```
Data mixture:
  Real windows:      163,100 (100 assets × 1,631 windows)
  FDA synthetic:     50,000 validated windows
  Total:             213,100 windows
  
  FDA synthetic weight in loss: 0.5 (half weight of real data)
  Rationale: real data is more reliable; synthetic supplements it

Optimizer: AdamW, lr=3e-4, cosine decay to 1e-5
Batch size: 128 windows
Training steps: ~20,000 (early stop on held-out contrastive accuracy)
Backbone: 8 hybrid blocks (RA-GDN + SWA) with shared weights
Gradient clip: 1.0
```

**Phase 1 validation criteria:**

```
Minimum to proceed to Phase 2:
  - Contrastive accuracy (positive pair > all negatives) > 0.60
    NOTE: necessary but NOT sufficient — this saturates at ~1.0 via trivial
    instance discrimination and says nothing about representation quality.
    Cluster silhouette (below) is the binding regime-separation gate.
  - Auxiliary return prediction R² > 0.01 (non-trivial but low is expected)
  - Regime separation: cluster silhouette > 0.10 on regime-stratified
    embeddings (quantitative proxy for "t-SNE shows 4 visible clusters")
  - SWA attention entropy < 0.8 (attending to specific positions, not uniform)
```

---

### 11.4 Phase 2 — Portfolio Optimization with Augmentation

**Duration:** ~50–70 H100-hours  
**v4.0 changes:** (a) Portfolio-Theoretic Augmentation applied at batch level; (b) FDA sequences used for head training with regime-balanced sampling; (c) Cross-asset pretraining for backbone

#### 11.4a Backbone Cross-Asset Pretraining (Phase 2 warm-up)

Before portfolio head training, the backbone is fine-tuned on the cross-asset correlation prediction task across all 100 assets simultaneously. This is the step that was described as "learned nothing" in v3.0 (N=1 pretraining).

```
Duration: ~10 H100-hours
Task: Given 100-asset feature window W_t ∈ ℝ^(T × 100 × 184),
      predict the 100×100 realized correlation matrix Σ̂_{t, t+20}
      (over the next 20 days)
Loss: Frobenius norm of (Σ̂ − Σ_realized)
Data: Real 100-asset joint windows + FDA joint windows
      100-asset joint windows: ℝ^(64 × 100 × 184)
      These are large — ~4.7M floats per window. Batch size = 8.
```

#### 11.4b Portfolio-Theoretic Augmentation (PTA)

Applied at the batch level during Phase 2 head training. For each real or synthetic training batch, PTA generates additional augmented batches using symmetry-preserving transformations.

```
PTA transformations (applied with probability p_aug = 0.5 per batch):

Transformation 1: Asset permutation preserving sector structure
  For each sector s, randomly permute asset indices within sector s.
  Portfolio weights are permuted accordingly.
  Covariance structure is approximately preserved within sectors.
  
  Example: AAPL ↔ MSFT in Tech sector, with weights swapped.
  Returns: w_augmented = permute(w_real), r_augmented = permute(r_real)
  Valid because: within-sector assets share similar factor loadings.
  Expected multiplier: ~12× (12 permutations per batch)

Transformation 2: Return sign + weight sign coupling
  negate: x_augmented = −x_real, w_augmented = −w_real
  P&L is unchanged: P&L = w^T × r = (−w)^T × (−r)
  Valid only when allowing short positions.
  Expected multiplier: 2× (original + negated)

Transformation 3: Volatility scaling
  Scale all returns and vol-dependent features by factor c ∈ [0.8, 1.2]:
  r_augmented = c × r_real
  Vol features (ATR, Bollinger %B) scaled accordingly
  Allocation weights unchanged (relative positions preserved)
  Expected multiplier: 4× (with c = 0.8, 0.9, 1.1, 1.2)

Combined PTA multiplier: ~12 × 2 × 4 = ~96× on average
Conservative effective sample size multiplier: ~8–12× (accounting for
correlation between augmented samples)

Implementation:
  def pta_augment(batch_features, batch_weights, batch_returns):
      augmented = [original]
      if rand() < 0.5:
          augmented += sector_permutations(batch, n=4)
      if rand() < 0.5:
          augmented += [sign_flip(batch)]
      if rand() < 0.7:
          augmented += vol_scales(batch, scales=[0.8, 0.9, 1.1, 1.2])
      return concat(augmented)
```

#### 11.4c Portfolio Head Training

```
Data:
  Real sequences:         1,695 days → 1,695 portfolio training steps
  FDA sequences:          50,000 windows (regime-balanced)
  PTA-augmented:          ~8–12× effective multiplier
  
  Total effective steps: ~1,695 × 8 (PTA) + 50,000 × 0.5 × 8 (FDA+PTA)
                       ≈ 13,560 + 200,000 ≈ 213,560 effective steps
  
Loss (unchanged from v3.0):
  L = −DSR + λ_r × L_return + λ_v × L_vol + λ_t × L_turnover
  
  DSR: Differentiable Sharpe Ratio
  L_return: next-period portfolio return
  L_vol: auxiliary volatility prediction
  L_turnover: L1 penalty on weight changes (prevents overtrading)
  
FDA data weighting:
  Real data: weight 1.0
  FDA data: weight 0.6
  
  Validation set: real data only (no synthetic in validation)
  
Regime-balanced sampling:
  Ensure each training batch contains at least one window from each
  of the 4 regimes. This prevents the head from optimizing only for
  the dominant (low-vol trending) regime.

Early stopping:
  Monitor: val_DSR on real-data-only validation split
  Patience: 200 steps
  Expected: convergence at 1,000–3,000 steps (vs. 2,000 in v3.0)
```

---

### 11.5 Phase 3 — Meta-Learning for Plasticity

Identical to v3.0. Regime transition episodes now use FDA-generated regime transition sequences (low-vol → crisis transitions are now plentiful due to crisis oversampling). This gives the meta-learning phase more realistic "sudden regime change" episodes to train adaptation on.

**Estimated duration:** ~40 H100-hours.

---

### 11.6 Phase 4 — Quantization-Aware Training (not run — quality-first)

> **Skipped (quality-first).** The deployment model is FP32 (§4, §10), so QAT is not run — it would only degrade quality. Retained as documentation for a constrained-deployment or NPU (INT8) fallback. The deployed weights are simply the Phase 2/3 FP32 checkpoint.

When run (fallback only): QAT for ternary backbone weights, INT8 activations, FP16 states. SWA projection weights are INT8-quantized in this phase.

**Estimated duration:** ~20 H100-hours.

**Total Phase 0–4: ~210–280 H100-hours with QAT, ~190–260 without** (within 500-hour budget, including FDA training).

---

## 12. Known Limitations & Expected Performance Bounds

This section is new in v4.0. It documents the honest constraints on Meridian's performance given the data environment, to prevent overclaiming and to set appropriate expectations before live deployment.

### 12.1 The Fundamental Data Constraint

**Problem:** The spec's original architecture (95M parameters) was designed for ~490,000 minute-bar training rows per asset. The finance-query-only constraint provides 1,695 daily rows — a 290× shortfall. No augmentation technique fully compensates for this gap.

**Current state (v3.0 baseline, validated on real OOS data):**

| Metric | Value |
|---|---|
| OOS Sharpe (2024+, daily) | 0.57 |
| OOS Total Return | +15.18% |
| OOS Max Drawdown | 13.05% |
| Win Rate | 67.25% |
| vs. Buy-and-Hold | +24.39 ppt |

**v4.0 expected improvement ranges:**

| Mechanism | Estimated Sharpe Contribution | Confidence |
|---|---|---|
| Contrastive pretraining (TCP) | +0.05 to +0.15 | Medium |
| SWA hybrid backbone | +0.05 to +0.12 | Medium |
| FDA synthetic augmentation | +0.10 to +0.25 | Low–Medium |
| Portfolio-theoretic augmentation | +0.05 to +0.15 | Medium |
| All combined (estimated) | +0.25 to +0.67 | Low |
| **v4.0 expected Sharpe range** | **0.82–1.24** | Low–Medium |

Confidence is "Low" for combined estimates due to interaction effects and the known limitations of synthetic data. The Sharpe > 1.0 target is achievable in the optimistic scenario but is **not guaranteed** from daily data alone.

### 12.2 The Stylized Fact Gap

FDA-generated sequences preserve 9 measurable stylized facts. However, there are known properties of financial time series that are difficult to capture:

- **Structural breaks:** Real markets have discrete regime changes driven by regulatory shifts, geopolitical events, and technology disruptions. FDA cannot generate sequences conditioned on events outside its training window.
- **Cross-sectional factor evolution:** Factor exposures (value, momentum, quality) drift over time in ways the FDA's 10-year training window does not fully capture.
- **Microstructure effects:** Daily bars smooth over intraday liquidity patterns that affect institutional execution. This is irrelevant at daily resolution but limits the model's future adaptability to higher frequencies.

### 12.3 The Single Data Source Constraint

Training exclusively on finance-query data ensures no distribution mismatch at inference. However, finance-query (Yahoo Finance backend) has known data quality issues:

- **Dividend adjustment timing:** Yahoo sometimes delays split/dividend adjustments, creating apparent price jumps that are data artifacts.
- **Volume data for ETFs:** Yahoo consolidated volume may differ from true primary exchange volume during pre/post-market periods.
- **Historical indicator data:** Rolling z-scores computed during training use the same normalization pipeline as inference, but the training distribution covers 2014–2024 while inference sees 2024 onward. Distributions may shift.

**Mitigation:** The z-score normalization uses rolling 252-day windows that are recomputed fresh at each inference step, anchoring to the current market environment rather than the historical training baseline.

### 12.4 The FDA Cannot Generate the Future

The FDA is trained on 2014–2024 data. It generates variations of what it saw. Performance claims based on backtests that include FDA-augmented training data are slightly optimistic because the FDA has indirectly seen the test period's market character (it's in the training data distribution, even if not the specific test sequences).

**Mitigation for backtesting:** When running the final backtest:
1. Split data: train/val on 2014–2021, OOS on 2022+
2. Train FDA only on 2014–2021 data
3. Evaluate Meridian on 2022+ data that neither the main model nor the FDA has seen
4. This is the cleanest possible test given the data constraints

---

## 13. Two-Head Strategy (Meridian-R / Meridian-S)

Unchanged from v3.0. Shared backbone, two output heads:
- **Meridian-R:** Return-maximizing head, DSR loss weighted toward return
- **Meridian-S:** Sharpe-maximizing head, DSR loss weighted toward risk-adjusted return

Both heads are trained with PTA augmentation in Phase 2. The backbone, pre-trained with TCP, is shared and frozen during the first 500 head training steps, then fine-tuned with a 10× lower learning rate.

Head blending at inference (regime-dependent):

```
active_weight_S = regime_posterior[crisis] + regime_posterior[high_vol]
active_weight_R = regime_posterior[low_vol] + regime_posterior[mean_rev]

final_weights = active_weight_S × weights_S + active_weight_R × weights_R
```

---

## 14. Historical Backtesting Framework

### 14.1 Meridian as a Backtesting Strategy

Unchanged from v3.0, with one critical addition — the **data contamination check**:

```rust
// IMPORTANT: Before running the backtest, verify that:
// 1. FDA was trained on data BEFORE the OOS split date
// 2. No synthetic data from after the split date is in training shards
// 3. The OOS period uses only real finance-query data for evaluation

struct BacktestDataAudit {
    train_end_date: NaiveDate,
    oos_start_date: NaiveDate,
    fda_train_end_date: NaiveDate,
}

impl BacktestDataAudit {
    fn verify(&self) -> Result<()> {
        assert!(self.fda_train_end_date <= self.train_end_date,
            "FDA must be trained on data no newer than main model training cutoff");
        assert!(self.oos_start_date > self.train_end_date,
            "OOS period must not overlap with training period");
        Ok(())
    }
}
```

The full `MeridianStrategy` implementation is unchanged from v3.0.

### 14.2–14.5: Benchmark Comparisons, Walk-Forward, Monte Carlo, Regime Analysis

Unchanged from v3.0. All benchmark comparisons, walk-forward configuration, MC analysis, and regime-segmented analysis procedures carry forward unchanged.

**v4.0 addition — ablation backtests:**

The following additional backtests are required to validate each v4.0 contribution:

| Ablation | What's Removed | Purpose |
|---|---|---|
| No FDA | Train without synthetic data | Quantify FDA contribution to Sharpe |
| No TCP | Use next-step prediction in Phase 1 | Quantify contrastive pretraining value |
| No SWA | Pure RA-GDN backbone | Quantify hybrid attention contribution |
| No PTA | No portfolio augmentation | Quantify PTA contribution |
| No FDA + no PTA | Equivalent to v3.0 | Verify v3.0 baseline reproducible |

Each ablation runs the full backtest pipeline and produces a Sharpe/Drawdown/Return table. Results guide future architectural decisions.

---

## 15. Paper Trading System

Unchanged from v3.0, with one addition to the graduation criteria:

### 15.1–15.3: Architecture, Performance Tracking, Adaptation Monitoring

Identical to v3.0 Sections 13.1–13.3.

### 15.4 Graduation Criteria (Paper → Live)

All v3.0 criteria are retained:

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
| Inference latency p99 | < 2,000 ms (≪ 60,000 ms per-minute budget; ~145 ms measured FP32 CPU) | Since inception |
| Data pipeline reliability | < 1% missed minutes | 6 months |

**v4.0 addition:**

| Criterion | Threshold | Measurement Period | Rationale |
|---|---|---|---|
| Real data Sharpe ≥ FDA-only backtest Sharpe × 0.70 | Within 30% of backtest | 3 months | Validates FDA didn't overfit to training distribution |
| FDA monthly MMD | < 0.3 vs current real data | Monthly check | Validates FDA stays calibrated |

**Revised Sharpe threshold rationale:** The > 1.0 Sharpe threshold in paper trading is retained. Given that v4.0 targets 0.82–1.24 from backtest, and paper trading adds realistic execution friction not fully captured in backtests, a 1.0 paper trading Sharpe is achievable in the optimistic scenario. If paper trading Sharpe settles around 0.8–0.9 after 6 months, the graduation criteria require reassessment before live deployment — not blind acceptance.

---

## 16. Inference Runtime

Per-minute inference loop on the reference x86-64 target (§4). The per-block millisecond figures below were the original Pi 5 / ternary-NEON projections and are kept for structural illustration only — on the reference Zen 5 target, FP32 eager PyTorch executes the full backbone in ~145 ms/step (measured), with the whole loop still using well under 1% of the per-minute budget. The data pipeline (200–500 ms, network-bound) dominates, exactly as before.

```
Time 0ms: New minute tick (PriceStream)
  │
  ├─ [Core 3, ASYNC] Data pipeline:
  │   ├── Fetch prices for 100 tickers (PriceStream — subscribed)
  │   ├── Compute 42 technical indicators per asset
  │   ├── Process any new RSS headlines (batch)
  │   ├── Check for new events (insider tx, filings, ratings)
  │   ├── Update rolling statistics (z-scores, EMAs, correlations)
  │   ├── Assemble 100 × 184 per-asset features
  │   ├── Assemble 96 market-wide features
  │   ├── Assemble 201 portfolio state features
  │   └── Duration: 200–500ms (indicator computation + normalization)
  │
  ├─ [CPU/iGPU] Model inference (per-block ms = legacy Pi/ternary projection; FP32 Zen 5 CPU measures ~145ms total):
  │   ├── Input projection + VSN gating: 0.5ms
  │   ├── Regime detection: 0.1ms
  │   ├── RA-GDN blocks 1-2 (100 assets): 1.0ms
  │   ├── RA-GDN block 3 + SWA(W=64) [NEW]: 1.5ms
  │   ├── RA-GDN blocks 4-6 (100 assets): 1.5ms
  │   ├── RA-GDN block 7 + SWA(W=64) [NEW]: 1.5ms
  │   ├── RA-GDN block 8 + FMoE: 1.0ms
  │   ├── Event token processing (~5 assets): 0.3ms
  │   ├── Temporal cascade (when active): 0.2ms
  │   ├── Cross-asset fusion (110 tokens, 2 layers): 0.5ms
  │   ├── FAISS episodic memory: 0.5ms
  │   ├── Output head: 0.05ms
  │   └── TOTAL: ~8.65ms  (vs. ~7ms in v3.0)
  │
  ├─ [Core 2, ASYNC] Post-inference:
  │   ├── Update SWA KV caches (100 assets × 2 layers): 0.2ms
  │   ├── Observe actual returns from t-1
  │   ├── Compute surprise signal
  │   ├── Tier 2 norm adaptation (every 60 min): 0.1ms
  │   ├── Log to SQLite: 0.5ms
  │   └── Paper broker simulation: 0.1ms
  │
  └─ Budget: 60,000ms available. Total used: ~350–650ms (pipeline-dominated; ~145ms model on FP32 CPU). CPU util: ~1%
```

SWA adds ~1.65 ms of inference latency (well within the 60,000 ms per-minute budget). The KV cache update is the most expensive part of SWA at inference — appending new (k, v) pairs and evicting the oldest — but this is a simple O(W) memory operation.

---

## 17. Memory Budget

> **Non-binding on the reference target.** With 128 GB unified memory (§4), the inference footprint is no longer a constraint. Measured FP32 process RSS is ~1.5 GB (dominated by the PyTorch runtime); FP16 weights are ~120 MB. The breakdown below reflects the original *quantized Pi 5* deployment and is kept for the constrained-deployment fallback. On the reference target, the "Ternary/INT8 weights" rows collapse into a single ~120 MB FP16 weight blob and the totals are immaterial.

### Static Allocation (legacy quantized Pi 5 fallback)

| Component | Size | Notes |
|---|---|---|
| Ternary weights (backbone RA-GDN + MoE + cascade) | 18.5 MB | 2-bit packed + FP16 scales |
| INT8 weights (SWA projections, 2 layers) | 3.0 MB | New in v4.0 |
| INT8 weights (regime detector + MarketReader) | 3.8 MB | |
| FP16 weights (SWA attention logits, output heads, projections, VSN) | 2.5 MB | |
| GDN hidden states (100 assets × 8 layers) | 75.5 MB | FP16 |
| SWA KV caches (100 assets × 2 layers × W=64 × D=768) | 37.7 MB | FP16 — new in v4.0 |
| Temporal cascade states | 16.4 MB | |
| Conv1d caches | 3.9 MB | |
| Episodic memory buffer (50K entries) | 13.0 MB | |
| FAISS index | 6.4 MB | |
| Rolling statistics + feature caches | 8.0 MB | |
| Event state buffer | 2.0 MB | |
| Text embedding cache | 1.0 MB | |
| HMM parameters | 0.1 MB | |
| **Total static** | **~192 MB** | |

### Dynamic Allocation

| Component | Size |
|---|---|
| Activation buffers | ~3.0 MB |
| SWA attention computation buffer | 0.8 MB |
| Cross-asset attention KV | 0.2 MB |
| Feature assembly buffer | 0.19 MB |
| RSS/news processing queue | 2.0 MB |
| **Total dynamic** | **~6.2 MB** |

### Periodic Allocation (Tier 3)

| Component | Size |
|---|---|
| Fisher diagonal | 60 MB |
| Parameter checkpoint | 24 MB |
| Replay buffer | 15 MB |
| LoRA adapters + gradients | 1.5 MB |
| **Total Tier 3** | **~100.5 MB** |

### SQLite Logging

| Component | Size per day | Notes |
|---|---|---|
| Per-minute log | ~20 MB/day | 390 rows |
| Daily summaries | ~10 KB/day | |
| Event log | ~1 MB/day | |
| NVMe budget | 256 GB | ~3.5 years of logs |

### Grand Total

| Scenario | Memory |
|---|---|
| Normal inference | ~198 MB |
| During Tier 3 consolidation | ~299 MB |
| OS + Rust data pipeline + Python bridge | ~500 MB |
| **Peak usage** | **~799 MB** |
| **Available (8GB Pi 5)** | **8,192 MB** |
| **Headroom** | **~7,393 MB (90%)** |

The SWA KV caches add ~37.7 MB to static allocation — a meaningful increase but still well within the Pi 5's 8GB. Headroom remains 90%.

---

## 18. Development Roadmap

### 18.0 Planned Quality Improvements

Consistent with the **quality over efficiency** priority (§2), the following improvements are
planned. They reuse the existing pretrained backbone — none require re-running Phase 1.

**Inference / runtime:**
- FP32 inference everywhere; no quantization.
- Run the **full harrier-oss + FinBERT text teachers in-process** for Block-8 features instead of the distilled student.
- **Exact k-NN** episodic memory (`IndexFlat`) with an enlarged buffer cap.
- Inference-time ensembling: **blended Meridian-R + Meridian-S heads**, test-time augmentation, and an always-on temporal cascade. (Blending requires both heads trained.)

**Phase 2 (portfolio) training:**
- **Joint FDA synthetic augmentation.** The portfolio loader (`SyntheticWindowPortfolioDataset`)
  requires joint `(n_assets, T, F)` windows; the per-asset FDA synth used in pretraining is the
  wrong shape and is silently skipped. Phase 2 reads joint synth (`data/mmaps_synth_joint`,
  produced by `make fda-joint`) via a dedicated `--synth-dir`, kept separate from `--data-dir`
  so the two synth formats never collide on the shared `FDA_SYNTH_` prefix.
- Loss/objective tuning (`losses.py`: DSR, turnover, volatility-aux), stronger PTA, and FDA
  mix-ratio / per-batch validation-gate tuning.

Model-capacity scaling (`d_model`/experts) is **not currently planned**: it would require a full
Phase 1 retrain and, under the data-scarcity regime (~1,695 bars/asset), is more likely to overfit
than to help. It is reconsidered only if a measured baseline shows the model underfitting.

### Phase 0: Data Infrastructure & FDA (Weeks 1–5)

- [ ] Build DataBus in Rust using finance-query as library dependency
- [ ] Implement all 42 indicator computations with rolling z-score normalization
- [ ] Implement RSS feed polling across all 31 sources with deduplication
- [ ] Implement event detection (diff-based) for insider tx, ratings, filings
- [ ] Build historical data download pipeline (finance-query only — no external providers)
- [ ] Implement SQLite logging schema
- [ ] Build feature assembly pipeline (184 per-asset + 96 market-wide)
- [ ] Verify all finance-query API integrations end-to-end
- [ ] Train BPE tokenizer on financial text corpus
- [ ] Pre-fit 4-state HMM on historical regime data
- [ ] **[NEW] Implement FDA score network (8M param U-Net)**
- [ ] **[NEW] Train FDA for 200 epochs on finance-query daily data**
- [ ] **[NEW] Implement 9-point stylized fact validation gate**
- [ ] **[NEW] Implement cross-asset correlation validation**
- [ ] **[NEW] Generate and validate 50,000 synthetic training windows**
- [ ] **[NEW] Run FDA quality verification — sign off before proceeding**

### Phase 1: Core Model (Weeks 6–12)

- [ ] Implement RA-GDN block in PyTorch (extending FLA Gated DeltaNet)
- [ ] Implement regime conditioning (α bias, β scaling)
- [ ] **[MODIFIED] Implement SWA layer with rolling KV cache (W=64)**
- [ ] **[MODIFIED] Assemble hybrid backbone (RA-GDN + SWA at blocks 3, 7)**
- [ ] Implement EventEncoder
- [ ] Implement FMoE with external regime routing
- [ ] Implement temporal cascade
- [x] Implement cross-asset attention with sector bias **+ dynamic-universe key-padding mask** (`python/core/cross_asset_fusion.py`)
- [ ] Implement VSN with regime-conditioned gating
- [x] **[NEW] Distillation trainer `python/train/train_marketreader.py` (dump-corpus / build-targets / distill / smoke) wired as `make marketreader`; distill loop smoke-tested**
- [ ] **[NEW] Run `make marketreader` on the real corpus (needs eGPU + teachers): harrier-oss→PCA(64) deep + FinBERT sentiment targets → distilled student checkpoint**
- [ ] **[NEW] Re-anchor historical Block-8 + sentiment features to the student**
- [ ] **[NEW] (optional) Port MarketReader forward to Rust (Candle/ONNX) for a server-free path — not required on the reference target, where the full teachers run in-process**
- [ ] **[MODIFIED] Implement TCP (temporal contrastive) loss**
- [ ] **[MODIFIED] Run Phase 1 TCP pretraining on real + FDA data (mixed 1.0/0.5 weight)**
- [ ] Validate: contrastive accuracy > 0.60, regime cluster separation visible

### Phase 2: Portfolio Training (Weeks 13–17)

- [x] Implement portfolio output heads **+ dynamic-universe active-mask renormalization** (`python/optimization/portfolio_heads.py`)
- [x] **[NEW] Dynamic-universe random-subset training (mask sampling in `portfolio_train.py`; `MeridianSpec.portfolio_state_dim` derived)**
- [ ] **[MODIFIED] Implement DSR loss + PTA augmentation pipeline**
- [ ] **[NEW] Implement sector-permutation, sign-flip, vol-scale transforms**
- [ ] Run Phase 2a cross-asset correlation pretraining (backbone fine-tune)
- [ ] Run Phase 2b return head training (Meridian-R) with PTA
- [ ] Run Phase 2c Sharpe head training (Meridian-S) with PTA
- [ ] Build MeridianStrategy wrapper for finance-query backtesting
- [ ] **[NEW] Run ablation backtests (no-FDA, no-TCP, no-SWA, no-PTA)**
- [ ] Run full backtests vs all benchmarks (Section 14.2)
- [ ] Run walk-forward validation (Section 14.3)
- [ ] Run Monte Carlo analysis (Section 14.4)
- [ ] Run regime-segmented analysis (Section 14.5)
- [ ] Iterate: if backtest fails criteria, diagnose and retrain

### Phase 3: Adaptation & QAT (Weeks 18–23)

- [ ] Implement 3-tier continuous learning system
- [ ] **[MODIFIED] Run Phase 3 meta-learning with FDA-generated regime transition episodes**
- [ ] Run Phase 4 QAT (including INT8 SWA projections)
- [ ] Validate quantized model performance within tolerance
- [ ] Ablation study: adaptation value vs frozen
- [ ] Ablation study: event stream value
- [ ] Ablation study: text value

### Phase 4: Deployment (Weeks 24–30)

- [ ] FP32 inference path in PyTorch on the reference x86-64 target (CPU; optional ROCm iGPU; FP16 optional). No hand-written ternary/NEON kernel required.
- [ ] **[NEW] Implement SWA inference with rolling KV cache (FP16, host)**
- [ ] FAISS episodic memory (faiss-cpu on x86-64)
- [ ] Port full inference pipeline to the Framework Desktop (validate first on Framework Laptop 13)
- [ ] Run 10M+ step state drift test
- [ ] **[NEW] Run SWA cache overflow test (ensure W=64 eviction works correctly)**
- [ ] ~~NEON ternary kernel / ARM port~~ (dropped — superseded by FP16 x86-64; keep only for a future constrained-deployment fallback)
- [x] **[NEW] Live universe control: hot-reload `portfolio_symbols.txt` (mtime watcher in `src/scheduler/watch.rs`) → warm up added symbols (price-history backfill in `src/scheduler/warmup.rs`; rolling-stat z-scores warm live), evict removed + free state, re-subscribe PriceStream on change (watch channel), emit only warmed symbols (`src/store/mod.rs`). Remaining: deeper rolling-stat backfill + sector/mktcap prefetch on add.**
- [ ] Implement paper trading engine
- [ ] Deploy to the Framework Desktop with monitoring dashboard
- [ ] 72-hour continuous operation soak test (functional/stability; Pi-era thermal/power stress no longer the binding concern on a desktop)
- [ ] Begin 6-month paper trading period

### Phase 5: Paper Trading & Iteration (Weeks 31–57)

- [ ] Monitor all graduation criteria (Section 15.4)
- [ ] **[NEW] Monthly FDA MMD check — retrain FDA if MMD > 0.3**
- [ ] Weekly review of adaptation health metrics
- [ ] Monthly comparison to benchmarks
- [ ] Iterate on model if needed (retrain specific phases)
- [ ] After 6 months: evaluate graduation criteria
- [ ] If passed: proceed to minimal live trading ($10K initial capital)

---

## 19. Risk Analysis & Mitigations

All risks from v3.0 are retained. The following risks are new or modified in v4.0:

### Risk #1–#10: Carried Forward from v3.0

State drift, catastrophic forgetting, regime miscalibration, overfitting, data pipeline latency, insufficient model capacity, feature bloat, RSS feed reliability, event misclassification, options data sparsity — all with their v3.0 mitigations.

### Risk #11: FDA Distribution Collapse (New — High Priority)

**Risk:** The FDA generates sequences that replicate training data statistics superficially but contain no genuine predictive signal. The main model trains on "realistic-looking noise" that degrades rather than improves generalization.

**Indicators:**
- Stylized fact pass rate is high (sequences look realistic) but validation Sharpe on real data is lower than the no-FDA ablation
- FDA-trained model's feature importance (VSN gates) differs dramatically from real-data-only model

**Mitigations:**
1. The 9-point validation gate is a necessary but not sufficient condition. Also validate: does adding FDA data improve or degrade real-data-only validation Sharpe?
2. Run the no-FDA ablation as a required step (Section 14.2). If FDA degrades performance, do not use it.
3. FDA data weight 0.5 (half real data weight) limits damage if FDA is low quality.
4. Hard stop: if no-FDA ablation shows Sharpe ≥ FDA Sharpe, drop FDA from the training pipeline entirely and revert to real-data-only training. A 0.57 Sharpe with clean data is more trustworthy than a 0.62 Sharpe with potentially contaminated data.

### Risk #12: SWA Cache Staleness (New — Medium Priority)

**Risk:** At inference, the SWA KV cache contains the last 64 trading days of features. If the system experiences downtime > 64 trading days (unlikely but possible), the cache is stale when resumed. The SWA layers will attend to an outdated context.

**Mitigation:**
1. On startup, detect if last log entry is > W=64 trading days ago
2. If stale: warm up the SWA cache by running the model in inference-only mode (no portfolio decisions) on the most recent 64 days of price data before resuming live operation
3. Implementation: `if days_since_last_inference > 64: warm_swa_cache(last_64_days_of_data)`

### Risk #13: PTA Asset Permutation Breaking Fundamental Features (New — Medium Priority)

**Risk:** PTA's asset permutation transformation swaps features between assets. This is valid for price/indicator features (which are already normalized cross-sectionally) but may produce inconsistent fundamental features — swapping AAPL's PE ratio into MSFT's slot while keeping MSFT's price momentum creates a chimera asset that no real security resembles.

**Mitigation:**
1. Permutation is applied only within sectors (assets share similar fundamental profiles)
2. Blocks 3–5 of per-asset features (options, fundamentals, ownership) are **not permuted** — only Blocks 1–2 (price/volume/indicators) are swapped
3. The asset identity embedding (sector + market cap) is updated to match the permuted asset
4. Validation: PTA-augmented batches should not produce VSN gate patterns that gate out all fundamental features. Monitor during training.

### Risk #14: Known Performance Ceiling with Daily Data (Acknowledged — Not Mitigated)

**Risk:** Even with FDA, TCP, SWA, and PTA, the model may not reach Sharpe > 1.0 from daily data alone. The graduation criteria require 6 months of paper trading before live deployment. If paper trading Sharpe is consistently 0.7–0.9, the graduation criteria (Sharpe > 1.0) will not be met.

**This is not a bug — it is an honest constraint.**

**Options if this occurs:**
1. Accept a lower graduation Sharpe threshold (e.g., 0.80) with tighter max drawdown constraints (< 15%)
2. Acquire daily data for a longer historical window (some free sources offer 20+ years of daily data vs. Yahoo's 10 years) — this is still finance-query compatible if finance-query can source it
3. Re-evaluate whether Sharpe > 1.0 is the right metric for a daily-bar system, or whether risk-adjusted alpha over passive is a more appropriate target

---

## Appendix A — Parameter Count Breakdown

| Component | Parameters | Active Per Step | Quantization | Size |
|---|---|---|---|---|
| Input projections + VSN (expanded) | 380K | 380K | Ternary | 95 KB |
| EventEncoder | 50K | 50K | FP16 | 100 KB |
| Regime detector (expanded) | 55K | 55K | INT8 | 55 KB |
| RA-GDN blocks (8) | 17.4M | 17.4M | Ternary | 4.3 MB |
| SWA layers (2 × 4 projections) | 4.7M | 4.7M | INT8 | 4.7 MB |
| FMoE blocks (3 × 4 experts) | 42.5M | 21.3M | Ternary | 10.6 MB |
| Temporal cascade (3 GDN + projections) | 4.2M | 4.2M | Ternary | 1.1 MB |
| Cross-asset attention (2 layers) | 11.8M | 11.8M | Ternary + FP16 | 3.0 MB |
| Episodic memory projections | 200K | 200K | FP16 | 400 KB |
| Portfolio heads (2) | 394K | 197K | FP16 | 788 KB |
| Auxiliary prediction head | 100K | 100K | FP16 | 200 KB |
| Sector bias matrix | 121 | 121 | FP32 | 484 B |
| MarketReader text encoder | 3.5M | 3.5M (async) | INT8 | 3.5 MB |
| RMSNorm parameters | ~25K | ~25K | FP32 | 100 KB |
| **Total (Meridian main model)** | **~85.3M** | **~36.5M** | **Mixed** | **~29.0 MB** |
| FDA model (offline only, never deployed) | ~8M | N/A | FP32 training | ~32 MB (H100 only) |

---

## Appendix B — Complete finance-query Integration Map

**Training data note:** All API calls in this map are used for both training data collection and live inference. This is the key v4.0 principle — no external training data sources.

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

### Historical Data Collection (Training Pipeline Only)

| API Call | Granularity | Lookback | Purpose |
|---|---|---|---|
| `ticker.chart(1d, 10y)` × 100 | Daily | 10 years | Primary OHLCV training data |
| `ticker.dividends(10y)` × 100 | Event | 10 years | Dividend-adjusted returns |
| `ticker.splits(10y)` × 100 | Event | 10 years | Split-adjusted prices |

### Backtesting Integration

| API Call | Purpose |
|---|---|
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
| Max universe size (`n_assets`) | 100 | Static trained asset-axis cap; live universe = any subset 1..n_assets |
| Portfolio state dim | 2·n_assets + 1 (=201) | Derived from `n_assets` (`MeridianSpec.portfolio_state_dim`) |
| Per-asset input dim | 184 | 120 continuous + 64 text |
| Market-wide input dim | 96 | |
| RA-GDN blocks | 8 | Core temporal layers |
| SWA layers | 2 | At blocks 3 and 7 |
| SWA window size | 64 | Trading days of exact recall |
| SWA heads / d_head | 4 / 192 | INT8 projections, FP16 logits |
| FMoE blocks | 3 | After blocks 2, 5, 8 |
| GDN heads / d_head / d_v | 4 / 64 / 192 | |
| MoE experts / active | 4 / 2 | Regime-routed |
| Regime embedding dim | 64 | |
| Regime classes | 4 | Low-vol, high-vol, mean-rev, crisis |
| Event types | 18 | |
| Event embedding dim | 64 | Per component |
| Temporal cascade dims | 512, 512, 384 | Hourly, daily, weekly |
| Cross-asset attention heads/layers | 8 / 2 | |
| Episodic memory slots | 50,000 | |
| MarketReader dim / layers | 256 / 4 | |
| MarketReader vocab | 4,096 | BPE |
| Text embedding output | 64 | Per-asset |
| RSS feed sources | 31 | |
| FRED macro series tracked | 12 | |
| Technical indicators used | 42 | |

### FDA Architecture

| Parameter | Value | Notes |
|---|---|---|
| Window size | 64 trading days | T dimension |
| Asset count | 100 | N dimension |
| Feature dims | 280 | 184 per-asset + 96 market-wide |
| Diffusion steps | 1000 | Linear noise schedule |
| Score network | 4-layer U-Net | 8M parameters |
| Regime conditioning | 4-class one-hot | AdaRMSNorm |
| Training epochs | 200 | |
| Validation gate points | 9 | See Appendix E |
| Target synthetic windows | 50,000 | After gate filtering |
| Crisis regime fraction | 20% | Oversampled vs ~10% real |

### Training

| Parameter | Value | Notes |
|---|---|---|
| Phase 0 (FDA) epochs | 200 | AdamW lr=3e-4 |
| Phase 1 (TCP) steps | ~20,000 | Early stop on contrastive acc |
| Phase 1 batch size | 128 windows | |
| TCP temperature τ | 0.1 | NT-Xent |
| TCP loss weights | λ_c=0.7, λ_p=0.3 | Contrastive + prediction |
| FDA data weight (Phase 1) | 0.5 | Half weight vs real |
| FDA data weight (Phase 2) | 0.6 | Slightly higher for portfolio |
| PTA permutations per batch | 4 | Within-sector permutations |
| PTA sign flip probability | 0.5 | |
| PTA vol scale values | 0.8, 0.9, 1.1, 1.2 | |
| Dynamic-universe subset prob | 0.5 | Fraction of Phase-2 steps drawing a random live subset |
| Dynamic-universe min live | 10 | Floor on live-asset count when subsetting |
| Phase 2 early stop patience | 200 steps | Real-data-only val DSR |

### Backtesting

| Parameter | Value | Notes |
|---|---|---|
| Initial capital | $1,000,000 | |
| Commission | 10 bps | |
| Slippage | 5 bps | |
| Min trade threshold | 0.5% | |
| Leverage limit | 2.0× | |
| Walk-forward in-sample | 1 year | Daily bars |
| Walk-forward OOS | 3 months | |
| Monte Carlo simulations | 10,000 | Block bootstrap |
| Consistency ratio target | > 0.6 | |
| Sharpe ratio target | > 1.0 | OOS, annualized |
| Max drawdown target | < 20% | 95% MC CI |
| OOS split date for clean backtest | 2021-12-31 | FDA trained on ≤ this date |

---

## Appendix D — Event Type Catalog

Unchanged from v3.0:

| Event Type | Source API | Detection Method | Magnitude Encoding | Typical Frequency |
|---|---|---|---|---|
| INSIDER_BUY | `insider_transactions()` | Diff consecutive polls | Dollar value (log) | ~2/week per asset |
| INSIDER_SELL | `insider_transactions()` | Diff | Dollar value (log) | ~5/week per asset |
| ANALYST_UPGRADE | `grading_history()` | Diff | Grade delta | ~1/month per asset |
| ANALYST_DOWNGRADE | `grading_history()` | Diff | Grade delta | ~1/month per asset |
| EARNINGS_BEAT | `earnings_history()` | Actual > estimate | Surprise % | Quarterly |
| EARNINGS_MISS | `earnings_history()` | Actual < estimate | Surprise % | Quarterly |
| SEC_FILING_10K | `sec_filings()` / `edgar::submissions()` | New filing detected | Binary | Annual |
| SEC_FILING_10Q | Same | New filing | Binary | Quarterly |
| SEC_FILING_8K | Same | New filing | Binary (high priority) | ~4/year per asset |
| SEC_FILING_OTHER | Same | New filing | Binary | Variable |
| FOMC_DECISION | `feeds::fetch(FederalReserve)` | RSS pattern match | Rate change bps | 8/year (global) |
| MACRO_RELEASE | FRED series update | New data point | Surprise vs consensus | Variable |
| NEWS_HIGH_IMPACT | MarketReader score | Magnitude > threshold | Sentiment score | ~1/day per asset |
| DIVIDEND_ANNOUNCED | `calendar_events()` | New dividend date | Yield change | Quarterly |
| SPLIT_ANNOUNCED | `splits()` | New split detected | Split ratio | Rare |
| OPTIONS_UNUSUAL | Options chain polling | Volume > 2σ AND vol/OI > threshold | Volume ratio | ~1/week per asset |
| TRENDING_ENTRY | `finance::trending()` | Ticker appears in list | Binary | Variable |
| RATING_CHANGE | `recommendation_trend()` | Distribution shift | Net consensus change | Monthly |

---

## Appendix E — FDA Stylized Fact Checklist

This checklist is implemented as an automated validation function run on every batch of synthetic sequences before admission to the training pool.

| # | Stylized Fact | Test Statistic | Pass Threshold | Implementation |
|---|---|---|---|---|
| 1 | Fat tails | Excess kurtosis of daily log-returns | > 1.0 | `scipy.stats.kurtosis(returns) - 3 > 1.0` |
| 2 | Volatility clustering | Autocorrelation of squared returns at lag 1 | > 0.10 | `np.corrcoef(r²[:-1], r²[1:])[0,1] > 0.10` |
| 3 | Negative skewness | Skewness of return distribution | < -0.05 | `scipy.stats.skew(returns) < -0.05` |
| 4 | No return autocorrelation | Ljung-Box test on raw returns, lag 10 | p-value > 0.05 | `statsmodels.stats.diagnostic.acorr_ljungbox(returns, lags=10).pvalue.min() > 0.05` |
| 5 | Leverage effect | Corr(r_t, σ_{t+1}) | < -0.05 | `np.corrcoef(returns[:-1], vol_estimate[1:])[0,1] < -0.05` |
| 6 | Volume-volatility correlation | Corr(log_volume, |return|) | > 0.15 | `np.corrcoef(np.log(volume), np.abs(returns))[0,1] > 0.15` |
| 7 | Cross-asset correlations | Mean pairwise corr vs real baseline | Within ±0.10 | `abs(synthetic_mean_corr - real_mean_corr) < 0.10` |
| 8 | Tail co-movement | Joint >2σ event fraction vs real | Within ±0.05 | `abs(synthetic_joint_tail_freq - real_joint_tail_freq) < 0.05` |
| 9 | Indicator consistency | RSI(computed from OHLCV) vs RSI(feature) | MAE < 0.01 | `mean_abs_error(rsi_from_ohlcv, rsi_feature) < 0.01` |

A synthetic batch (1,000 windows) must achieve >85% pass rate on each individual check and >90% overall pass rate to be admitted. Batches below these thresholds are discarded and the FDA is queried for replacement sequences.

---

*Meridian v4.0 — Harvey, April 2026*
*Built on Gated DeltaNet (ICLR 2025), BitNet b1.58, Samba hybrid architecture, FinDiff, and the full finance-query ecosystem.*

*"The market is not short of data — it is short of the right data, correctly sourced, honestly evaluated."*

*v4.0 key principle: train/inference distribution alignment is non-negotiable. All training data is sourced from finance-query. The FDA augments but does not replace real data. Performance claims are validated on real OOS data only. Known limitations are documented, not papered over.*
