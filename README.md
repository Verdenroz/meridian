# Meridian

Meridian is a regime-adaptive portfolio model designed for CPU-native inference on a Raspberry Pi 5 with H100-based training. It ingests **every available data stream** from finance-query — price, volume, 42 technical indicators, 31 RSS/news feeds, full options chains, EDGAR filings, XBRL fundamentals, earnings transcripts, analyst ratings, insider transactions, institutional ownership, macro series, yield curves, sector/industry data, Fear & Greed, world indices, trending tickers, and crypto market data — producing allocation weights every minute for up to 100 US-listed assets (equities, ETFs, crypto).

The backbone is a **Regime-Aware Gated DeltaNet (RA-GDN)** with regime-conditioned dynamics, surprise-gated online plasticity, and ternary-native quantization. The model makes **proactive decisions** by processing discrete events (insider buys, analyst upgrades, SEC filings, earnings surprises, FOMC announcements) as first-class inputs the moment they arrive.

For the full architectural specification, see [`meridian-spec.md`](meridian-spec.md).

## Key metrics

| Metric | Value |
|---|---|
| Total parameters | ~80.6M (~35M active per forward pass) |
| Per-asset features | 224 continuous + 64 text = 288 |
| Market-wide features | 96 |
| Portfolio state features | 201 |
| Event types tracked | 18 discrete categories |
| Text sources ingested | 31 RSS feeds + per-ticker news + transcripts |
| Quantized model size | ~24.2 MB |
| Normal inference RAM | ~156 MB |
| Inference latency (100 assets) | ~7 ms |
| Data pipeline latency | ~200–500 ms (network-bound) |
| Adaptation tiers | 3 (sub-second / hourly / daily) |
| Training compute budget | ~250–400 H100-hours |

## Repository layout

```text
.
├── src/
│   └── main.rs                                    # Rust DataBus — schedules all data feeds,
│                                                  # detects events, assembles feature vectors
├── c/
│   ├── kernels/
│   │   └── ternary_neon.c                         # TODO: ARM NEON ternary inference kernel
│   └── runtime/
│       └── inference_runtime.c                    # TODO: Pi 5 inference runtime glue
├── python/
│   ├── cli.py                                     # `meridian` CLI entrypoint
│   ├── train/
│   │   ├── feature_schema.py                      # Feature dimension contracts (288/96)
│   │   ├── spec.py                                # Architecture hyperparameter contracts
│   │   ├── splits.py                              # Deterministic train/validation/test splits
│   │   └── qat.py                                 # TODO: quantization-aware training
│   ├── market_data/
│   │   ├── environment.py                         # H100 environment validation
│   │   ├── workspace.py                           # Data workspace management
│   │   ├── pipeline.py                            # Data pipeline orchestration
│   │   ├── feature_assembly.py                    # TODO: 288-dim per-asset + 96-dim market assembler
│   │   ├── dataset_ingestion.py                   # JSONL → memory-mapped shards
│   │   ├── finance_query_collector.py             # Historical data collector (Rust binary bridge)
│   │   ├── marketreader.py                        # MarketReader text encoder (3-tier: headline/summary/deep)
│   │   └── regime_hmm.py                         # 4-state HMM regime pre-fitting
│   ├── core/
│   │   ├── regime_aware_backbone.py               # RA-GDN blocks + regime-routed FMoE
│   │   ├── event_encoder.py                       # TODO: EventEncoder (18 event types → D-dim tokens)
│   │   ├── self_supervised_pretraining.py         # Return/volatility/correlation predictive objectives
│   │   ├── temporal_cascade.py                    # Hourly/daily/weekly GDN stack with surprise injection
│   │   ├── cross_asset_fusion.py                  # 2-layer MHA with sector bias prior (110+ tokens)
│   │   └── episodic_memory.py                     # FAISS-backed financial episodic memory
│   ├── optimization/
│   │   ├── portfolio_heads.py                     # TODO: Meridian-R + Meridian-S output heads
│   │   └── losses.py                              # TODO: DSR loss, turnover penalty, aux tasks
│   ├── adaptation/
│   │   └── adaptation.py                          # TODO: 3-tier continuous learning system
│   ├── backtesting/
│   │   └── backtesting.py                         # TODO: MeridianStrategy + walk-forward + Monte Carlo
│   ├── paper_trading/
│   │   └── paper_trading.py                       # TODO: PaperBroker + SQLite logging + graduation checks
│   ├── monitoring/
│   │   └── monitoring.py                          # TODO: adaptation health dashboard
│   ├── runtime/
│   │   ├── inference_pipeline.py                  # TODO: Pi 5 per-minute inference orchestration
│   │   └── memory_store.py                        # TODO: recurrent state + FAISS store management
│   ├── quantization/
│   │   └── ternary_kernels.py                     # TODO: C kernel Python bindings
├── scripts/
│   └── setup_h100_env.sh                          # H100 node bootstrap
├── pyproject.toml
└── meridian-spec.md                               # Architectural Specification v3.0
```

## Quick start (project setup before training)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
cp .env.example .env
meridian data init --root .
```

Run setup checks:

```bash
# Validation
meridian check spec
meridian check features --asset-class equity
meridian check split --timestamp 2024-08-15T13:30:00
meridian check env --allow-non-h100

# Data pipeline
meridian data plan --assets AAPL,SPY,BTC-USD --output reports/data_pipeline_manifest.json
meridian data ingest --input data/raw/AAPL_features.jsonl --asset AAPL --output-root .
meridian data collect --asset AAPL --start 2020-01-01 --end 2025-06-30

# Component training
meridian train plan --stage foundation
meridian train marketreader --plan --corpus data/raw/news_corpus.jsonl
meridian train marketreader --corpus data/raw/news_corpus.jsonl --epochs 1 --max-documents 5000
meridian train hmm --plan --features vix_level,adx14,corr_dispersion,drawdown
meridian train hmm --input data/processed/regime_features.jsonl --features vix_level,adx14,corr_dispersion,drawdown

# Model smoke tests
meridian smoke core --batch-size 2 --seq-len 16
meridian smoke backbone --batch-size 2 --seq-len 16
meridian smoke cascade --batch-size 2 --seq-len 16
meridian smoke fusion --n-assets 100 --n-ctx 10 --batch-size 2
meridian smoke train --steps 5 --batch-size 2 --seq-len 16
meridian smoke train --steps 20 --seq-len 16 --prediction-ramp-steps 10
meridian smoke train --steps 20 --seq-len 16 --input-shard data/mmaps/train/AAPL.f32bin --max-sequences 512
meridian smoke train --steps 20 --seq-len 16 --input-shards data/mmaps/train/AAPL.f32bin,data/mmaps/train/SPY.f32bin --max-sequences 256
meridian smoke train --steps 10 --resume-checkpoint checkpoints/backbone_pretrain.pt --eval-only
meridian smoke eval --shards-root data/mmaps --splits train,validation,test --resume-checkpoint checkpoints/backbone_pretrain.pt
meridian smoke eval --shards-root data/mmaps --splits train,validation --include-per-shard
meridian smoke eval --shards-root data/mmaps --splits train,validation --report-output reports/backbone_split_eval.json
```

For H100 nodes, use the bootstrap script:

```bash
bash scripts/setup_h100_env.sh
```

### Expected JSONL row schema for `data ingest`

```json
{
  "timestamp": "2024-06-30T10:00:00",
  "asset_class": "equity",
  "price_volume":        [0.0, "... 10 total ..."],
  "technical":           [0.0, "... 42 total ..."],
  "options":             [0.0, "... 12 total ..."],
  "fundamental":         [0.0, "... 30 total ..."],
  "risk_analytics":      [0.0, "... 8 total ..."],
  "event_state":         [0.0, "... 10 total ..."],
  "cross_asset_context": [0.0, "... 8 total ..."],
  "text":                [0.0, "... 64 total ..."]
}
```

The command writes split shards to `data/mmaps/{train|validation|test}/` and a manifest to `reports/ingestion/`.

`data collect` uses the Rust DataBus binary at `src/main.rs`
(`cargo run -- --symbol {symbol} --start {start} --end {end}`)
which ingests historical data via finance-query and emits schema-compatible JSONL rows.
Override with `--command-template` for a custom collector command.

---

## Development Roadmap

### Phase 0: Data Infrastructure (Weeks 1–4)

- [ ] Build DataBus in Rust using finance-query as library dependency
- [ ] Implement all 42 indicator computations with rolling z-score normalization
- [ ] Implement RSS feed polling across all 31 sources with deduplication
- [ ] Implement event detection (diff-based) for insider tx, ratings, filings, macro
- [ ] Build historical data download pipeline for training corpus
- [ ] Implement SQLite logging schema
- [ ] Build feature assembly pipeline (288 per-asset + 96 market-wide)
- [ ] Verify all finance-query API integrations end-to-end
- [ ] Train BPE tokenizer on financial text corpus
- [ ] Pre-fit 4-state HMM on historical regime data

### Phase 1: Core Model (Weeks 5–10)

- [ ] RA-GDN block in PyTorch with regime conditioning (α bias, β scaling)
- [ ] FMoE with external regime-posterior routing
- [ ] VSN input gating
- [ ] Self-supervised pre-training scaffold with return/volatility/correlation objectives
- [ ] Checkpoint save/resume with format version and compatibility checks
- [ ] Split-aware shard evaluation
- [ ] Temporal cascade: hourly/daily/weekly GDN stacks with surprise-gated context injection
- [ ] Cross-asset fusion: 2-layer MHA with sector bias prior (110+ tokens)
- [ ] EventEncoder: 18 event types → D-dim tokens with time decay
- [ ] MarketReader text encoder (3-tier: headline / summary / deep)
- [ ] Run Phase 1 self-supervised pre-training on H100 (~80 GPU-hours)
- [ ] Validate: prediction accuracy, recurrent state stability, regime detection quality

### Phase 2: Portfolio Training (Weeks 11–14)

- [ ] Portfolio output heads: Meridian-R (return) and Meridian-S (Sharpe) with leverage normalization
- [ ] DSR (Differential Sharpe Ratio) loss function, turnover penalty, volatility auxiliary task
- [ ] Build MeridianStrategy wrapper for finance-query backtesting engine
- [ ] Run Phase 2a (return head, ~50K steps) and Phase 2b (Sharpe head, ~30K steps)
- [ ] Run backtests vs all benchmarks (buy-and-hold SPY, equal-weight, risk parity)
- [ ] Run walk-forward validation, Monte Carlo analysis, regime-segmented analysis

### Phase 3: Adaptation & QAT (Weeks 15–20)

- [ ] Tier 1 — Surprise-gated plasticity (SGP): surprise signal modulates β update strength
- [ ] Tier 2 — Normalization adaptation: online running stats update
- [ ] Tier 3 — EWC + Ternary LoRA periodic consolidation
- [ ] Run Phase 3 meta-learning training (~40 GPU-hours)
- [ ] Run Phase 4 QAT training (~20 GPU-hours)
- [ ] Validate: quantized model performance within tolerance of full-precision
- [ ] Ablation studies: event stream value, text value, adaptation value

### Phase 4: Deployment (Weeks 21–26)

- [ ] NEON ternary inference kernel in C
- [ ] FAISS-backed episodic memory on ARM
- [ ] Full Pi 5 inference pipeline orchestration
- [ ] 10M+ step recurrent state drift test
- [ ] Paper trading engine with SQLite logging
- [ ] Deploy to Pi 5 with monitoring dashboard
- [ ] 72-hour continuous thermal and power stress test
- [ ] Begin 6-month paper trading period

### Phase 5: Paper Trading & Iteration (Weeks 27–52)

- [ ] Monitor all graduation criteria continuously (Section 13.4 of spec)
- [ ] Weekly review of adaptation health metrics (surprise signal, norm drift, consolidation loss)
- [ ] Monthly benchmark comparison (Meridian-R and Meridian-S vs SPY)
- [ ] Iterate on model if needed (retrain specific phases)
- [ ] After 6 months: evaluate all graduation criteria
- [ ] If passed: proceed to minimal live trading (~$10K initial capital)

---

## Contributing

Please read:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
