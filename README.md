# Meridian

Meridian is a regime-adaptive portfolio model targeting CPU-native inference (Raspberry Pi 5) with H100-based training.

The backbone is a **Regime-Aware Gated DeltaNet** (RA-GDN) — a novel adaptation of the Gated DeltaNet linear attention architecture with regime-conditioned dynamics, surprise-gated online plasticity, and ternary-native quantization. It manages a portfolio of up to 100 US-listed assets (equities, ETFs, crypto) 24/7, producing allocation weights every minute.

## Repository layout

```text
.
├── src/
│   └── main.rs                                    # Rust finance-query collector
├── c/
│   ├── kernels/
│   │   └── ternary_neon.c                         # TODO: Section 8.2/8.3 kernel
│   └── runtime/
│       └── inference_runtime.c                    # TODO: Section 11 runtime glue
├── python/
│   ├── cli.py                                     # `meridian` CLI entrypoint
│   ├── train/                                     # contracts + schedules
│   │   ├── feature_schema.py
│   │   ├── spec.py
│   │   ├── splits.py
│   │   └── qat.py                                 # TODO: Section 8 + 9.5
│   ├── market_data/                               # historical/live data foundation
│   │   ├── environment.py
│   │   ├── workspace.py
│   │   ├── pipeline.py
│   │   ├── feature_assembly.py
│   │   ├── dataset_ingestion.py
│   │   ├── finance_query_collector.py
│   │   ├── marketreader.py
│   │   └── regime_hmm.py
│   ├── core/                                      # RA-GDN + regime-routed backbone
│   │   ├── regime_aware_backbone.py
│   │   ├── self_supervised_pretraining.py
│   │   ├── temporal_cascade.py                    # TODO: Section 6.7
│   │   ├── cross_asset_fusion.py                  # TODO: Section 6.8
│   │   └── episodic_memory.py                     # TODO: Section 6.9
│   ├── optimization/
│   │   ├── portfolio_heads.py                     # TODO: Section 6.10/10
│   │   └── losses.py                              # TODO: Section 9.3 losses
│   ├── adaptation/
│   │   └── adaptation.py                          # TODO: Section 7
│   ├── runtime/
│   │   ├── inference_pipeline.py                  # TODO: Section 11
│   │   └── memory_store.py                        # TODO
│   ├── quantization/
│   │   └── ternary_kernels.py                     # TODO: C kernel bindings
│   └── tests/
│       └── test_*.py
├── scripts/
│   └── setup_h100_env.sh
├── meridian-spec.md
└── research.md
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

PYTHONPATH=python python -m unittest discover -s python/tests -q
```

Expected JSONL row schema for `data ingest`:

```json
{
  "timestamp": "2024-06-30T10:00:00",
  "asset_class": "equity",
  "price_volume": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "technical": [0.0, "... 20 total ..."],
  "options": [0.0, 0.0, 0.0, 0.0, 0.0],
  "fundamental": [0.0, "... 10 total ..."],
  "text": [0.0, "... 64 total ..."],
  "asset_calendar": [0.0, 0.0, 0.0]
}
```

The command writes split shards to `data/mmaps/{train|validation|test}/` and a manifest to `reports/ingestion/`.

`data collect` defaults to a **Rust collector binary** at `src/main.rs`
(`cargo run -- --symbol {symbol} --start {start} --end {end}`)
that uses the `finance-query` Rust library directly and emits schema-compatible JSONL rows.
Override with `--command-template` only if you want a custom collector command.

For H100 nodes, use the bootstrap script:

```bash
bash scripts/setup_h100_env.sh
```

---

## Development Roadmap

### Phase 0: Foundation

- [x] Set up training infrastructure (H100 env script, PyTorch environment)
- [x] Implement data pipeline: historical data ingestion, preprocessing, memory-mapped shard storage
- [x] Implement feature engineering: full 112 per-asset feature schema (`feature_schema.py`, `feature_assembly.py`)
- [x] Rust finance-query collector pulling OHLCV, indicators, options, fundamentals, news (`src/main.rs`)
- [x] MarketReader text encoder training pipeline (`marketreader.py`)
- [x] 4-state HMM on historical regime data (`regime_hmm.py`)
- [x] BPE tokenizer for MarketReader
- [x] CLI with all Phase 0 commands (`meridian check spec`, `data plan`, `train hmm`, etc.)
- [x] Deterministic train/validation/test split definitions (`splits.py`)
- [x] Environment validation contract (`environment.py`)

### Phase 1: Core Model

- [x] RA-GDN block in PyTorch with regime conditioning (α bias, β scaling) (`regime_aware_backbone.py`)
- [x] FMoE with external regime-posterior routing
- [x] VSN input gating
- [x] Self-supervised pre-training scaffold with return/volatility/correlation predictive objectives (`self_supervised_pretraining.py`)
- [x] Checkpoint save/resume with format version and compatibility checks
- [x] Split-aware shard evaluation (`smoke eval`)
- [x] Temporal cascade: hourly/daily/weekly GDN stacks with surprise-gated context injection (`temporal_cascade.py`)
- [x] Cross-asset fusion module: 2-layer multi-head attention with sector bias prior (`cross_asset_fusion.py`)
- [ ] Run Phase 1 training on H100 (~80 GPU-hours)
- [ ] Validate: prediction accuracy, recurrent state stability, regime detection quality

### Phase 2: Portfolio Training

- [ ] Portfolio output heads: Meridian-R (return) and Meridian-S (Sharpe) with leverage normalization (`portfolio_heads.py`)
- [ ] DSR (Differential Sharpe Ratio) loss function (`losses.py`)
- [ ] Turnover penalty and volatility auxiliary task
- [ ] Run Phase 2a (return head, ~50K steps) and Phase 2b (Sharpe head frozen backbone, ~30K steps)
- [ ] Backtest on validation set
- [ ] Compare against baselines: buy-and-hold SPY, equal-weight, risk parity

### Phase 3: Adaptation

- [ ] Tier 1 — Surprise-gated plasticity (SGP): surprise signal modulates β update strength (`adaptation.py`)
- [ ] Tier 2 — Normalization adaptation: online running stats update
- [ ] Tier 3 — EWC + Ternary LoRA periodic consolidation
- [ ] Run Phase 3 meta-learning training (~40 GPU-hours)
- [ ] Validate: adaptation speed on held-out regime transitions
- [ ] Stress test: market crash, flash crash, Fed surprise scenarios

### Phase 4: Quantization & Deployment

- [ ] BitNet b1.58 QAT: absmean ternary weights with straight-through estimators (`qat.py`)
- [ ] Run Phase 4 QAT training (~20 GPU-hours)
- [ ] Validate: quantized model performance within tolerance of full-precision
- [ ] NEON ternary inference kernel in C (`c/kernels/ternary_neon.c`)
- [ ] FAISS-backed episodic memory on ARM (`episodic_memory.py`)
- [ ] Full Pi 5 inference pipeline orchestration (`inference_pipeline.py`, `c/runtime/inference_runtime.c`)
- [ ] 10M+ step recurrent state drift test (Risk #1 validation)
- [ ] End-to-end integration test: live data → inference → allocation weights
- [ ] 72-hour continuous thermal and power testing on Pi 5

### Phase 5: Live Operation

- [ ] Paper trading with both Meridian-R and Meridian-S heads
- [ ] Monitor state drift, surprise distribution, regime detection accuracy
- [ ] Tune Tier 2/3 adaptation hyperparameters from live performance
- [ ] Build monitoring dashboard (state norms, surprise levels, allocation drift)
- [ ] Implement alerting for anomalous model behavior

---

## Contributing

Please read:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
