# Meridian

Meridian is a regime-adaptive portfolio model designed for CPU-native inference on a Raspberry Pi 5 with H100-based training. It ingests **every available data stream** from finance-query — price, volume, 42 technical indicators, RSS/news feeds, options chains, EDGAR/XBRL fundamentals, transcripts, analyst/insider/ownership data, macro series, yield curves, sector data, world indices, and crypto market context — producing allocation weights every minute for up to 100 US-listed assets (equities, ETFs, crypto).

The canonical architecture is now **v4.0** and is constrained by train/inference source alignment: training data is sourced from finance-query (daily history), not paid external minute-bar vendors. To address reduced data volume, v4 introduces FDA (diffusion augmentation), TCP (contrastive pretraining), PTA (portfolio-theoretic augmentation), and a hybrid RA-GDN + SWA backbone.

For the full canonical specification, see [`meridian-spec.md`](meridian-spec.md) (v4.0). `meridian-spec4.md` is retained as a reference draft artifact.

## Validation snapshot (v4)

| Area | Current status |
|---|---|
| Data-source fidelity (finance-query only) | Validated and aligned with current daily data paths |
| Arithmetic and assumptions in v4 spec | Validated with corrections applied in `meridian-spec.md` |
| Feature block coverage (all 280 dims) | All dims populated in both live and historical modes — no zero-filled blocks |
| Train/inference alignment | VIX-based Vol/Sentiment now consistent across both modes; sector ETFs and international index data fetched in historical mode; residual divergences documented in spec §6.3.1 |
| RA-GDN core | Implemented |
| SWA hybrid blocks | Implemented |
| FDA / TCP / PTA | Implemented in training pipelines |
| Performance range (Sharpe 0.82-1.24) | Projected, not guaranteed |

## Key metrics

| Metric | Value |
|---|---|
| Total parameters | ~95-96M target architecture; ~60.9M currently implemented |
| Per-asset features | 120 continuous + 64 text = 184 |
| Market-wide features | 96 |
| Portfolio state features | 201 |
| Event types tracked | 18 discrete categories |
| Training source / granularity | finance-query only / daily history |
| v3 validated OOS Sharpe baseline | 0.57 |
| v4 projected OOS Sharpe range | 0.82-1.24 (projected) |
| Quantized model size (target) | ~23 MB |
| Runtime memory footprint (target) | ~185 MB |
| Inference latency (100 assets, target) | ~6-11 ms |
| Data pipeline latency | ~200–500 ms (network-bound) |
| Adaptation tiers | 3 (sub-second / hourly / daily) |
| Training compute budget (v4 target) | ~180-280 H100-hours |

## Repository layout

```text
.
├── src/
│   └── main.rs                                    # Rust DataBus — schedules all data feeds,
│                                                  # detects events, assembles feature vectors
├── c/
│   ├── kernels/
│   │   └── ternary_neon.c                         # ARM NEON ternary inference kernel (stub)
│   └── runtime/
│       └── inference_runtime.c                    # Pi 5 inference runtime glue (stub)
├── python/
│   ├── cli.py                                     # `meridian` CLI entrypoint (stub)
│   ├── train/
│   │   ├── feature_schema.py                      # Feature dimension contracts (184/96)
│   │   ├── spec.py                                # Architecture hyperparameter contracts
│   │   ├── splits.py                              # Deterministic train/validation/test splits
│   │   ├── pretrain.py                            # Self-supervised pre-training entry point
│   │   ├── portfolio_train.py                     # Portfolio optimization training (Meridian-R/S)
│   │   └── qat.py                                 # Quantization-aware training (stub)
│   ├── market_data/
│   │   ├── environment.py                         # H100 environment validation
│   │   ├── workspace.py                           # Data workspace management
│   │   ├── pipeline.py                            # Data pipeline orchestration
│   │   ├── feature_assembly.py                    # 184-dim per-asset + 96-dim market assembler (stub)
│   │   ├── dataset_ingestion.py                   # JSONL → memory-mapped shards
│   │   ├── finance_query_collector.py             # Historical data collector (Rust binary bridge)
│   │   ├── marketreader.py                        # MarketReader text encoder (3-tier: headline/summary/deep)
│   │   └── regime_hmm.py                         # 4-state HMM regime pre-fitting (stub)
│   ├── core/
│   │   ├── regime_aware_backbone.py               # RA-GDN blocks + regime-routed FMoE
│   │   ├── event_encoder.py                       # EventEncoder: 18 event types → D-dim tokens with time decay
│   │   ├── self_supervised_pretraining.py         # TCP + auxiliary return objective (Phase 1)
│   │   ├── temporal_cascade.py                    # Hourly/daily/weekly GDN stack with surprise injection
│   │   ├── cross_asset_fusion.py                  # 2-layer MHA with sector bias prior (110+ tokens)
│   │   └── episodic_memory.py                     # FAISS-backed financial episodic memory
│   ├── optimization/
│   │   ├── portfolio_heads.py                     # Meridian-R + Meridian-S output heads with leverage normalization
│   │   └── losses.py                              # DSR loss, turnover penalty, volatility auxiliary task
│   ├── adaptation/
│   │   └── adaptation.py                          # 3-tier continuous learning system (stub)
│   ├── backtesting/
│   │   └── backtesting.py                         # MeridianStrategy + walk-forward + Monte Carlo
│   ├── paper_trading/
│   │   └── paper_trading.py                       # PaperBroker + SQLite logging + graduation checks (stub)
│   ├── monitoring/
│   │   └── monitoring.py                          # Adaptation health dashboard (stub)
│   ├── runtime/
│   │   ├── inference_pipeline.py                  # Pi 5 per-minute inference orchestration (stub)
│   │   └── memory_store.py                        # Recurrent state + FAISS store management (stub)
│   ├── quantization/
│   │   └── ternary_kernels.py                     # C kernel Python bindings (stub)
├── scripts/
│   └── setup_h100_env.sh                          # H100 node bootstrap
├── pyproject.toml
└── meridian-spec.md                               # Architectural Specification v4.0 (canonical)
```

## Quick start (project setup before training)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
cp .env.example .env
meridian data init --root .
```

### Makefile workflow (clean targets)

```bash
make stats                         # verify existing split shards
make data-ready FDA_DEVICE=cuda    # collect + FDA + augmentation + stats
make pretrain
make portfolio-return
make portfolio-sharpe
make backtest
```

Training targets default to `data/mmaps_aug`; override with `TRAIN_DATA=data/mmaps` when needed.
`make collect` performs a fresh rebuild of `data/mmaps/{train,validation,test}` shards before recollecting.
Portfolio head training uses the 100 real per-asset shards, enforces an exact 100-asset universe for train/validation, and skips synthetic non-asset shards (e.g., `FDA_SYNTH_*`) in that stage.
Both portfolio heads mix FDA synthetic window batches (`fda_loss_weight=0.6`, `fda_mix_prob=0.5` by default) with real joint-asset batches.
Checkpoint selection uses validation DSR for return-head training and validation rolling-Sharpe loss for sharpe-head training, with batch-level four-regime coverage enforced by default.
PTA uses batch-level view expansion with sector-preserving permutations derived from `portfolio_symbols.txt` (default: `pta_prob=0.5`, `pta_permutations=4`, `pta_sign_flip_prob=0.5`, `pta_vol_scale_prob=0.7`, scales `0.8/0.9/1.1/1.2`; sign-flip views are gated behind `--allow-short-positions`).
Cross-asset warm-up is run before return-head training by default (`--cross-asset-warmup-steps 180000`, `--cross-asset-corr-window 20`, `--cross-asset-warmup-batch-size 8`) and saves a stage-boundary checkpoint (`*_cross_asset_warmup.pt`).
Both portfolio heads use staged backbone fine-tuning: freeze for the first 500 steps, then unfreeze at 0.1× head LR.
Phase-2 early stopping uses step-based patience (`--patience-steps`, default `200`) on the real-data validation monitor.
Phase-2 training now applies a cold-start trim (`--state-warmup-steps`, default `8`) so loss/validation ignore initial cache-warmup timesteps.
Phase-2 completion is gated by validation Sharpe/drawdown by default (`--phase2-gate-min-sharpe 0.0`, `--phase2-gate-max-drawdown 0.25`; disable with `--disable-phase2-gate`).
Spec-aligned pretrain defaults are `PRETRAIN_STEPS=20000`, `PRETRAIN_BATCH=128`, `temperature=0.1`, and loss weights `lambda_contrastive=0.7` / `lambda_prediction=0.3` with `fda_loss_weight=0.5`. Override `PRETRAIN_BATCH` for smaller GPUs.
Portfolio defaults remain tuned for 24GB GPUs (RTX 3090): `BATCH_PORT=4`, dedicated `WARMUP_BATCH=8`, and gradient checkpointing enabled for portfolio training.
Enable pretrain compilation with `PRETRAIN_COMPILE=1` (example: `make pretrain PRETRAIN_COMPILE=1`).
`make pretrain`, `make portfolio-return`, and `make portfolio-sharpe` automatically resume from their stage-best checkpoints when those files exist.
`make backtest` automatically uses blended return/sharpe inference when `checkpoints/portfolio/sharpe_head_best.pt` exists.
Those training targets run in `tmux` under `systemd-inhibit` (sleep/idle blocked during runs).
Attach with `tmux attach -t pretrain` or `tmux attach -t portfolio`.

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

> v4 implementation status: core training pipelines include FDA/TCP/PTA/SWA support; remaining work is full-scale training, evaluation, and ablations.

### Data infrastructure

- [x] Build DataBus in Rust using finance-query as library dependency
- [x] Implement all 42 indicator computations with rolling z-score normalization
- [x] Implement RSS feed polling across all 31 sources with deduplication
- [x] Implement event detection (diff-based) for insider tx, ratings, filings, macro
- [x] Build historical data download pipeline for training corpus
- [x] Build feature assembly pipeline (184 per-asset + 96 market-wide)
- [x] Implement SQLite logging schema
- [x] Train BPE tokenizer on financial text corpus
- [x] Pre-fit 4-state HMM on historical regime data
- [x] Verify all finance-query API integrations end-to-end

### Core model

- [x] RA-GDN block in PyTorch with regime conditioning (α bias, β scaling)
- [x] FMoE with external regime-posterior routing
- [x] VSN input gating
- [x] Self-supervised pre-training scaffold with return/volatility/correlation objectives
- [x] Checkpoint save/resume with format version and compatibility checks
- [x] Split-aware shard evaluation
- [x] Temporal cascade: hourly/daily/weekly GDN stacks with surprise-gated context injection
- [x] Cross-asset fusion: 2-layer MHA with sector bias prior (110+ tokens)
- [x] EventEncoder: 18 event types → D-dim tokens with time decay
- [x] MarketReader text encoder (3-tier: headline / summary / deep)
- [ ] Run self-supervised backbone pre-training on H100 (~80 GPU-hours)
- [ ] Validate: prediction accuracy, recurrent state stability, regime detection quality

### Portfolio training

- [x] Portfolio output heads: Meridian-R (return) and Meridian-S (Sharpe) with leverage normalization
- [x] DSR (Differential Sharpe Ratio) loss function, turnover penalty, volatility auxiliary task
- [x] Build MeridianStrategy wrapper for finance-query backtesting engine
- [ ] Run return-head (~50K steps) and sharpe-head (~30K steps) training
- [ ] Run backtests vs all benchmarks (buy-and-hold SPY, equal-weight, risk parity)
- [ ] Run walk-forward validation, Monte Carlo analysis, regime-segmented analysis

### Adaptation and QAT

- [ ] Tier 1 — Surprise-gated plasticity (SGP): surprise signal modulates β update strength
- [ ] Tier 2 — Normalization adaptation: online running stats update
- [ ] Tier 3 — EWC + Ternary LoRA periodic consolidation
- [ ] Run meta-learning training (~40 GPU-hours)
- [ ] Run QAT training (~20 GPU-hours)
- [ ] Validate: quantized model performance within tolerance of full-precision
- [ ] Ablation studies: event stream value, text value, adaptation value

### Deployment

- [ ] NEON ternary inference kernel in C
- [ ] FAISS-backed episodic memory on ARM
- [ ] Full Pi 5 inference pipeline orchestration
- [ ] 10M+ step recurrent state drift test
- [ ] Paper trading engine with SQLite logging
- [ ] Deploy to Pi 5 with monitoring dashboard
- [ ] 72-hour continuous thermal and power stress test
- [ ] Begin 6-month paper trading period

### Paper trading and iteration

- [ ] Monitor all graduation criteria continuously (Section 15.4 of spec)
- [ ] Weekly review of adaptation health metrics (surprise signal, norm drift, consolidation loss)
- [ ] Monthly benchmark comparison (Meridian-R and Meridian-S vs SPY)
- [ ] Iterate on model if needed (retrain specific training components)
- [ ] After 6 months: evaluate all graduation criteria
- [ ] If passed: proceed to minimal live trading (~$10K initial capital)

---

## Contributing

Please read:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
