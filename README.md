# Meridian

Meridian is a regime-adaptive portfolio model targeting CPU-native inference (Raspberry Pi 5) with H100-based training.

This repository now includes a **pre-training setup foundation** aligned to `meridian-spec.md` and `research.md`:

- canonical architecture/training constants (RA-GDN, FMoE, adaptation phases)
- feature schema contracts for the 112 per-asset input dimensions
- deterministic train/validation/test split definitions
- a CLI to validate setup contracts before expensive training runs
- baseline open-source contribution and security documentation

## Repository layout

```text
.
├── src/
│   └── main.rs                                    # Rust finance-query collector
├── c/
│   ├── kernels/
│   │   ├── .gitkeep
│   │   └── ternary_neon.c                         # TODO: Section 8.2/8.3 kernel
│   └── runtime/
│       └── inference_runtime.c                    # TODO: Section 11 runtime glue
├── python/
│   ├── meridian_train/
│   │   ├── cli.py                                 # `meridian` CLI
│   │   ├── train/                                 # spec/train contracts and schedules
│   │   │   ├── feature_schema.py
│   │   │   ├── spec.py
│   │   │   ├── splits.py
│   │   │   └── qat.py                             # TODO: Section 8 + 9.5
│   │   ├── data/                                  # Historical/live data foundation
│   │   │   ├── environment.py
│   │   │   ├── workspace.py
│   │   │   ├── pipeline.py
│   │   │   ├── feature_assembly.py
│   │   │   ├── dataset_ingestion.py
│   │   │   ├── finance_query_collector.py
│   │   │   ├── marketreader.py
│   │   │   └── regime_hmm.py
│   │   ├── core/                                  # RA-GDN + regime-routed backbone
│   │   │   ├── regime_aware_backbone.py
│   │   │   ├── self_supervised_pretraining.py
│   │   │   ├── temporal_cascade.py                # TODO: Section 6.7
│   │   │   ├── cross_asset_fusion.py              # TODO: Section 6.8
│   │   │   └── episodic_memory.py                 # TODO: Section 6.9
│   │   ├── optimization/
│   │   │   ├── portfolio_heads.py                 # TODO: Section 6.10/10
│   │   │   └── losses.py                          # TODO: Section 9.3 losses
│   │   ├── adaptation/
│   │   │   └── adaptation.py                      # TODO: Section 7
│   │   ├── runtime/
│   │   │   ├── inference_pipeline.py              # TODO: Section 11
│   │   │   └── memory_store.py                    # TODO
│   │   ├── quantization/
│   │   │   └── ternary_kernels.py                 # TODO: C kernel bindings
│   │   └── *.py                                   # compatibility module aliases
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
meridian init-workspace --root .
```

Run setup checks:

```bash
meridian spec-check
meridian feature-vector-check --asset-class equity
meridian phase-plan --phase phase1
meridian split-check --timestamp 2024-08-15T13:30:00
meridian data-plan --assets AAPL,SPY,BTC-USD --output reports/data_pipeline_manifest.json
meridian marketreader-plan --corpus data/raw/news_corpus.jsonl
meridian marketreader-train --corpus data/raw/news_corpus.jsonl --epochs 1 --max-documents 5000
meridian hmm-spec --features vix_level,adx14,corr_dispersion,drawdown
meridian hmm-train --input data/processed/regime_features.jsonl --features vix_level,adx14,corr_dispersion,drawdown
meridian phase1-smoke --batch-size 2 --seq-len 16
meridian phase1-backbone-smoke --batch-size 2 --seq-len 16
meridian phase1-train-smoke --steps 5 --batch-size 2 --seq-len 16
meridian env-check --allow-non-h100
meridian ingest-jsonl --input data/raw/AAPL_features.jsonl --asset AAPL --output-root .
meridian collect-ingest --asset AAPL --start 2020-01-01 --end 2025-06-30
# optional passthrough:
# meridian collect-ingest --asset AAPL --start 2020-01-01 --end 2025-06-30 \
#   --command-template "cargo run --quiet -- --symbol {symbol} --start {start} --end {end} --asset-class equity --max-rows 365"
PYTHONPATH=python python -m unittest discover -s python/tests -q
```

Expected JSONL row schema for `ingest-jsonl`:

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

`collect-ingest` now defaults to a **Rust collector binary** at `src/main.rs`  
(`cargo run -- --symbol {symbol} --start {start} --end {end}`)  
that uses the `finance-query` Rust library directly and emits schema-compatible JSONL rows.  
Override with `--command-template` only if you want a custom collector command.

The Rust collector now pulls from multiple finance-query surfaces per asset: chart OHLCV,
indicator primitives, options chain, fundamentals, profile metadata, and news headlines.

`marketreader-train` writes tokenizer/model artifacts and training metrics for H100 training workflows.

`hmm-train` writes a pickled model checkpoint plus JSON metadata with state counts and transition stats.

`phase1-smoke` runs a synthetic forward-pass check for the Phase 1 Regime Detector + RA-GDN core.

`phase1-backbone-smoke` runs a synthetic check for the stacked Phase 1 backbone (8 RA-GDN blocks with regime-routed FMoE insertion points).

`phase1-train-smoke` runs a short synthetic self-supervised Phase 1 training loop and writes checkpoint/metrics artifacts, useful for validating end-to-end optimizer/checkpoint wiring before real shard-based training.

For H100 nodes, use the bootstrap script:

```bash
bash scripts/setup_h100_env.sh
```

## Contributing

Please read:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
