# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project conventions

### Code comments
- Do **not** reference `meridian-spec.md` or its section numbers (e.g. §8.5, §11.4) in code comments. The spec is an internal-only design doc; spec citations don't belong in shipped source.
- Keep comments concise — explain the *why* briefly, not the spec lineage.
- `meridian-spec.md` (v4.0) is the canonical architecture spec. Consult it for design intent, but never cite it from code.

## What this is

Meridian is a regime-adaptive portfolio model: it ingests every finance-query data stream and emits per-minute allocation weights for up to 100 US-listed assets. Training runs on H100/consumer GPUs; the inference target is a small x86-64 host — reference hardware is a **Framework Desktop (Ryzen AI Max+ 395, 128 GB unified)**, validated on a Framework Laptop 13.

**Governing priority: quality over efficiency.** The hardware budget is non-binding, so every design choice resolves toward predictive/risk-adjusted quality, not speed or size. Concretely: inference is **FP32, never quantized**; Block-8 text features use the **full harrier-oss + FinBERT teachers in-process** (not the distilled student); episodic memory uses **exact k-NN**. The Raspberry Pi 5 / ternary / NEON / distilled-student design is superseded and survives only as a constrained-deployment fallback (see `meridian-spec.md` §4 and Design Philosophy). When a change trades accuracy for compute/memory, default against it.

**Planned quality improvements** (all reuse the existing pretrained backbone — no Phase 1 retrain; see `meridian-spec.md` §18.0):
- *Inference:* FP32 (no quantization); run the full harrier-oss + FinBERT text teachers in-process (not the distilled student); exact `IndexFlat` episodic memory; blended Meridian-R + Meridian-S heads + test-time augmentation.
- *Phase 2:* portfolio training consumes **joint** FDA synth (`data/mmaps_synth_joint` from `make fda-joint`) via the dedicated `--synth-dir` arg — kept separate from `--data-dir` because per-asset and joint synth share the `FDA_SYNTH_` prefix but have different shapes. The per-asset synth in `mmaps_aug` is for pretraining only.
- Model-capacity scaling is **not** planned (would force a full retrain and likely overfit at ~1,695 bars/asset).

There are three languages, each owning a distinct stage:

- **Rust (`src/`)** — the *DataBus*. Polls/ingests all data feeds via the `finance-query` crate, detects events, assembles feature vectors, and emits schema-compatible JSONL. Built as both a binary (`meridian-bus`) and a PyO3 extension module (`meridian` / `meridian_py`).
- **Python (`python/`)** — all model code and training pipelines (PyTorch). This is where almost all active development happens.
- **C (`c/`)** — ARM NEON ternary inference kernels and Pi 5 runtime glue. Currently stubs, and now **deprecated** by the x86-64 + FP16 pivot; kept only as a constrained-deployment fallback.

## Critical: how the project is actually driven

- **`python/cli.py` is a stub** (`raise NotImplementedError`). The `meridian ...` CLI commands documented in `README.md` are **not** implemented. Do not assume they run.
- The **real entry points are the `Makefile` and direct module invocation**. Python modules are run as `PYTHONPATH=python python -m <package>.<module>` from the repo root (the Makefile's `$(PYTHON)` macro does `source .venv/bin/activate && PYTHONPATH=python python`). Always set `PYTHONPATH=python` when running modules by hand.
- Long-running training targets (`pretrain`, `portfolio-return`, `portfolio-sharpe`) launch inside `tmux` under `systemd-inhibit`. Attach with `tmux attach -t pretrain` or `tmux attach -t portfolio`.

## Common commands

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"   # or: uv sync
cp .env.example .env            # FRED/CoinGecko/HF tokens, SEC user-agent
```

### Build
```bash
cargo build --release           # Rust DataBus binary (used by `make collect`)
maturin develop                 # build + install the PyO3 extension into the venv
```

### Lint / type-check / test
```bash
ruff check python               # linter (config is defaults; no [tool.ruff] block)
mypy python                     # type checker
pytest python/tests             # test suite (intentionally minimal — see below)
pytest python/tests/test_dynamic_universe.py::<name>   # single test
```
This project deliberately keeps automated tests minimal; correctness is validated via smoke runs and backtests, not broad unit coverage. Do not assume an 80%-coverage gate applies here.

### Data + training pipeline (run in order)
```bash
make collect FDA_DEVICE=cuda    # Rust collect → JSONL → mmap shards → FDA synth → pack/upload → augment → stats
make stats                      # verify split shards
make pretrain                   # Phase 1: TCP self-supervised backbone
make portfolio-return           # Phase 2a: Meridian-R (return) head
make portfolio-sharpe           # Phase 2b: Meridian-S (Sharpe) head
make backtest                   # evaluate on held-out test region
```
- Override the GPU batch size with vars, e.g. `make pretrain PRETRAIN_BATCH=64`. Portfolio defaults are tuned for 24 GB (RTX 3090): `BATCH_PORT=4`. The Makefile auto-detects VRAM (`VRAM_TIER`) to pick devices.
- `make pretrain` / `make portfolio-*` auto-resume from their stage-best checkpoints when compatible ones exist (compatibility is checked via `CKPT_IS_COMPAT`); incompatible checkpoints are ignored, not overwritten.
- Standalone FDA targets: `make fda`, `make fda-joint`, `make pack-fda`, `make pack-fda-joint`, `make upload`. Text/regime prep: `make fnspid`, `make text-pca`, `make tokenizer`, `make marketreader`.
- `make clean-data` wipes derived data (caches, PCA bases, shards) but keeps expensive inputs (FNSPID, EDGAR text, FinBERT weights); `DEEP=1` also drops input caches.

## Data flow and directory layout

`data/`, `checkpoints/`, `reports/`, `logs/` are all gitignored (regenerated, never committed).

Shard directories (set in the Makefile, not discoverable from a single file):
- `data/mmaps/{train,validation,test}` — real per-asset memory-mapped `.f32bin` shards from `collect`.
- `data/mmaps_synth/train` — FDA diffusion-augmented synthetic shards.
- `data/mmaps_aug/{train,validation,test}` — merged real+synth; **this is the default `TRAIN_DATA`/`PRETRAIN_DATA`**. Override with `TRAIN_DATA=data/mmaps` to train on real-only.

Pipeline: Rust DataBus (`cargo run -- --symbol … --start … --end …`) emits JSONL → `python -m market_data.pipeline collect` / `dataset_ingestion` writes mmap shards → FDA (`train/fda_pipeline.py`) generates synthetic windows packed into WebDataset tars (optionally uploaded to HF) → augment merges them. The expected JSONL row schema (per-asset feature blocks) is documented in `README.md`.

## Architecture (the parts that span files)

**Feature/spec contracts are the source of truth — read these before touching model shapes:**
- `python/train/spec.py` — `MeridianSpec` dataclass: `d_model=768`, `n_assets=100` (max universe; a trained tensor dim — growing it requires retraining + widening Rust portfolio assembly), `n_regimes=4`, hybrid backbone layout (`n_gdn_blocks=8`, FMoE after GDN blocks 2/5/8, SWA after GDN blocks 3/7), expert config, etc.
- `python/train/feature_schema.py` — exact per-asset block offsets/dims (price_volume 10, technical 42, options 12, fundamental 30, risk 8, event_state 10, cross_asset 8, text 64 = 184 per-asset; 96 market-wide). Changing the Rust feature assembler and this file must stay in lockstep.

**Model core (`python/core/`):**
- `regime_aware_backbone.py` — RA-GDN blocks with regime conditioning (α bias / β scaling) + FMoE routed by an external regime posterior + VSN input gating. The hybrid RA-GDN + SWA stack.
- `temporal_cascade.py` — hourly/daily/weekly GDN stacks with surprise-gated context injection.
- `cross_asset_fusion.py` — 2-layer multi-head attention over 110+ tokens with a sector-bias prior.
- `event_encoder.py` — 18 event types → D-dim tokens with time decay.
- `self_supervised_pretraining.py` — TCP contrastive objective + auxiliary prediction target (forward realized volatility, not one-step return).
- `episodic_memory.py` — FAISS-backed episodic memory.

**Training stages (`python/train/`):**
- Phase 1 `pretrain.py` — self-supervised TCP backbone. Supports regime-balanced sampling, FDA-mixed batches, and an (unstable, off-by-default) adversarial regularization head. Early stopping gates on **validation loss**, not contrastive accuracy.
- Phase 2 `portfolio_train.py` — `--head return` then `--head sharpe`. Includes a cross-asset warm-up stage (saves `*_cross_asset_warmup.pt`), staged backbone fine-tuning (freeze then unfreeze at 0.1× head LR), PTA view augmentation (sector-preserving permutations from `portfolio_symbols.txt`), and FDA-mixed real/synth batches. Checkpoint selection uses validation DSR (return head) / rolling-Sharpe loss (sharpe head). Portfolio stage enforces an **exact 100-asset universe** and skips synthetic non-asset shards (`FDA_SYNTH_*`).
- `portfolio_heads.py` — Meridian-R / Meridian-S heads with leverage normalization. `losses.py` — DSR loss, turnover penalty, volatility auxiliary.

**Symbol universes:** `pretrain_symbols.txt` (pretraining) and `portfolio_symbols.txt` (the 100-asset portfolio universe) — distinct on purpose; pretraining and portfolio universes are decoupled.

## Colab notebooks

`colab_pretrain.ipynb`, `colab_portfolio_return.ipynb`, `colab_portfolio_sharpe.ipynb` mirror the Makefile training stages for GPU-rental environments. Keep their data-generation/setup cells in sync with the Makefile when pipeline defaults change.

## External services

- `services/finbert/` — Dockerized FinBERT sentiment sidecar; `make sidecar` starts it (port 8088), `make collect` waits on its health check.
- `make collect` requires a real SEC user-agent (`SEC_UA="Name email@host"`) or EDGAR 403s and text features collapse to zeros; it also probes a Block-8 text embedder endpoint (override/skip with `ALLOW_NO_EMBEDDER=1`).
- HF upload (real shards + packed FDA tars) needs `HF_TOKEN` with write scope; leave empty to run everything locally without uploading.
