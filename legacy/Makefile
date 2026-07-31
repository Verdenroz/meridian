SHELL        := /bin/bash

# Load .env if present so HF_TOKEN / HF_DATA_REPO / FDA_TAR_REPO flow through
# to subprocesses without the user having to `export` them every shell.
-include .env
export

PYTHON       := source .venv/bin/activate && PYTHONPATH=python python
DATA_DIR     := data/mmaps
SYNTH_DIR    := data/mmaps_synth
AUG_DATA_DIR := data/mmaps_aug
TRAIN_DATA   ?= $(AUG_DATA_DIR)
PRETRAIN_DATA ?= $(AUG_DATA_DIR)
SYMBOLS      ?= pretrain_symbols.txt
# Portfolio symbol list — 100 assets Meridian manages. Phase 2 fusion+head training.
# Must be a subset of SYMBOLS with exactly spec.n_assets entries.
PORTFOLIO_SYMBOLS ?= portfolio_symbols.txt
# Pretraining symbol list (S&P 500 + crypto + reference ETFs, ~500 symbols).
PRETRAIN_SYMBOLS ?= pretrain_symbols.txt
# Rust collector reads caches from this root ($MERIDIAN_DATA_ROOT, default ./data).
# FNSPID news cache (.fnspid_cache/) and macro cache (.macro_cache.json) live here.
DATA_ROOT    := data
FNSPID_END   ?= 2023-12-31
# SEC fair-access User-Agent for EDGAR. Override with a real contact if needed.
SEC_UA       ?= Harvey Tseng harveytseng2@gmail.com

# Block-8 text embedder (harrier-oss via OpenAI-compatible /v1/embeddings).
# `collect` probes this before the long run; set ALLOW_NO_EMBEDDER=1 to skip.
EMBED_MODEL    ?= harrier-oss:0.6b

EMBED_ENDPOINT ?= http://127.0.0.1:12434/v1/embeddings

# Propagate endpoint/model to the Rust collector.
export MERIDIAN_EMBED_ENDPOINT = $(EMBED_ENDPOINT)
export MERIDIAN_EMBED_MODEL    = $(EMBED_MODEL)

# Embedder request shaping (read by src/historical/embedder.rs).
EMBED_CHAR_BUDGET ?= 56000
EMBED_MAX_ITEMS   ?= 64
EMBED_MAX_RETRIES ?= 6
# Inter-request pause (ms). Default 0; raise if eGPU RPC timeouts reappear.
EMBED_BATCH_PAUSE_MS ?= 0
export MERIDIAN_EMBED_CHAR_BUDGET = $(EMBED_CHAR_BUDGET)
export MERIDIAN_EMBED_MAX_ITEMS   = $(EMBED_MAX_ITEMS)
export MERIDIAN_EMBED_MAX_RETRIES = $(EMBED_MAX_RETRIES)
export MERIDIAN_EMBED_BATCH_PAUSE_MS = $(EMBED_BATCH_PAUSE_MS)

# FinBERT sentiment sidecar. Idempotent. Sentiment degrades to neutral 0.0 if unavailable.
SIDECAR_IMAGE     ?= meridian-finbert
SIDECAR_NAME      ?= meridian-finbert
SIDECAR_PORT      ?= 8088
SIDECAR_HEALTH    ?= http://127.0.0.1:$(SIDECAR_PORT)/info
SIDECAR_WAIT_SECS ?= 180
CKPT_PRE     := checkpoints/pretrain
CKPT_PORT    := checkpoints/portfolio
BACKBONE     ?= $(CKPT_PRE)/backbone_pretrain.pt
RETURN_CKPT  ?= $(CKPT_PORT)/return_head_best.pt
PRETRAIN_BEST := $(CKPT_PRE)/backbone_pretrain.pt
RETURN_BEST   := $(CKPT_PORT)/return_head_best.pt
SHARPE_BEST   := $(CKPT_PORT)/sharpe_head_best.pt

# VRAM auto-detection — picks tiered batch/size defaults. All overridable via env.
VRAM_GB   := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($$1/1024)}')
VRAM_TIER := $(shell v="$(VRAM_GB)"; if [ -z "$$v" ] || [ "$$v" = "0" ]; then echo cpu; elif [ $$v -ge 75 ]; then echo h80; elif [ $$v -ge 35 ]; then echo a40; elif [ $$v -ge 22 ]; then echo mid; else echo low; fi)
$(info VRAM detected: $(VRAM_GB) GB → tier=$(VRAM_TIER))

# Portfolio (N=100) tiers
ifeq ($(VRAM_TIER),h80)
  BATCH_PORT      ?= 6
  BPTT_CHUNK_PORT ?= 16
  PTA_MICRO_BS    ?= 10
  REGIME_POOL     ?= 6000
  PRETRAIN_BATCH         ?= 128
  PRETRAIN_BPTT          ?= 0
  PRETRAIN_COMPILE_TIER  ?= 1
  PRETRAIN_GRAD_CKPT_TIER?= 0
else ifeq ($(VRAM_TIER),a40)
  BATCH_PORT      ?= 4
  BPTT_CHUNK_PORT ?= 16
  PTA_MICRO_BS    ?= 8
  REGIME_POOL     ?= 4000
  PRETRAIN_BATCH         ?= 128
  PRETRAIN_BPTT          ?= 0
  PRETRAIN_COMPILE_TIER  ?= 1
  PRETRAIN_GRAD_CKPT_TIER?= 1
else ifeq ($(VRAM_TIER),mid)
  BATCH_PORT      ?= 2
  BPTT_CHUNK_PORT ?= 8
  PTA_MICRO_BS    ?= 4
  REGIME_POOL     ?= 2000
  PRETRAIN_BATCH         ?= 128
  PRETRAIN_BPTT          ?= 32
  PRETRAIN_COMPILE_TIER  ?= 0
  PRETRAIN_GRAD_CKPT_TIER?= 1
else
  BATCH_PORT      ?= 1
  BPTT_CHUNK_PORT ?= 8
  PTA_MICRO_BS    ?= 2
  REGIME_POOL     ?= 1000
  PRETRAIN_BATCH         ?= 64
  PRETRAIN_BPTT          ?= 16
  PRETRAIN_COMPILE_TIER  ?= 0
  PRETRAIN_GRAD_CKPT_TIER?= 1
endif

FDA_BATCH        ?= 64
FDA_TARGET_VALID ?= 50000
FDA_DEVICE       ?= cuda
FDA_WINDOW       ?= 128
FDA_STRIDE       ?= 4
FDA_EPOCHS	   ?= 200
FDA_DIFFUSION    ?= 1000
FDA_MAX_ABS      ?= 10.0
FDA_SOFT_TARGET_PASS ?= 0.90
FDA_ETA          ?= 0.0
FDA_X0_CLIP      ?= 5.0
FDA_MAX_GEN_ROUNDS ?= 3000
FDA_SAMPLING     ?= 100

# Joint (multi-asset) FDA for portfolio training — 2D score network producing
# (N, T, F) windows. Each joint window packs n_assets, so far fewer are needed
# than the per-asset count (and each is ~9.5MB, so 50k would be ~476GB).
FDA_JOINT_OUT        ?= data/mmaps_synth_joint
FDA_JOINT_TARGET     ?= 1000
FDA_JOINT_N_ASSETS   ?= 100
FDA_JOINT_SEQ_LEN    ?= 64
FDA_JOINT_CORR_WIN   ?= 20
FDA_JOINT_ASSET_PASS ?= 0.75
FDA_JOINT_SAMPLING   ?= 250
FDA_JOINT_REF_WINDOWS ?= 512
# Keep the tuned effective batch (FDA_BATCH × this) on a small GPU: raise accum
# and lower FDA_BATCH instead of shrinking the effective batch.
FDA_JOINT_GRAD_ACCUM ?= 1
FDA_JOINT_TAR_DIR    ?= data/fda_joint_tars
FDA_JOINT_TAR_PATH   ?= fda_joint_tars

# HuggingFace upload settings. No-ops when HF_TOKEN is empty.
HF_DATA_REPO     ?= Verdenroz/meridian-data
HF_DATA_PRIVATE  ?= 1
FDA_TAR_DIR      ?= data/fda_tars
FDA_TAR_REPO     ?= $(HF_DATA_REPO)
FDA_TAR_PATH     ?= fda_tars
FDA_TAR_MIB      ?= 100
FDA_TAR_PRIVATE  ?= $(HF_DATA_PRIVATE)

PORT_EXTRA ?= --gradient-checkpointing --workers 2 --pta-deterministic --compile
PRETRAIN_STEPS ?= 20000
PRETRAIN_WORKERS ?= 8
PRETRAIN_COMPILE ?= $(PRETRAIN_COMPILE_TIER)
PRETRAIN_GRAD_CKPT ?= $(PRETRAIN_GRAD_CKPT_TIER)
PRETRAIN_SEQ_LEN ?= 64
PRETRAIN_LR ?= 3e-4
PRETRAIN_GRAD_CLIP ?= 1.0
PRETRAIN_TEMPERATURE ?= 0.1
PRETRAIN_WARMUP_STEPS ?= 3000
PRETRAIN_EVAL_EVERY ?= 500
PRETRAIN_EVAL_MAX_BATCHES ?= 50
PRETRAIN_SAVE_EVERY ?= 2500
PRETRAIN_LOG_EVERY ?= 10
PRETRAIN_CHECKPOINT_DIR ?= $(CKPT_PRE)
PRETRAIN_TSNE_REPORTS_DIR ?= reports/pretrain
PRETRAIN_TSNE_EVERY ?= 5000
PRETRAIN_POSITIVE_OFFSET ?= 64
PRETRAIN_NEGATIVE_MIN_SEPARATION ?= 50
PRETRAIN_MULTI_ASSET_CORR_THRESHOLD ?= 0.6
PRETRAIN_LAMBDA_CONTRASTIVE ?= 0.7
PRETRAIN_LAMBDA_PREDICTION ?= 0.3
PRETRAIN_LAMBDA_REGIME_CE ?= 0.2
PRETRAIN_FDA_LOSS_WEIGHT ?= 0.5
PRETRAIN_REGIME_BALANCED ?= 1
PRETRAIN_ADVERSARIAL_LAMBDA ?= 0.2
PRETRAIN_ADVERSARIAL_WARMUP ?= 2000
PRETRAIN_CROSS_ASSET_NO_SECTOR ?= 0
PRETRAIN_SYNTHETIC_FRACTION ?= 0.2
# Duration-matched to the intended ~10 H100h warm-up budget at measured
# throughput (~0.3 it/s); watch corr_frob and extend via resume if still falling.
WARMUP_STEPS ?= 15000
WARMUP_BATCH ?= 8
# Joint FDA windows are asset-anonymous: safe at a high mix in the identity-free
# corr warm-up, lower in head training where synth return fidelity matters.
FDA_MIX        ?= 0.25
WARMUP_FDA_MIX ?= 0.5

INHIBIT    := systemd-inhibit --what=sleep:idle --who=meridian --why=gpu-training
TMUX_RUN    = tmux kill-session -t $1 2>/dev/null || true; \
              tmux new-session -d -s $1 "$(INHIBIT) bash -c '$2'"
CKPT_IS_COMPAT = $(PYTHON) -c "import sys, torch; d = torch.load(sys.argv[1], map_location='cpu', weights_only=True); ok = d.get('format_version') == 2 and d.get('architecture_version') == 'v4.0' and d.get('pretraining_objective') == 'tcp'; raise SystemExit(0 if ok else 1)"

.PHONY: collect clean-data sidecar sidecar-stop fnspid text-pca stats fda fda-joint pack-fda pack-fda-joint upload-fda upload augment-data stats-aug \
        text-corpus tokenizer marketreader setup \
        pretrain portfolio-return portfolio-sharpe backtest

# Wipe derived data so next `make collect` rebuilds from scratch.
# Deletes: embed caches, PCA bases, output shards, regime labels.
# Keeps: FNSPID, EDGAR text, sentiment cache, FinBERT weights.
# DEEP=1 also drops input caches (FNSPID, EDGAR text, sentiment).
clean-data:
	@scope="embed_cache + block8_text_pca.bin + market_news_pca.bin + macro/news caches + $(DATA_DIR)/{train,validation,test} + $(SYNTH_DIR) + $(AUG_DATA_DIR) + regime labels"; \
	if [ "$(DEEP)" = "1" ]; then scope="$$scope + .fnspid_cache + edgar_text + sentiment_cache (DEEP)"; fi; \
	echo "⚠  clean-data will PERMANENTLY delete derived data:"; \
	echo "     $$scope"; \
	if [ -n "$$CLEAN_DATA_YES" ]; then \
		echo "   CLEAN_DATA_YES set — proceeding without prompt."; \
	elif [ ! -t 0 ]; then \
		echo "   ERROR: refusing to run non-interactively (stdin is not a terminal)."; \
		echo "          clean-data is destructive and must be confirmed by a human at the keyboard"; \
		echo "          — this is what stops a stray script/agent/Make dependency from wiping data."; \
		echo "          To run it deliberately from a script: make clean-data CLEAN_DATA_YES=1"; \
		exit 1; \
	else \
		printf '   Type exactly \"delete\" to proceed (anything else aborts): '; \
		read ans; \
		if [ "$$ans" != "delete" ]; then echo "   aborted — nothing was deleted."; exit 1; fi; \
	fi
	@echo "clearing stale embedding caches + PCA bases…"
	@rm -rf $(DATA_ROOT)/embed_cache
	@rm -f  $(DATA_ROOT)/block8_text_pca.bin $(DATA_ROOT)/market_news_pca.bin
	@echo "clearing refetchable macro/news caches…"
	@rm -f  $(DATA_ROOT)/.macro_cache.json $(DATA_ROOT)/.market_news_daily.json
	@echo "clearing output shards (real / synth / augmented) + regime labels…"
	@rm -rf $(DATA_DIR)/train $(DATA_DIR)/validation $(DATA_DIR)/test
	@rm -rf $(SYNTH_DIR) $(AUG_DATA_DIR)
	@rm -f  $(DATA_ROOT)/spy_regime_labels_*.npy
	@if [ "$(DEEP)" = "1" ]; then \
		echo "DEEP=1 → also dropping FNSPID news, EDGAR text, and sentiment caches…"; \
		rm -rf $(DATA_ROOT)/.fnspid_cache $(DATA_ROOT)/edgar_text $(DATA_ROOT)/sentiment_cache; \
		echo "  (kept FinBERT weights at $(DATA_ROOT)/finbert — rm by hand to force re-download)"; \
	fi
	@echo "✓ clean-data done — run 'make collect' for a fresh build"

# ── One-time setup ────────────────────────────────────────────────────────────
# Run ONCE before the first `make collect`. Downloads FNSPID news corpus and
# fits the Block-8 PCA(64) basis. Requires the embedder endpoint up (llama-swap :12434).
setup:
	@echo "=== Meridian one-time setup ==="
	@echo "  step 1/2: downloading FNSPID news corpus ($(SYMBOLS))…"
	@$(MAKE) fnspid
	@echo "  step 2/2: fitting Block-8 PCA(64) basis (one-time)…"
	@$(MAKE) text-pca
	@echo ""
	@echo "✓ setup complete — PCA basis persisted to $(DATA_ROOT)/block8_text_pca.bin"
	@echo "  Changing portfolio_symbols.txt does NOT require re-running setup."
	@echo "  Next: make collect            (default: $(SYMBOLS))"
	@echo "        make pretrain           # Phase 1 TCP backbone"
	@echo "        make portfolio-return   # Phase 2 return head ($(PORTFOLIO_SYMBOLS))"
	@echo "        make portfolio-sharpe   # Phase 2 sharpe head"

# Train-data pipeline. Collects feature shards for Phase 1 (all symbols) and Phase 2 (subset).
# Override: `make collect SYMBOLS=portfolio_symbols.txt`
# PCA basis is a one-time artifact from `make setup`; per-symbol collection loads it from disk.
collect:
	@ua="$${MERIDIAN_SEC_UA:-$(SEC_UA)}"; \
	if printf '%s' "$$ua" | grep -qiE 'meridian\.local|example\.(com|org|net)|admin@|^[[:space:]]*$$' \
		|| ! printf '%s' "$$ua" | grep -q '@'; then \
		echo "ERROR: SEC User-Agent is a placeholder or missing a real contact:"; \
		echo "         MERIDIAN_SEC_UA / SEC_UA = '$$ua'"; \
		echo "       SEC 403s such requests, so EDGAR text collapses to edgar=0."; \
		echo "       Set a routable contact, e.g.:"; \
		echo "         make collect SEC_UA=\"Jane Doe jane@company.com\""; \
		exit 1; \
	fi
	@if [ "$(ALLOW_NO_EMBEDDER)" = "1" ]; then \
		echo "⚠ ALLOW_NO_EMBEDDER=1 — skipping embedder probe; Block 8 may be zeros."; \
	elif curl -fsS -m 10 -X POST "$(EMBED_ENDPOINT)" \
			-H 'Content-Type: application/json' \
			-d '{"model":"$(EMBED_MODEL)","input":["meridian embedder readiness probe"]}' \
			>/dev/null 2>&1; then \
		echo "✓ Block-8 embedder reachable at $(EMBED_ENDPOINT) ($(EMBED_MODEL))"; \
	else \
		echo "ERROR: Block-8 embedder not reachable at $(EMBED_ENDPOINT) (model $(EMBED_MODEL))."; \
		echo "       harrier-oss must be served (e.g. llama-swap on :12434), or all"; \
		echo "       per-symbol Block-8 vectors collapse to zeros."; \
		echo "       Start the embedder, or rerun with ALLOW_NO_EMBEDDER=1."; \
		exit 1; \
	fi
	@echo "  embedder shaping: char_budget=$(EMBED_CHAR_BUDGET) max_items=$(EMBED_MAX_ITEMS) max_retries=$(EMBED_MAX_RETRIES)"
	cargo build --release
	@mkdir -p $(DATA_DIR)/train $(DATA_DIR)/validation $(DATA_DIR)/test
	@$(MAKE) sidecar
	MERIDIAN_SEC_UA="$${MERIDIAN_SEC_UA:-$(SEC_UA)}" $(PYTHON) -m market_data.pipeline collect \
		--symbols-file $(SYMBOLS) --interval 1d \
		--data-dir $(DATA_DIR) --workers 8 --skip-existing \
		--max-retries 10 --retry-base-seconds 5 --retry-max-seconds 180 \
		--retry-jitter-seconds 30 --request-permits 2
	@$(MAKE) fda
	@$(MAKE) pack-fda
	@$(MAKE) upload
	@$(MAKE) fda-joint
	@$(MAKE) pack-fda-joint
	@$(MAKE) augment-data
	@$(MAKE) stats-aug
	@echo ""
	@echo "✓ make collect complete — training DATA is ready:"
	@echo "  real shards  : $(DATA_DIR)/{train,validation,test}"
	@echo "  synth shards : $(SYNTH_DIR)/train (per-asset, for pretrain)"
	@echo "  joint synth  : $(FDA_JOINT_OUT)/train (joint, for portfolio --synth-dir)"
	@echo "  merged       : $(AUG_DATA_DIR)/{train,validation,test} (used by TRAIN_DATA)"
	@echo ""
	@echo "  next — model training (GPU, run each in order, monitor via tmux):"
	@echo "    make pretrain          # Phase 1 TCP self-supervised backbone"
	@echo "    make portfolio-return  # Phase 2 return head"
	@echo "    make portfolio-sharpe  # Phase 2 sharpe head"
	@echo "    make backtest          # evaluate on the held-out test region"

# Download FNSPID news corpus, filter to universe, write per-symbol caches to .fnspid_cache/.
fnspid:
	$(PYTHON) -m market_data.fnspid_ingest \
		--symbols-file $(SYMBOLS) --data-dir $(DATA_ROOT) --end $(FNSPID_END)

# Fit the global Block-8 text-embedding PCA(64) basis — a one-time setup step.
# The 1024-dim embedding is PCA-reduced to 64 dims and persisted to
# $(DATA_ROOT)/block8_text_pca.bin. Requires the embedder endpoint up.
# Non-fatal: on failure writes a zero basis (Block 8 degrades to zeros).
text-pca:
	@if [ ! -x ./target/release/meridian-bus ]; then cargo build --release; fi
	@MERIDIAN_DATA_ROOT=$(DATA_ROOT) ./target/release/meridian-bus fit-text-pca \
		|| echo "⚠ fit-text-pca failed — Block 8 will be zeros for this run"

# ── MarketReader distillation ───────────────────────────────────────────────
# Distills ~3.5M param MarketReader from harrier-oss + FinBERT teachers.
# Pipeline: text-corpus → tokenizer → build-targets → distill
TOKENIZER   ?= models/tokenizer/bpe_4096.json
MR_CORPUS   ?= $(DATA_ROOT)/text_corpus
MR_TARGETS  ?= $(DATA_ROOT)/text_corpus/targets.jsonl
MR_CKPT     ?= checkpoints/marketreader/marketreader_best.pt
MR_STEPS    ?= 5000

text-corpus:
	$(PYTHON) -m train.train_marketreader --mode dump-corpus \
		--fnspid-dir $(DATA_ROOT)/.fnspid_cache --edgar-dir $(DATA_ROOT)/edgar_text \
		--corpus-dir $(MR_CORPUS)

tokenizer: text-corpus
	$(PYTHON) -m train.train_tokenizer --corpus-dir $(MR_CORPUS) --output $(TOKENIZER)

marketreader: tokenizer
	$(PYTHON) -m train.train_marketreader --mode build-targets \
		--corpus-dir $(MR_CORPUS) --targets $(MR_TARGETS) \
		--embed-endpoint $(EMBED_ENDPOINT) --embed-model $(EMBED_MODEL) \
		--sentiment-endpoint http://127.0.0.1:$(SIDECAR_PORT)/predict
	$(PYTHON) -m train.train_marketreader --mode distill \
		--targets $(MR_TARGETS) --tokenizer $(TOKENIZER) --out $(MR_CKPT) --steps $(MR_STEPS)
	@echo "✓ MarketReader distilled → $(MR_CKPT)"

# Start FinBERT sentiment sidecar. Idempotent: reuses running container.
# Mounts data/finbert weight cache to skip re-download.
sidecar:
	@set -u; \
	if ! command -v docker >/dev/null 2>&1; then \
		echo "⚠ docker not found — FinBERT sidecar NOT started; sentiment will be neutral 0.0."; \
		echo "  install Docker (or start the sidecar manually) for real sentiment scoring."; \
		exit 0; \
	fi; \
	if [ -n "$$(docker ps -q -f name=^/$(SIDECAR_NAME)$$)" ]; then \
		echo "FinBERT sidecar '$(SIDECAR_NAME)' already running."; \
	elif [ -n "$$(docker ps -aq -f name=^/$(SIDECAR_NAME)$$)" ]; then \
		echo "starting existing FinBERT sidecar container…"; \
		docker start $(SIDECAR_NAME) >/dev/null; \
	else \
		if ! docker image inspect $(SIDECAR_IMAGE) >/dev/null 2>&1; then \
			echo "building $(SIDECAR_IMAGE) image (first run, CPU-only)…"; \
			docker build -t $(SIDECAR_IMAGE) services/finbert; \
		fi; \
		echo "launching FinBERT sidecar on :$(SIDECAR_PORT)…"; \
		docker run -d --name $(SIDECAR_NAME) -p $(SIDECAR_PORT):80 \
			-v "$$PWD/data/finbert:/data" $(SIDECAR_IMAGE) >/dev/null; \
	fi; \
	printf "waiting for FinBERT to load"; \
	for i in $$(seq 1 $(SIDECAR_WAIT_SECS)); do \
		if curl -fsS $(SIDECAR_HEALTH) >/dev/null 2>&1; then \
			printf " ✓ ready (%s)\n" "$(SIDECAR_HEALTH)"; \
			exit 0; \
		fi; \
		printf "."; sleep 1; \
	done; \
	printf "\n⚠ sidecar not ready within $(SIDECAR_WAIT_SECS)s; sentiment may degrade to 0.0 (docker logs $(SIDECAR_NAME)).\n"; \
	exit 0

sidecar-stop:
	@docker rm -f $(SIDECAR_NAME) 2>/dev/null && echo "stopped $(SIDECAR_NAME)" || echo "$(SIDECAR_NAME) not running"

stats:
	$(PYTHON) -m market_data.pipeline stats --data-dir $(DATA_DIR)

fda:
	$(PYTHON) -m train.fda_pipeline \
		--input-dir $(DATA_DIR)/train --output-dir $(SYNTH_DIR) \
		--window-len $(FDA_WINDOW) --stride $(FDA_STRIDE) \
		--epochs $(FDA_EPOCHS) --batch-size $(FDA_BATCH) \
		--target-valid-windows $(FDA_TARGET_VALID) --device $(FDA_DEVICE) \
		--max-abs-feature $(FDA_MAX_ABS) --diffusion-steps $(FDA_DIFFUSION) \
		--soft-target-pass $(FDA_SOFT_TARGET_PASS) \
		--eta $(FDA_ETA) --x0-clip $(FDA_X0_CLIP) \
		--sampling-steps $(FDA_SAMPLING) \
		--max-generation-rounds $(FDA_MAX_GEN_ROUNDS)
	@set -euo pipefail; \
	if [ ! -f "$(SYNTH_DIR)/fda_manifest.json" ]; then \
		echo "missing $(SYNTH_DIR)/fda_manifest.json; FDA likely failed"; \
		exit 1; \
	fi; \
	shopt -s nullglob; \
	synth_files=($(SYNTH_DIR)/train/*.f32bin); \
	if [ $${#synth_files[@]} -eq 0 ]; then \
		echo "no synthetic shards in $(SYNTH_DIR)/train; FDA likely failed"; \
		exit 1; \
	fi; \
	echo "FDA complete: $${#synth_files[@]} synthetic shards available"

# Joint multi-asset FDA for portfolio head + cross-asset warm-up.
fda-joint:
	$(PYTHON) -m train.fda_pipeline --joint \
		--input-dir $(DATA_DIR)/train --output-dir $(FDA_JOINT_OUT) \
		--joint-n-assets $(FDA_JOINT_N_ASSETS) \
		--joint-seq-len $(FDA_JOINT_SEQ_LEN) \
		--joint-corr-window $(FDA_JOINT_CORR_WIN) \
		--joint-asset-pass-frac $(FDA_JOINT_ASSET_PASS) \
		--joint-ref-windows $(FDA_JOINT_REF_WINDOWS) \
		--joint-grad-accum $(FDA_JOINT_GRAD_ACCUM) \
		--stride $(FDA_STRIDE) \
		--epochs $(FDA_EPOCHS) --batch-size $(FDA_BATCH) \
		--target-valid-windows $(FDA_JOINT_TARGET) --device $(FDA_DEVICE) \
		--sampling-steps $(FDA_JOINT_SAMPLING) \
		--max-abs-feature $(FDA_MAX_ABS) --diffusion-steps $(FDA_DIFFUSION) \
		--soft-target-pass $(FDA_SOFT_TARGET_PASS) \
		--eta $(FDA_ETA) --x0-clip $(FDA_X0_CLIP) \
		--max-generation-rounds $(FDA_MAX_GEN_ROUNDS)
	@set -euo pipefail; \
	if [ ! -f "$(FDA_JOINT_OUT)/fda_manifest.json" ]; then \
		echo "missing $(FDA_JOINT_OUT)/fda_manifest.json; joint FDA likely failed"; \
		exit 1; \
	fi; \
	shopt -s nullglob; \
	synth_files=($(FDA_JOINT_OUT)/train/*.f32bin); \
	if [ $${#synth_files[@]} -eq 0 ]; then \
		echo "no joint shards in $(FDA_JOINT_OUT)/train; joint FDA likely failed"; \
		exit 1; \
	fi; \
	echo "joint FDA complete: $${#synth_files[@]} joint shards available"

# Upload already-packed tars from FDA_TAR_DIR to HuggingFace.
upload-fda:
	@set -euo pipefail; \
	if [ -z "$${HF_TOKEN:-}" ]; then \
		echo "HF_TOKEN not set — set it in .env or export it first"; \
		exit 1; \
	fi; \
	shopt -s nullglob; \
	tar_files=($(FDA_TAR_DIR)/*.tar); \
	if [ $${#tar_files[@]} -eq 0 ]; then \
		echo "no tars in $(FDA_TAR_DIR); run 'make pack-fda' first"; \
		exit 1; \
	fi; \
	echo "uploading $${#tar_files[@]} tar(s) from $(FDA_TAR_DIR) → $(FDA_TAR_REPO)/$(FDA_TAR_PATH)"; \
	private_flag=""; \
	if [ "$(FDA_TAR_PRIVATE)" = "1" ]; then private_flag="True"; else private_flag="False"; fi; \
	$(PYTHON) -c "\
from tools.hf_common import upload_folder_to_dataset; \
from pathlib import Path; \
url = upload_folder_to_dataset(Path('$(FDA_TAR_DIR)'), '$(FDA_TAR_REPO)', private=$$private_flag, allow_patterns=['*.tar', '*.json'], path_in_repo='$(FDA_TAR_PATH)'); \
print('uploaded →', url)"

pack-fda:
	@set -euo pipefail; \
	if [ ! -d "$(SYNTH_DIR)/train" ]; then \
		echo "missing $(SYNTH_DIR)/train; run 'make fda' first"; \
		exit 1; \
	fi; \
	shopt -s nullglob; \
	synth_files=($(SYNTH_DIR)/train/FDA_SYNTH_*.f32bin); \
	if [ $${#synth_files[@]} -eq 0 ]; then \
		echo "no FDA_SYNTH_* shards under $(SYNTH_DIR)/train; run 'make fda' first"; \
		exit 1; \
	fi; \
	upload_args=""; \
	if [ -n "$${HF_TOKEN:-}" ]; then \
		upload_args="--upload-repo $(FDA_TAR_REPO) --upload-path-in-repo $(FDA_TAR_PATH)"; \
		if [ "$(FDA_TAR_PRIVATE)" = "1" ]; then \
			upload_args="$$upload_args --upload-private"; \
		fi; \
		echo "uploading to HuggingFace: $(FDA_TAR_REPO)/$(FDA_TAR_PATH) (private=$(FDA_TAR_PRIVATE))"; \
	else \
		echo "HF_TOKEN not set — packing locally only, skipping upload"; \
		echo "  (export HF_TOKEN=hf_... to push to $(FDA_TAR_REPO))"; \
	fi; \
	$(PYTHON) -m tools.pack_fda_shards \
		--src $(SYNTH_DIR)/train \
		--out $(FDA_TAR_DIR) \
		--target-mib $(FDA_TAR_MIB) \
		--source-manifest $(SYNTH_DIR)/fda_manifest.json \
		$$upload_args

# Pack + upload joint FDA shards to a distinct repo path so the portfolio
# notebook pulls them separately from the per-asset (TCP) shards.
pack-fda-joint:
	@set -euo pipefail; \
	if [ ! -d "$(FDA_JOINT_OUT)/train" ]; then \
		echo "missing $(FDA_JOINT_OUT)/train; run 'make fda-joint' first"; \
		exit 1; \
	fi; \
	shopt -s nullglob; \
	synth_files=($(FDA_JOINT_OUT)/train/FDA_SYNTH_*.f32bin); \
	if [ $${#synth_files[@]} -eq 0 ]; then \
		echo "no FDA_SYNTH_* shards under $(FDA_JOINT_OUT)/train; run 'make fda-joint' first"; \
		exit 1; \
	fi; \
	upload_args=""; \
	if [ -n "$${HF_TOKEN:-}" ]; then \
		upload_args="--upload-repo $(FDA_TAR_REPO) --upload-path-in-repo $(FDA_JOINT_TAR_PATH)"; \
		if [ "$(FDA_TAR_PRIVATE)" = "1" ]; then \
			upload_args="$$upload_args --upload-private"; \
		fi; \
		echo "uploading to HuggingFace: $(FDA_TAR_REPO)/$(FDA_JOINT_TAR_PATH) (private=$(FDA_TAR_PRIVATE))"; \
	else \
		echo "HF_TOKEN not set — packing locally only, skipping upload"; \
	fi; \
	$(PYTHON) -m tools.pack_fda_shards \
		--src $(FDA_JOINT_OUT)/train \
		--out $(FDA_JOINT_TAR_DIR) \
		--target-mib $(FDA_TAR_MIB) \
		--source-manifest $(FDA_JOINT_OUT)/fda_manifest.json \
		$$upload_args

# Push real shards to HF_DATA_REPO. No-op when HF_TOKEN is empty.
upload:
	@set -euo pipefail; \
	if [ -z "$${HF_TOKEN:-}" ]; then \
		echo "HF_TOKEN not set — skipping real-shard upload to $(HF_DATA_REPO)"; \
		echo "  (set HF_TOKEN in .env to enable)"; \
		exit 0; \
	fi; \
	if [ ! -d "$(DATA_DIR)/train" ]; then \
		echo "missing $(DATA_DIR)/train; run 'make collect' first"; \
		exit 1; \
	fi; \
	echo "uploading real shards to HuggingFace: $(HF_DATA_REPO) (private=$(HF_DATA_PRIVATE))"; \
	$(PYTHON) -m tools.hf_upload_shards \
		--src $(DATA_DIR) \
		--repo $(HF_DATA_REPO) \
		$$( [ "$(HF_DATA_PRIVATE)" = "1" ] && echo --private )

augment-data:
	@set -euo pipefail; \
	if [ ! -d "$(DATA_DIR)/train" ]; then \
		echo "missing $(DATA_DIR)/train; run 'make collect' first"; \
		exit 1; \
	fi
	$(PYTHON) -m tools.check_shard_sizes --dir $(DATA_DIR)/train
	@set -euo pipefail; \
	mkdir -p $(AUG_DATA_DIR)/train $(AUG_DATA_DIR)/validation $(AUG_DATA_DIR)/test; \
	find $(AUG_DATA_DIR)/train     -maxdepth 1 -name "*.f32bin" -delete 2>/dev/null || true; \
	find $(AUG_DATA_DIR)/validation -maxdepth 1 -name "*.f32bin" -delete 2>/dev/null || true; \
	find $(AUG_DATA_DIR)/test       -maxdepth 1 -name "*.f32bin" -delete 2>/dev/null || true; \
	find $(DATA_DIR)/train     -maxdepth 1 -name "*.f32bin" -print0 | xargs -0 -r cp --target-directory=$(AUG_DATA_DIR)/train/; \
	find $(DATA_DIR)/validation -maxdepth 1 -name "*.f32bin" -print0 | xargs -0 -r cp --target-directory=$(AUG_DATA_DIR)/validation/; \
	find $(DATA_DIR)/test       -maxdepth 1 -name "*.f32bin" -print0 | xargs -0 -r cp --target-directory=$(AUG_DATA_DIR)/test/; \
	synth_count=$$(find $(SYNTH_DIR)/train -maxdepth 1 -name "*.f32bin" 2>/dev/null | wc -l); \
	if [ "$$synth_count" -eq 0 ]; then \
		echo "no synthetic shards found in $(SYNTH_DIR)/train; run 'make fda' first"; \
		exit 1; \
	fi; \
	find $(SYNTH_DIR)/train -maxdepth 1 -name "*.f32bin" -print0 | xargs -0 -r cp --target-directory=$(AUG_DATA_DIR)/train/; \
	train_count=$$(find $(AUG_DATA_DIR)/train     -maxdepth 1 -name "*.f32bin" | wc -l); \
	val_count=$$(find   $(AUG_DATA_DIR)/validation -maxdepth 1 -name "*.f32bin" | wc -l); \
	test_count=$$(find  $(AUG_DATA_DIR)/test        -maxdepth 1 -name "*.f32bin" | wc -l); \
	echo "augmented dataset ready: train=$$train_count val=$$val_count test=$$test_count"

stats-aug:
	$(PYTHON) -m market_data.pipeline stats --data-dir $(AUG_DATA_DIR)

pretrain:
	@set -euo pipefail; \
	resume_arg=""; \
	compile_arg=""; \
	mem_args=""; \
	if [ -f "$(PRETRAIN_BEST)" ] && $(CKPT_IS_COMPAT) "$(PRETRAIN_BEST)" >/dev/null 2>&1; then \
		resume_arg="--resume-from $(PRETRAIN_BEST)"; \
		echo "resuming pretrain from $(PRETRAIN_BEST)"; \
	elif [ -f "$(PRETRAIN_BEST)" ]; then \
		echo "ignoring incompatible pretrain checkpoint at $(PRETRAIN_BEST)"; \
	fi; \
	if [ "$(PRETRAIN_COMPILE)" = "1" ]; then \
		compile_arg="--compile"; \
		echo "enabling torch.compile for pretrain"; \
	fi; \
	if [ "$(PRETRAIN_GRAD_CKPT)" = "1" ]; then \
		mem_args="$$mem_args --gradient-checkpointing"; \
	fi; \
	if [ "$(PRETRAIN_BPTT)" != "0" ]; then \
		mem_args="$$mem_args --bptt-horizon $(PRETRAIN_BPTT)"; \
	fi; \
	antishortcut_args=""; \
	if [ "$(PRETRAIN_REGIME_BALANCED)" = "1" ]; then \
		antishortcut_args="$$antishortcut_args --regime-balanced-sampling --synthetic-fraction $(PRETRAIN_SYNTHETIC_FRACTION)"; \
	fi; \
	if [ "$(PRETRAIN_CROSS_ASSET_NO_SECTOR)" = "1" ]; then \
		antishortcut_args="$$antishortcut_args --cross-asset-no-sector"; \
	fi; \
	if [ -n "$(PRETRAIN_ADVERSARIAL_LAMBDA)" ] && [ "$(PRETRAIN_ADVERSARIAL_LAMBDA)" != "0" ] && [ "$(PRETRAIN_ADVERSARIAL_LAMBDA)" != "0.0" ]; then \
		antishortcut_args="$$antishortcut_args --adversarial-lambda $(PRETRAIN_ADVERSARIAL_LAMBDA) --adversarial-warmup-steps $(PRETRAIN_ADVERSARIAL_WARMUP)"; \
	fi; \
	$(call TMUX_RUN,pretrain,source .venv/bin/activate && PYTHONPATH=python python -m train.pretrain \
		--data-dir $(PRETRAIN_DATA) --symbols-file $(PRETRAIN_SYMBOLS) --steps $(PRETRAIN_STEPS) --batch-size $(PRETRAIN_BATCH) --seq-len $(PRETRAIN_SEQ_LEN) \
		--lr $(PRETRAIN_LR) --warmup-steps $(PRETRAIN_WARMUP_STEPS) --grad-clip $(PRETRAIN_GRAD_CLIP) \
		--temperature $(PRETRAIN_TEMPERATURE) --lambda-contrastive $(PRETRAIN_LAMBDA_CONTRASTIVE) --lambda-prediction $(PRETRAIN_LAMBDA_PREDICTION) \
		--lambda-regime-ce $(PRETRAIN_LAMBDA_REGIME_CE) \
		--fda-loss-weight $(PRETRAIN_FDA_LOSS_WEIGHT) --positive-offset $(PRETRAIN_POSITIVE_OFFSET) --multi-asset-corr-threshold $(PRETRAIN_MULTI_ASSET_CORR_THRESHOLD) \
		--negative-min-separation $(PRETRAIN_NEGATIVE_MIN_SEPARATION) \
		--eval-every $(PRETRAIN_EVAL_EVERY) --eval-max-batches $(PRETRAIN_EVAL_MAX_BATCHES) \
		--save-every $(PRETRAIN_SAVE_EVERY) --log-every $(PRETRAIN_LOG_EVERY) --checkpoint-dir $(PRETRAIN_CHECKPOINT_DIR) \
		--tsne-reports-dir $(PRETRAIN_TSNE_REPORTS_DIR) --tsne-every $(PRETRAIN_TSNE_EVERY) \
		--workers $(PRETRAIN_WORKERS) \
		$$resume_arg $$compile_arg $$mem_args $$antishortcut_args)
	@echo "→ tmux attach -t pretrain"

portfolio-return:
	@set -euo pipefail; \
	resume_arg=""; \
	if [ -f "$(RETURN_BEST)" ] && $(CKPT_IS_COMPAT) "$(RETURN_BEST)" >/dev/null 2>&1; then \
		resume_arg="--resume-backbone $(RETURN_BEST)"; \
		echo "resuming portfolio-return from $(RETURN_BEST)"; \
	elif [ -f "$(BACKBONE)" ] && $(CKPT_IS_COMPAT) "$(BACKBONE)" >/dev/null 2>&1; then \
		resume_arg="--resume-backbone $(BACKBONE)"; \
		echo "starting portfolio-return from $(BACKBONE)"; \
	elif [ -f "$(BACKBONE)" ]; then \
		echo "ignoring incompatible backbone checkpoint at $(BACKBONE)"; \
	fi; \
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && PYTHONPATH=python python -m train.portfolio_train \
		--head return --data-dir $(TRAIN_DATA) --synth-dir $(FDA_JOINT_OUT) --symbols-file $(PORTFOLIO_SYMBOLS) \
		--steps 50000 --cross-asset-warmup-steps $(WARMUP_STEPS) --cross-asset-warmup-batch-size $(WARMUP_BATCH) --batch-size $(BATCH_PORT) --seq-len 64 \
		--fda-mix-prob $(FDA_MIX) --warmup-fda-mix-prob $(WARMUP_FDA_MIX) \
		--bptt-chunk $(BPTT_CHUNK_PORT) --pta-micro-batch-size $(PTA_MICRO_BS) --regime-pool-size $(REGIME_POOL) \
		--lr 3e-5 --log-every 10 --eval-every 250 --save-every 250 --checkpoint-dir $(CKPT_PORT) $(PORT_EXTRA) \
		$$resume_arg)
	@echo "→ tmux attach -t portfolio"

portfolio-sharpe:
	@set -euo pipefail; \
	resume_arg=""; \
	if [ -f "$(SHARPE_BEST)" ] && $(CKPT_IS_COMPAT) "$(SHARPE_BEST)" >/dev/null 2>&1; then \
		resume_arg="--resume-backbone $(SHARPE_BEST)"; \
		echo "resuming portfolio-sharpe from $(SHARPE_BEST)"; \
	elif [ -f "$(RETURN_CKPT)" ] && $(CKPT_IS_COMPAT) "$(RETURN_CKPT)" >/dev/null 2>&1; then \
		resume_arg="--resume-backbone $(RETURN_CKPT)"; \
		echo "starting portfolio-sharpe from $(RETURN_CKPT)"; \
	elif [ -f "$(RETURN_CKPT)" ]; then \
		echo "ignoring incompatible sharpe bootstrap checkpoint at $(RETURN_CKPT)"; \
	fi; \
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && PYTHONPATH=python python -m train.portfolio_train \
		--head sharpe --data-dir $(TRAIN_DATA) --synth-dir $(FDA_JOINT_OUT) --symbols-file $(PORTFOLIO_SYMBOLS) \
		--steps 30000 --cross-asset-warmup-steps 0 --cross-asset-warmup-batch-size $(WARMUP_BATCH) --batch-size $(BATCH_PORT) --seq-len 64 \
		--fda-mix-prob $(FDA_MIX) \
		--bptt-chunk $(BPTT_CHUNK_PORT) --pta-micro-batch-size $(PTA_MICRO_BS) --regime-pool-size $(REGIME_POOL) \
		--lr 3e-5 --log-every 10 --eval-every 250 --save-every 250 --checkpoint-dir $(CKPT_PORT) $(PORT_EXTRA) \
		$$resume_arg)
	@echo "→ tmux attach -t portfolio"

backtest:
	@set -euo pipefail; \
	head_mode="return"; \
	sharpe_arg=""; \
	if [ -f "$(CKPT_PORT)/sharpe_head_best.pt" ]; then \
		head_mode="blended"; \
		sharpe_arg="--checkpoint-sharpe $(CKPT_PORT)/sharpe_head_best.pt"; \
	fi; \
	$(PYTHON) -m backtesting.backtesting \
		--checkpoint $(CKPT_PORT)/return_head_best.pt --data-dir $(DATA_DIR) --head $$head_mode $$sharpe_arg
