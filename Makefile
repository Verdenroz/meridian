SHELL      := /bin/bash
PYTHON     := source .venv/bin/activate && PYTHONPATH=python python
DATA_DIR   := data/mmaps
SYMBOLS    := symbols.txt
CKPT_PRE   := checkpoints/pretrain
CKPT_PORT  := checkpoints/portfolio
BACKBONE   ?= $(CKPT_PRE)/backbone_pretrain.pt
RETURN_CKPT ?= $(CKPT_PORT)/return_head_best.pt

# systemd-inhibit runs inside tmux so the lock stays alive for the full run.
# Attach: tmux attach -t <session>   Detach: Ctrl+B D
INHIBIT    := systemd-inhibit --what=sleep:idle --who=meridian --why=gpu-training
TMUX_RUN    = tmux kill-session -t $1 2>/dev/null; \
              tmux new-session -d -s $1 "$(INHIBIT) bash -c '$2'"

.PHONY: collect stats pretrain pretrain-resume smoke-pretrain \
        portfolio-return portfolio-return-resume \
        portfolio-sharpe portfolio-sharpe-resume smoke-portfolio \
        build-py backtest backtest-smoke

# ── Data ──────────────────────────────────────────────────────────────────────

collect:
	$(PYTHON) -m market_data.pipeline collect \
		--symbols-file $(SYMBOLS) --interval 1d --start 2010-01-01 \
		--data-dir $(DATA_DIR) --workers 8

stats:
	$(PYTHON) -m market_data.pipeline stats --data-dir $(DATA_DIR)

# ── Pre-training ──────────────────────────────────────────────────────────────

pretrain:
	$(call TMUX_RUN,pretrain,source .venv/bin/activate && PYTHONPATH=python python -m train.pretrain \
		--data-dir $(DATA_DIR) --steps 200000 --batch-size 64 --seq-len 64 \
		--lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
		--save-every 5000 --log-every 10 --checkpoint-dir $(CKPT_PRE) --workers 4)
	@echo "→ tmux attach -t pretrain"

pretrain-resume:
	$(call TMUX_RUN,pretrain,source .venv/bin/activate && PYTHONPATH=python python -m train.pretrain \
		--data-dir $(DATA_DIR) --steps 200000 --batch-size 64 --seq-len 64 \
		--lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
		--save-every 5000 --log-every 10 --checkpoint-dir $(CKPT_PRE) --workers 4 \
		--resume-from $(CKPT_PRE)/backbone_pretrain.pt)
	@echo "→ tmux attach -t pretrain"

smoke-pretrain:
	$(PYTHON) -m train.pretrain --smoke --steps 5 --batch-size 2

# ── Portfolio training ────────────────────────────────────────────────────────

# Batch size for portfolio training.
# On 24GB GPUs (RTX 3090): use 4 with --gradient-checkpointing
# On A100/H100 (40-80GB):  use 16+ without --gradient-checkpointing
BATCH_PORT ?= 4
PORT_EXTRA ?= --gradient-checkpointing --compile --workers 2

portfolio-return:
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && PYTHONPATH=python python -m train.portfolio_train \
		--head return --data-dir $(DATA_DIR) --steps 50000 --batch-size $(BATCH_PORT) --seq-len 64 \
		--lr 3e-5 --checkpoint-dir $(CKPT_PORT) $(PORT_EXTRA) \
		--resume-backbone $(BACKBONE))
	@echo "→ tmux attach -t portfolio"

portfolio-return-resume:
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && PYTHONPATH=python python -m train.portfolio_train \
		--head return --data-dir $(DATA_DIR) --steps 50000 --batch-size $(BATCH_PORT) --seq-len 64 \
		--lr 3e-5 --checkpoint-dir $(CKPT_PORT) $(PORT_EXTRA) \
		--resume-backbone $(RETURN_CKPT))
	@echo "→ tmux attach -t portfolio"

portfolio-sharpe:
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && PYTHONPATH=python python -m train.portfolio_train \
		--head sharpe --data-dir $(DATA_DIR) --steps 30000 --batch-size $(BATCH_PORT) --seq-len 64 \
		--lr 3e-5 --checkpoint-dir $(CKPT_PORT) $(PORT_EXTRA) \
		--resume-backbone $(RETURN_CKPT))
	@echo "→ tmux attach -t portfolio"

portfolio-sharpe-resume:
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && PYTHONPATH=python python -m train.portfolio_train \
		--head sharpe --data-dir $(DATA_DIR) --steps 30000 --batch-size $(BATCH_PORT) --seq-len 64 \
		--lr 3e-5 --checkpoint-dir $(CKPT_PORT) $(PORT_EXTRA) \
		--resume-backbone $(CKPT_PORT)/sharpe_head_best.pt)
	@echo "→ tmux attach -t portfolio"

smoke-portfolio:
	$(PYTHON) -m train.portfolio_train --head return --smoke --steps 3

# ── Backtesting ──────────────────────────────────────────────────────────────

build-py:
	CONDA_PREFIX= maturin develop --release

backtest:
	$(call TMUX_RUN,backtest,source .venv/bin/activate && PYTHONPATH=python python -m backtesting.backtesting \
		--checkpoint $(CKPT_PORT)/return_head_best.pt --data-dir $(DATA_DIR) --head return)
	@echo "→ tmux attach -t backtest"

backtest-smoke:
	$(PYTHON) -m backtesting.backtesting --smoke
