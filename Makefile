SHELL      := /bin/bash
PYTHON     := source .venv/bin/activate && python
DATA_DIR   := data/mmaps
SYMBOLS    := symbols.txt
CKPT_PRE   := checkpoints/pretrain
CKPT_PORT  := checkpoints/portfolio

# systemd-inhibit runs inside tmux so the lock stays alive for the full run.
# Attach: tmux attach -t <session>   Detach: Ctrl+B D
INHIBIT    := systemd-inhibit --what=sleep:idle --who=meridian --why=gpu-training
TMUX_RUN    = tmux kill-session -t $1 2>/dev/null; \
              tmux new-session -d -s $1 "$(INHIBIT) bash -c '$2'"

.PHONY: collect stats pretrain pretrain-resume smoke-pretrain \
        portfolio-return portfolio-sharpe smoke-portfolio

# ── Data ──────────────────────────────────────────────────────────────────────

collect:
	$(PYTHON) -m market_data.pipeline collect \
		--symbols-file $(SYMBOLS) --interval 1d --start 2010-01-01 \
		--data-dir $(DATA_DIR) --workers 8

stats:
	$(PYTHON) -m market_data.pipeline stats --data-dir $(DATA_DIR)

# ── Pre-training ──────────────────────────────────────────────────────────────

pretrain:
	$(call TMUX_RUN,pretrain,source .venv/bin/activate && python -m train.pretrain \
		--data-dir $(DATA_DIR) --steps 200000 --batch-size 64 --seq-len 64 \
		--lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
		--save-every 5000 --log-every 10 --checkpoint-dir $(CKPT_PRE) --workers 4)
	@echo "→ tmux attach -t pretrain"

pretrain-resume:
	$(call TMUX_RUN,pretrain,source .venv/bin/activate && python -m train.pretrain \
		--data-dir $(DATA_DIR) --steps 200000 --batch-size 64 --seq-len 64 \
		--lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
		--save-every 5000 --log-every 10 --checkpoint-dir $(CKPT_PRE) --workers 4 \
		--resume-from $(CKPT_PRE)/backbone_pretrain.pt)
	@echo "→ tmux attach -t pretrain"

smoke-pretrain:
	$(PYTHON) -m train.pretrain --smoke --steps 5 --batch-size 2

# ── Portfolio training ────────────────────────────────────────────────────────

portfolio-return:
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && python -m train.portfolio_train \
		--head return --data-dir $(DATA_DIR) --steps 50000 --batch-size 16 --seq-len 64 \
		--lr 3e-5 --checkpoint-dir $(CKPT_PORT) \
		--resume-backbone $(CKPT_PRE)/backbone_pretrain.pt)
	@echo "→ tmux attach -t portfolio"

portfolio-sharpe:
	$(call TMUX_RUN,portfolio,source .venv/bin/activate && python -m train.portfolio_train \
		--head sharpe --data-dir $(DATA_DIR) --steps 30000 --batch-size 16 --seq-len 64 \
		--lr 3e-5 --checkpoint-dir $(CKPT_PORT) \
		--resume-backbone $(CKPT_PRE)/backbone_pretrain.pt)
	@echo "→ tmux attach -t portfolio"

smoke-portfolio:
	$(PYTHON) -m train.portfolio_train --head return --smoke --steps 3
