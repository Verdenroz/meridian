---
name: train-gate
description: Validate-before-spend workflow for Meridian training quality. Use when iterating on pretrain or portfolio (Phase 2) training quality, before funding any long GPU run, when a training metric is flat/saturated/suspicious, or when the user asks to gate, canary, or verify a checkpoint or fix.
---

# Train-Gate: validate before you spend

The core discipline: **no full training run gets funded until a cheap experiment
with a pre-registered decision rule says the ingredients work.** Every fix gets
verified in isolation before it ships. Every claim gets a control.

## 1. The three-arm comparison (Gate 0)

Before any Phase-2 GPU spend, run all three arms on identical data/protocol:

| arm | command core | purpose |
|---|---|---|
| naive floor | `PYTHONPATH=python python -m tools.equal_weight_floor` | absolute floor; costs ~1 min, no model |
| random control | canary WITHOUT `--resume-backbone` | what "no pretraining" earns |
| candidate | canary WITH `--resume-backbone <ckpt>` | the thing being judged |

Canary protocol (frozen-backbone linear probe, real data — see issue #4):

```bash
PYTHONPATH=python python -m train.portfolio_train \
  --head return --data-dir data/mmaps \
  --resume-backbone <CHECKPOINT> \
  --steps 300 --eval-every 50 --log-every 10 \
  --cross-asset-warmup-steps 0 --freeze-backbone-steps 999999 \
  --fda-mix-prob 0 --pta-prob 0 --pta-vol-scale-prob 0 --pta-sign-flip-prob 0 \
  --batch-size 4 --disable-regime-balance --disable-phase2-gate \
  --lr 3e-4 \
  --checkpoint-dir <SCRATCH_DIR>   # NEVER checkpoints/portfolio (auto-resume)
```

Decision rule (register the numbers BEFORE running):
- candidate ≪ random control on val dsr (lower = better) AND sharpe above it → fund the full run
- candidate ≈ control → the upstream stage adds nothing; fix it first
- both ≈ floor → head/loss problem, debug before scaling

Standing baselines (2026-07-22, data/mmaps val, 47 batches): floor dsr **−0.0429**
/ sharpe 0.651; random control dsr **−0.0483** / sharpe 0.654 (converges by step
50, flat after). Recompute the floor if the val split or loss config changes.

Gotchas baked into that command:
- `--lr 3e-4` not 3e-5: the 500-step LR warmup is hardcoded in
  `PortfolioTrainConfig` (not CLI-exposed); at 3e-5 a 300-step probe never
  reaches meaningful LR. Apply identical LR to every arm.
- batch-size < 4 crashes regime-balanced sampling (needs ≥ n_regimes) unless
  `--disable-regime-balance`.
- Checkpoint dir must be outside `checkpoints/` — the Makefile targets
  auto-resume from anything compatible they find there.

Timing (RTX 3090 eGPU): ~2.3 s/step at batch 4, ~1 min per full-val eval →
whole three-arm comparison ≈ 30–40 min. CPU is not viable (~100 s/step,
~50 min/eval).

## 2. In-run gates for the full Phase-2 run

- **Gate 1** — end of cross-asset warm-up: `corr_frob` in the log must trend
  down; the warm-up trains the whole backbone stack at 0.1× LR.
- **Gate 2** — head-training steps 500–1,000 (through the freeze→unfreeze
  boundary): val DSR above equal-weight floor, weights not collapsed to
  uniform, daily turnover sane (spec target < 0.10), no loss spike at
  unfreeze. Frozen-phase numbers are directly comparable to Gate 0.
- **Gate 3** — ~10% of total steps: best checkpoint must have improved since
  the frozen phase; if not, fine-tuning adds nothing — stop the run.
- Sharpe stage re-enters a freeze/unfreeze cycle from the return checkpoint:
  repeat Gate 2 there.

## 3. When a training metric is flat or saturated: elimination probes

Never tune blind. Form the failure hypotheses, then run the cheapest
discriminating probe for each, in this order (each clears or convicts in
minutes):

1. **Data/labels**: iterate real batches; print label mix per batch and a
   physical-sanity correlation (e.g. realized vol by regime label).
2. **Loss in isolation**: optimize FREE embedding tensors against the loss
   (no model). Include the pathological init (cone-collapsed) — a loss that
   minimizes from spread inits may still matter from collapsed ones.
3. **Numerics**: A/B the exact training step with autocast on vs off.
4. **Gradient competition**: train the suspect term ALONE at weight 1.0.
5. **Representation capacity**: cache frozen embeddings; train only the head
   on them, plus a linear probe for decodability. If the head can't win on
   cached embeddings, the objective is mismatched to the representation —
   change the objective, don't tune the optimizer.

Keep an elimination table in the conversation/issue; pre-register what each
outcome would mean before results arrive (prevents motivated reading).

Reusable probes in `python/tools/`:
- `equal_weight_floor.py` — Phase-2 naive floor
- `asset_shortcut_probe.py` — same-asset NN rate in embedding space (asset-
  identity shortcut strength; chance ≈ 0.001, random-init ≈ 0.91)
- `eval_pretrain.py` — trained-vs-random-init pretrain eval + naive vol
  baselines + SupCon gradient probe

## 4. Fix → verify → commit → document

1. Implement the minimal fix.
2. Verify with a short ISOLATED run (300–1,500 steps on the eGPU) where the
   fix's effect is the only thing that can move the metric. Compare against a
   matched control — same weights/config, only the fix toggled. A control
   with two variables changed attributes nothing.
3. Commit only after verification, with the evidence and numbers in the
   commit message. Known-caveat lines belong there too (e.g. single seed).
4. Post the numbers to the relevant GitHub issue (tables, not prose), with
   the go/no-go bar for the next step stated explicitly.
5. Colab picks up fixes ONLY after push + clearing its cached `/content`
   clone. The old checkpoint on HF and Drive are auto-resume traps after any
   objective change — archive them (notebook `FRESH_START = True`).

## 5. Session-learned failure signatures (check before debugging fresh)

- Flat loss pinned at exactly log(2B−1) → contrastive term has nothing to
  grab (class overlap), not an optimizer bug.
- contrastive_acc saturating early → shortcut (input overlap at offset < 64;
  asset re-identification otherwise — see `asset_shortcut_probe.py`).
- Adversarial CE exploding (6.9 → 90+) → feature-norm escape; head input must
  be normalized (fixed in 7344d4e).
- `BackendCompilerFailed ... invalid, but is output` mid-run under
  `--compile` → runt batch changed shapes; training loaders need
  `drop_last=True` (fixed in 5d3e77f for Phase 2; pretrain regime-balanced
  sampler is epoch-less and immune, but its val/finite loaders are not).
- Val regime CE high while train falls → val split skew (87% bull, 0
  crisis), not a bug; judge geometry by silhouette.
- OOM at pretrain batch 64 on 24 GB → use batch 32 + `--gradient-checkpointing`
  + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
