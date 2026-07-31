# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Meridian is

Meridian is a **regime-adaptive risk engine that manages real money**. It forecasts risk —
volatility, correlation, market regime — and allocates an ETF core plus a 20–30 stock
risk-weighted sleeve accordingly: near-full equity exposure in benign regimes, cash-heavy
defense in stress. It does **not** forecast returns (measured IC ≈ 0; this is a settled
decision, not an open question). Decisions are daily, with a rare event-driven intraday
break-glass path. The bar it must clear: **beat SPY + a simple vol-target rule, after costs,
out-of-sample** — that null baseline is implemented in the same harness and raced honestly.
Bar amended 2026-07-31 (§1.3): outright Sharpe win, or Sharpe non-inferiority (−0.05) with
max drawdown ≤⅔ of the null's and a matched-vol placebo control — prospective only; held
candidates pass on post-amendment evidence, never on the backtest that motivated the change.

`meridian-spec.md` is the canonical spec; its **Charter (§1) is authoritative** over
everything, including this file. Read it before design work. Do not cite spec section
numbers in code comments — the spec is internal; comments explain the *why* briefly.

## Non-negotiable invariants

- **Risk, not returns.** No component predicts returns. Reintroducing return prediction
  requires the challenger gate, explicitly.
- **Challengers, never incumbents.** Any learned component ships only after beating its
  classical counterpart in the walk-forward harness. Classical spine runs production.
- **Point-in-time honesty.** Text/event features key on `fetched_at` (when we could have
  known), never publisher timestamps. The news archive is append-only; never mutate or
  hand-edit `archive/feeds/`. Enrichment (LLM scores etc.) is stored separately, versioned.
- **Guardrails outside the model.** Position/turnover/exposure limits are enforced in code
  the model cannot touch.
- **Excluded instruments:** crypto, margin/leverage, short selling (inverse ETFs allowed).
- **finance-query is an unmodified upstream API.** All development is self-contained in this
  repo; never patch finance-query (or any external service) as part of Meridian work, even
  though it is our own project.
- **Don't rebuild what the finance-query crate has.** Its Rust crate ships a signal-based
  backtester (indicator strategies, stops, grid/Bayesian walk-forward parameter search,
  Monte Carlo) and risk math (VaR, beta, ratios; `/v2/risk` is the API surface). If Meridian
  ever needs those, consume the crate (Rust side) or the API — don't port them to Python.
  Meridian's own harness exists because it is a different abstraction the crate does not
  provide: daily *target-weight* allocation from regime posteriors, T-bill cash accrual,
  charter guardrails, and one-engine-for-everything promotion gates.

## Repository state

The repo has been rebuilt around the charter. Current inventory:

- **Live, new core:** `python/feeds_archive/` — append-only news/feeds archiver
  (`fetch` / `loop` / `build-db` / `status`), stdlib-only. Runs via
  `.github/workflows/archive-feeds.yml` (10-min cron) committing JSONL to `archive/feeds/`.
  The SQLite index (`data/feeds_archive.db`) is derived and rebuildable; JSONL is truth.
- **Live, new core:** `python/market_data/` — historical loaders (charts/dividends/splits,
  FRED, treasury curve) with a loud-failure adjClose cross-check; snapshots under
  `data/market/` (derived, rebuildable, `fetched_at`-stamped).
- **Live, new core:** `python/harness/` — walk-forward backtest engine (after-cost returns,
  long-only/unlevered guardrails in the engine, lookahead-proof `HistoryView`) and the null
  baselines (SPY buy-hold, SPY 15% vol-target). Every component ships only by beating the
  null in this engine.
- **Being repurposed:** Rust `src/` (DataBus) becomes the live market/event poller for
  daily decisions and break-glass monitoring — no longer a training-corpus builder.
- **Legacy — do not extend:** the prior v4 training stack is archived under `legacy/`
  (old `python/` packages, `c/`, FDA/TCP/pretraining pipelines, the training Makefile,
  colab notebooks, the v4 README). It is reference-only. Don't build on it, don't "fix"
  it, don't assume its Makefile targets are the way to run new work.

New work goes in fresh packages under `python/` (small focused modules), driven by direct
module invocation — `PYTHONPATH=python python -m <package> ...` from the repo root.

## Common commands

Python tooling is **uv** (`uv sync` creates `.venv` from `pyproject.toml`/`uv.lock`;
dev tools live in the `dev` dependency group). The repo is dual-language: Python under
`python/`, Rust under `src/` (the DataBus live poller, built with cargo).

```bash
uv sync                           # create/update .venv (includes dev group)

uv run ruff check python          # lint
uv run mypy python                # types
uv run pytest python/tests       # tests (deliberately minimal — see below)

# News/feeds archive
uv run python -m feeds_archive fetch      # one-shot poll (CI uses PYTHONPATH=python instead)
uv run python -m feeds_archive build-db   # compile JSONL → SQLite index
uv run python -m feeds_archive status     # archive freshness

# Historical market/macro data (ETF core + VIX + treasury curve)
uv run python -m market_data fetch        # fetch, cross-check, store
uv run python -m market_data verify       # re-run adjClose cross-check
uv run python -m market_data status       # what's in data/market/

# Null baseline race (the bar everything must clear)
uv run python -m harness null             # full-sample + walk-forward OOS

# Classical spine
uv run python -m spine race               # walk-forward race: spine vs nulls
uv run python -m spine vol-eval           # EWMA forecast quality vs 0.67 bar

# Event study (§5 gate; criteria are pre-registered — don't retune post hoc)
uv run python -m event_study run          # passed 2026-07-31, scoped to 8-Ks

# LLM event pipeline (8-K scoring; local qwen via chimeric)
uv run python -m event_pipeline score --limit 5   # score newest unscored 8-Ks
uv run python -m event_pipeline status            # counts by scorer version

# Score study (pre-registered gate; FAILED 2026-07-31 — severity ranks no vol uplift,
# so scores are break-glass inputs only; don't retune the gate post hoc)
uv run python -m score_study run

# Occurrence-flag race (pre-registered; PASSED 2026-07-31 but fragile — held, not
# shipped; re-run as OOS accrues, never retune the frozen parameters)
uv run python -m eightk_race run

# Vol-forecaster race (pre-registered; FAILED 2026-07-31 — VIX beats EWMA on forecast
# correlation but loses the allocator race; forecast quality never gates shipping)
uv run python -m vol_race run

# Credit-spread race (pre-registered; nominal PASS 2026-07-31 but margins are noise —
# held; BAA10Y is the first regime feature with OOS value: spine 4f 0.44 → 5f 0.50
# at a third of the null's drawdown. Re-run as OOS accrues; never retune)
uv run python -m credit_race run

# Daily shadow decisions (paper-trading record; runs post-close via CI cron).
# data/shadow/decisions.jsonl is append-only, git-tracked, point-in-time — never
# regenerate or hand-edit rows; retire+rename contenders instead of editing them
uv run python -m shadow run               # fetch latest, log today's decisions
uv run python -m shadow status            # log summary

# Monthly re-race of the held gates (CI cron, 1st of month; verdicts append to
# git-tracked data/races/reraces.jsonl — one row per (month, race), idempotent)
uv run python -m rerace run               # fetch, re-run eightk_race + credit_race, log
uv run python -m rerace status            # verdict history

# Behavior-match gate (pre-registered 2026-07-31, frozen — never edit criteria.py;
# the evaluator refuses to run before 2026-11-02, and PASS satisfies §1.3(b) only,
# it is not a performance gate)
uv run python -m behavior_match criteria  # print the frozen pre-registration
uv run python -m behavior_match run       # evaluate the closed shadow window
```

Testing philosophy: correctness is validated primarily through the walk-forward harness,
backtests against the null baseline, and smoke runs — not broad unit coverage. Don't assume
an 80%-coverage gate; do test point-in-time logic and dedupe/storage invariants, where bugs
silently poison data.

## Data

- All market/text data comes from `https://finance-query.com` (v2). Verified availability
  (depths, endpoints, the one gap — historical news) is tabled in the spec's Appendix A.
- Total-return series come from the chart endpoint's `adjClose` (populated upstream since
  2026-07-30), cross-checked against a local dividends+splits reconstruction in the loader.
- `data/`, `checkpoints/`, `reports/`, `logs/` are gitignored (derived, regenerated).
  `archive/` is **git-tracked** — it is the permanent news record.
- Survivorship bias exists in Yahoo-sourced stock history; every backtest reports it as a
  known limitation.

## LLM access (for the event pipeline, §5 — gated on the event study)

- Local-first: OpenAI-compatible endpoint at `http://127.0.0.1:12434/v1` (llama-swap on the
  appliance; verified 2026-07-31: `qwen3.6:35b-a3b` chat + `harrier-oss:0.6b` /
  `nomic-embed-text` embeddings). Never assume it's running — check `/v1/models` first.
- Client abstraction: **chimeric** (Harvey's published multi-provider LLM library, on PyPI) —
  local model is the default, hosted models are config swaps.
- ML/NN components remain welcome as **challengers** through the walk-forward gate (§1.7);
  the local eGPU is available for training/testing runs.

## Build order

The spec's §10 roadmap is the sequence. Headline rule: **the null baseline and walk-forward
harness are built before any model component is evaluated**, and the event study gates all
LLM/text pipeline investment.
