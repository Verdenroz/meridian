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

## Repository state

The repo has been rebuilt around the charter. Current inventory:

- **Live, new core:** `python/feeds_archive/` — append-only news/feeds archiver
  (`fetch` / `loop` / `build-db` / `status`), stdlib-only. Runs via
  `.github/workflows/archive-feeds.yml` (10-min cron) committing JSONL to `archive/feeds/`.
  The SQLite index (`data/feeds_archive.db`) is derived and rebuildable; JSONL is truth.
- **Being repurposed:** Rust `src/` (DataBus) becomes the live market/event poller for
  daily decisions and break-glass monitoring — no longer a training-corpus builder.
- **Legacy — do not extend:** the prior v4 training stack is archived under `legacy/`
  (old `python/` packages, `c/`, FDA/TCP/pretraining pipelines, the training Makefile,
  colab notebooks, the v4 README). It is reference-only. Don't build on it, don't "fix"
  it, don't assume its Makefile targets are the way to run new work.

New work goes in fresh packages under `python/` (small focused modules), driven by direct
module invocation — `PYTHONPATH=python python -m <package> ...` from the repo root.

## Common commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check python                 # lint
mypy python                       # types
pytest python/tests               # tests (deliberately minimal — see below)

# News/feeds archive (the one live new component)
PYTHONPATH=python python -m feeds_archive fetch       # one-shot poll (what CI runs)
PYTHONPATH=python python -m feeds_archive build-db    # compile JSONL → SQLite index
PYTHONPATH=python python -m feeds_archive status      # archive freshness
```

Testing philosophy: correctness is validated primarily through the walk-forward harness,
backtests against the null baseline, and smoke runs — not broad unit coverage. Don't assume
an 80%-coverage gate; do test point-in-time logic and dedupe/storage invariants, where bugs
silently poison data.

## Data

- All market/text data comes from `https://finance-query.com` (v2). Verified availability
  (depths, endpoints, the one gap — historical news) is tabled in the spec's Appendix A.
- Total-return series are constructed locally from chart + dividends + splits endpoints
  (`adjClose` is not populated upstream).
- `data/`, `checkpoints/`, `reports/`, `logs/` are gitignored (derived, regenerated).
  `archive/` is **git-tracked** — it is the permanent news record.
- Survivorship bias exists in Yahoo-sourced stock history; every backtest reports it as a
  known limitation.

## Build order

The spec's §10 roadmap is the sequence. Headline rule: **the null baseline and walk-forward
harness are built before any model component is evaluated**, and the event study gates all
LLM/text pipeline investment.
