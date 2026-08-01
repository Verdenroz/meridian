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
- **Live, new core:** Rust `src/` — `meridian-bus`, the live market-hours poller and
  break-glass monitor (§3.5/§4.5): quote + intraday-bar sweeps, calibrated price/volume
  severity rules (LLM-free triggers), SEC 8-K feed as context. Heartbeat journal in
  `logs/bus/`; alerts append to git-tracked, append-only `data/breakglass/alerts.jsonl` —
  advisory only until execution (step 10) exists. Trigger thresholds are calibrated to the
  §4.5 base rate (~2 break-glass episodes/yr on 1993–2026 SPY); don't nudge them casually.
- **Legacy — do not extend:** the prior v4 training stack is archived under `legacy/`
  (old `python/` packages, `c/`, FDA/TCP/pretraining pipelines, the training Makefile,
  colab notebooks, the v4 README, and the old 12k-line Rust training-corpus DataBus in
  `legacy/databus-v4/`). It is reference-only. Don't build on it, don't "fix" it, don't
  assume its Makefile targets are the way to run new work.

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

# Quality-spread race (pre-registered; FAILED 2026-08-01 — Baa−Aaa adds nothing
# beyond BAA10Y's level, 6f dilutes the spine; matched-vol placebo killed the
# (b)-route correctly. HY OAS itself is unusable: upstream truncates ICE BofA ~3y)
uv run python -m quality_race run

# VRP race (pre-registered; FAILED 2026-08-01 decisively — premium terciles and
# turnover bands don't move VIX sizing off 0.41 vs null 0.49; VIX branch closed)
uv run python -m vrp_race run

# Deep-history 8-K occurrence race (pre-registered; FAILED 2026-08-01 at high
# power — OOS 2004+, 15 names, 1242 active days: brake loses, Sharpe 0.633 vs
# 0.666, signal spearman 0.011. The original held race's monthly re-race stands)
uv run python -m deep_eightk fetch && uv run python -m deep_eightk run

# Tone study (pre-registered; PASSED 2026-08-01 — FinBERT negative-tone days beat
# a matched news-active control +6.5% p=0.008; market tone spearman 0.128 on SPY
# 5d residual vol. Content, not volume; short-horizon only, gone by 21d. Unlocks
# tone features via their own future races — criteria frozen, don't retune)
uv run python -m tone_study run

# Tone-brake race (pre-registered; FAILED 2026-08-01 — the market-tone burst
# brake loses to the null (0.786 vs 0.803) with a *worse* tail, despite the
# strongest signal diagnostic yet (spearman 0.144). Third brake failure in a
# row: brakes on the null are a dead deployment family; orthogonal channels
# only add value as regime features. Tone market-defense closed)
uv run python -m tone_race run

# Rate-vol race (pre-registered; PASSED 2026-08-01 route (a) — first outright
# Sharpe wins: 5f-rate 0.52 and 6f credit+ratevol 0.53 vs null 0.49 at a third
# of the drawdown; channels compound, no quality-race dilution. Bootstrap
# margins are noise → held, not shipped; 6f is the new leading candidate)
uv run python -m rate_race run

# Hedge-veto race (pre-registered; PASSED 2026-08-01 route (a) — TLT barred
# from the defensive sleeve 21 sessions after a rate-vol burst strictly
# improves the 5f-credit spine (0.509/12.9%dd vs 0.502/13.7%). Held. Value is
# conditional on the regime model lacking the rate channel — see compound race)
uv run python -m hedge_race run

# Stock-bond correlation race (pre-registered; FAILED 2026-08-01 — 60d
# corr(SPY, −ΔDGS10) adds nothing alone (placebo kill) and dilutes the 6f
# leader (7f 0.43 vs 0.53). Co-movement sign regime closed as a feature)
uv run python -m corr_race run

# Compound race (pre-registered; PASSED 2026-08-01 vs null but BELOW its 6f
# parent (0.521 vs 0.528) — veto double-counts once rate vol is in the HMM.
# 6f alone leads; don't stack the veto on rate-aware regime models)
uv run python -m compound_race run

# Regime-granularity race (pre-registered; PASSED 2026-08-01 route (a) — the
# largest margin yet: 3-state HMM on 6f, Sharpe 0.595 vs null 0.491 (2-state
# 0.528) at half the null's drawdown. Derived from the disjoint-episode
# diagnostic: stress comes in kinds. Bootstrap P(≤0)=0.15, strongest evidence
# of any candidate → held, new leading candidate, in rerace + shadow)
uv run python -m nstate_race run

# Four-state race (pre-registered; FAILED 2026-08-01 decisively — 0.438 vs
# null 0.491, maxdd 31%. Granularity curve peaks at 3 states (0.528/0.595/
# 0.438); n-scanning closed, learned regime challengers are the next axis)
uv run python -m nstate4_race run

# Deep filing-tone study (pre-registered; FAILED 2026-08-01 — 4,856 8-K texts
# FinBERT-scored on the RTX 3090 (~3 min), 4,871 filing days 1994-2026: only
# 1.3% of filing days read negative (boilerplate), rank spearman 0.051 < 0.10
# gate. Filing text closed on all three fronts (severity/occurrence/content);
# headline content remains the only passing text signal)
#   uv run python -m filing_tone fetch && uv run python -m filing_tone status
#   uv run --with torch --with transformers python -m filing_tone score
uv run python -m filing_tone run

# Topic-salience race (pre-registered; PASSED 2026-08-01 on the letter, the
# weakest held candidate — 7f 0.883 vs null 0.864 on the easy 2010-2024 span,
# salience alone FAILS, bootstrap noise. Held, low confidence; LOCAL-ONLY
# adjudication: needs data/embed_cache, so excluded from CI rerace)
uv run python -m salience_race run

# Champion consolidation race (pre-registered; PASSED 2026-08-01 route (a) —
# 7f × 3-state 0.900 > 6f-3s 0.885 > 7f-2s 0.883 > null 0.864 on the clipped
# span: the feature and state axes are additive. 6f 3-state stays the
# full-span leader; 7f 3-state is the salience-era variant, local-only)
uv run python -m champion_race run

# Episodic analog memory (roadmap step 8) — nearest historical analogs + briefing.
# Explanation only: analogs never touch allocation without a new pre-registered
# race. News analogs added 2026-08-01: per-day headline-embedding centroids;
# live dates embed archive titles via the local endpoint
uv run python -m episodic run                     # analogs for the latest close
uv run python -m episodic run --date 2020-03-16   # any panel date
uv run python -m episodic build-news              # rebuild news centroid store
uv run python -m episodic news --date 2020-03-16  # news-similar days

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

# Live poller + break-glass monitor (Rust; runs on the appliance during market hours;
# appliance service unit: deploy/meridian-bus.service, install steps in its comments.
# Optional MERIDIAN_ALERT_WEBHOOK env var POSTs alert rows, e.g. to an ntfy topic)
cargo run --release --bin meridian-bus -- live     # persistent poll loop
cargo run --bin meridian-bus -- once               # one cycle, print snapshot
cargo run --bin meridian-bus -- status             # alert + journal summary
cargo test                                         # severity/profile unit tests
```

Testing philosophy: correctness is validated primarily through the walk-forward harness,
backtests against the null baseline, and smoke runs — not broad unit coverage. Don't assume
an 80%-coverage gate; do test point-in-time logic and dedupe/storage invariants, where bugs
silently poison data.

## Hypothesis framework (pre-registered races & studies)

Every hypothesis gets a small dedicated package under `python/` whose module docstring
*is* the frozen pre-registration, racing challengers against the untouched null in the
one walk-forward engine. Shared mechanics live in `python/harness/`; a race module owns
only the hypothesis — features, challengers, frozen constants:

- `harness/racing.py` — `run_contender` (walk-forward + after-cost metrics, optional
  `oos_start`), `race_contenders` (labeled lineup, same engine/span), `metrics_table`,
  `plain` (JSON-safe verdict serialization).
- `harness/gates.py` — `amended_bar` (the §1.3 promotion bar: outright Sharpe win, or
  the held (b)-route: non-inferiority + tail bound + matched-vol placebo) and the
  `ScaledVolTarget` placebo. Gate margins are required arguments — criteria live in each
  race's frozen `GATE` dict, never as defaults in the shared layer.
- `harness/signals.py` — `causal_zscores`, `brake_series`, `BrakeVolTarget`, the
  inherited-frozen burst/brake mechanics.
- `harness/stats.py` — rank stats (`spearman`, `mann_whitney_u`) and the seeded
  `sharpe_diff_bootstrap` (block length / resamples / seed are per-study registrations,
  passed explicitly).

New race checklist: (1) write the module docstring as the pre-registration — motivation,
challengers, gate — *before* computing any result; (2) freeze every constant in a `GATE`
dict; (3) contenders always include the untouched null; run `race_contenders`, judge
with `amended_bar(runs[c], null, panel, first, GATE)`; (4) record the verdict in the
spec and CLAUDE.md and never retune post hoc. Completed race modules are historical
record: framework changes must never alter a frozen race's output (verdict equality was
checked when the shared layer was extracted, 2026-08-01).

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
