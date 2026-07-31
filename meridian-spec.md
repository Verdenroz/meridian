# Meridian — System Specification

**Status:** ratified 2026-07-30. This document is the canonical spec. Section 1 (the Charter)
is authoritative; when any later section, code comment, or plan conflicts with it, the Charter wins.

---

## 1. Charter

### 1.1 Purpose

Meridian exists to manage Harvey's real money. Not a research vehicle, not a résumé piece —
every design decision resolves toward one question: *would I trust this system with my own
capital?* Capital is earned in stages: **paper trading → small taxable account → meaningful
taxable capital**, each promotion contingent on demonstrated performance.

### 1.2 Objective: risk, not returns

Meridian forecasts **risk** — volatility, correlation, and market regime — and allocates
accordingly. It does **not** forecast returns. This is evidence-based: cross-sectional return
predictability measured at IC ≈ 0 across every representation tried (raw features, trained
embeddings, random embeddings, momentum), while volatility forecasting reached 0.67 correlation
against a 0.60 classical baseline. Risk skill alone pays: min-variance and vol-targeted
portfolios beat 1/N with no return signal. No component may quietly reintroduce return
prediction; revisiting that thesis requires challenger-gate proof (§1.7).

### 1.3 The bar

**The null hypothesis is SPY plus a simple volatility-target rule** — implemented honestly,
first, in the same harness. Meridian must beat it on risk-adjusted return, after costs,
out-of-sample, walk-forward. Below that bar it deserves zero dollars, and that result would
be accepted.

### 1.4 Mandate: regime-dependent exposure

No fixed volatility target. **Benign regimes:** up to fully invested equity with risk-weighted
tilts. **Stress regimes:** cash-heavy defense — primarily T-bill ETFs, with GLD/TLT included
only when correlation-verified safe (2022 proved "bonds = safety" is conditional). Inverse
ETFs are permitted as hedges; the portfolio is never net short. Regime-transition sensitivity
is tuned **cost-optimally** in walk-forward backtest — no deliberate bias toward jumpy or
sluggish.

### 1.5 Universe and constraints

- **ETF core:** liquid ETFs spanning equities, treasuries, gold, T-bills, plus a small
  inverse-ETF hedge sleeve — expresses the regime allocation.
- **Stock sleeve:** 20–30 liquid large-caps inside the equity bucket, **risk-weighted only**
  (volatility, correlation, event risk). Meridian does not pick winners.
- **Excluded:** crypto, margin/leverage, short selling.

### 1.6 Decision engine

**Classical spine:** EWMA/GARCH-class volatility, shrinkage-estimated correlations, an
HMM-class regime detector. **Events are first-class:** news flow and SEC filings are
structural inputs to the risk forecast — a deteriorating filing raises that name's forecast
risk, a market-wide news shock raises system-level risk, and the allocator responds through
normal sizing math. Text is scored by an **LLM in the loop** — local-first on the appliance,
behind an OpenAI-compatible endpoint abstraction so any hosted model slots in by
configuration.

**Cadence:** daily decisions after the close, plus a **break-glass path** — severe events may
trigger intraday de-risking, expected a few times a year. If it fires weekly, the threshold
is wrong.

### 1.7 ML policy: challengers, never incumbents

A learned component ships only when it beats its classical counterpart out-of-sample in the
walk-forward harness. Until then the classical component runs production. The most promising
challenger territory is the event/text data — and the claim that text improves risk forecasts
is testable and **must be tested early** via an event study before heavy pipeline investment.

### 1.8 Self-improvement

Two first-class mechanisms:

1. **Gated auto-retrain** — models refit on fresh data on a schedule; a refit is promoted
   only if it beats the incumbent in walk-forward validation. Improvement is automatic;
   regression is structurally impossible to promote silently.
2. **Episodic/analog memory** — live market states matched against decades of history to
   condition risk estimates and explain decisions ("today resembles Feb 2018").

### 1.9 Data

Training corpus: decades of daily OHLCV with total-return adjustment; macro series; EDGAR
filings and earnings transcripts for historical text; a self-collected, append-only news
archive from day one. The live feed is real-time prices and events for daily decisions and
break-glass monitoring. Details in §3.

### 1.10 Infrastructure

The **Framework Desktop (Ryzen AI Max+ 395, 128 GB unified)** is the always-on appliance:
ingestion, event monitoring, local LLM, inference, dashboard, execution. Hardware is
otherwise a **non-requirement** — correctness and quality are the only design constraints.
Data collection that must never gap (the news archive) runs in CI, not on the appliance.

### 1.11 Execution and guardrails

**Auto with kill switch:** broker API executes within hard guardrails — position, turnover,
and exposure limits enforced outside the model's control. Every action notifies; a kill
switch halts instantly. Guardrails start maximally conservative and loosen on the same
schedule capital grows. Component failures degrade safely: rule-based volume/volatility
triggers work without the LLM; no failure produces unbounded action.

### 1.12 Trust ladder

1. **Walk-forward backtest** — no leakage, realistic costs, the null run in the same harness. Must win.
2. **≥3 months live paper trading** — realized behavior must match backtest expectations
   (exposure profile, turnover, regime responses), not merely profit.
3. **Small taxable capital** — tight guardrails, trivial sizing.
4. **Meaningful capital** — only while it keeps earning it. Any stage can demote.

### 1.13 Implementation stance

Fresh core inside this repository; prior model/training code is archived, not extended.
Development is **self-contained in this repo** — external services (finance-query included)
are consumed as APIs, never modified as part of Meridian work.

---

## 2. System overview

```
finance-query.com/v2 (unmodified upstream API)
        │
        ├── historical loaders ──► total-return panel, macro panel, text corpus
        │                                   │
        ├── feeds archive (CI cron) ──► archive/feeds/*.jsonl ──► SQLite index
        │                                   │
        └── live poller (appliance) ─┐      ▼
                                     │   RISK ENGINE
                                     │   ├─ volatility forecasts (per asset)
                                     │   ├─ correlation estimates (shrunk)
                                     │   ├─ regime state (posterior)
                                     │   ├─ event risk scores (LLM, versioned)
                                     │   └─ episodic analogs (k-NN vs history)
                                     │      │
                                     │      ▼
                                     │   ALLOCATOR ──► target weights
                                     │      │
                                     ▼      ▼
                                  BREAK-GLASS   EXECUTION (broker API,
                                  monitor       hard guardrails, kill switch)
```

All components are independently testable against the walk-forward harness; every learned
or tuned piece has a classical incumbent it must displace (§1.7).

---

## 3. Data layer

All market/text data is consumed from `https://finance-query.com` (v2 API). Availability of
every dependency below was verified live on 2026-07-30 (Appendix A).

### 3.1 Historical market data

- Daily OHLCV via `/v2/chart/{symbol}?range=max&interval=1d` (batch: `/v2/charts`).
  Depth: to instrument inception (SPY 1993, KO 1985, TLT 2002, GLD 2004, BIL 2007, SH 2006).
- **Total-return series:** the chart endpoint populates `adjClose` (dividend- and
  split-adjusted; verified 2026-07-31 — zero nulls to inception, ratio steps on ex-div
  dates, converges to close on the latest bar). Upstream `adjClose` is the canonical input
  for all return, volatility, and backtest math; raw close is retained for price-level
  features. The data loader cross-checks `adjClose` against a local reconstruction from
  `/v2/dividends/{symbol}` and `/v2/splits/{symbol}` and fails loudly on divergence — this
  field was null until 2026-07-30, so upstream regression is a live risk, and silent
  adjustment errors would poison every downstream estimate.
- Intraday bars (1m–1h) are available for live monitoring; they are not training data.

### 3.2 Macro data

- FRED series via `/v2/fred/series/{id}` (e.g. VIXCLS from 1990). Treasury curve via
  `/v2/fred/treasury-yields?year=`. Candidate default set: VIX level/term proxy, yield-curve
  slopes, credit-spread series. Feature selection is decided by the regime model's
  walk-forward performance, not by fiat.

### 3.3 Historical text corpus

- **SEC filings:** `/v2/edgar/search` (full-text, date-filtered; verified to 2008),
  `/v2/edgar/submissions/{symbol}` (per-company history), `/v2/edgar/facts/{symbol}` (XBRL).
- **Earnings transcripts:** `/v2/transcripts/{symbol}/all` (~15 years per large-cap).
- **Pre-archive news:** the FNSPID dataset already on disk covers the historical news era.
  finance-query has no historical news API; that gap is closed going forward by §3.4.

### 3.4 News/feeds archive (implemented)

Self-contained package `python/feeds_archive/` polling `/v2/feeds` (25 finance/macro source
slugs, including `sec-filings`) and `/v2/news` (sentiment-labeled). Design rules:

- **Append-only.** Git-tracked JSONL day files under `archive/feeds/`; rows are never
  mutated. A headline revision is a new row (same URL, new content hash).
- **Point-in-time honesty.** Every row records `fetched_at` — when Meridian could have known
  the item. All event-model training and backtesting keys on `fetched_at`, never on
  publisher timestamps.
- **Dedupe** on `(source, canonical URL, content hash)`; tracking params stripped; sentiment
  churn excluded from the hash so re-scored items do not duplicate.
- **Heartbeats.** Every poll appends a heartbeat row (timestamp, per-source counts, errors)
  so coverage gaps are *known* — a backtest can mask blind windows instead of reading
  silence as "no news".
- **Runner:** GitHub Actions cron (`.github/workflows/archive-feeds.yml`, every 10 minutes,
  concurrency-guarded) — collection depends on no personal machine. The appliance may
  additionally run `loop` mode; the union of both covers either side's gaps.
- **SQLite index** (`data/feeds_archive.db`, gitignored) built idempotently from JSONL via
  `build-db`; it is a derived artifact, rebuildable at any time.
- Enrichment (LLM scores, entity tags) is stored separately, keyed by item id + scorer
  version — never written into the raw archive.

### 3.5 Live ingestion

The Rust DataBus (`src/`) is repurposed as the live feed: quotes, intraday bars, and feed
polling during market hours for the daily decision inputs and break-glass monitoring. Its
former role as training-corpus builder is retired.

---

## 4. Risk engine (classical spine)

Design-level commitments; exact estimator variants are chosen by walk-forward comparison,
not decided here.

- **4.1 Volatility.** Per-asset daily forecasts from the EWMA/GARCH family (RiskMetrics-style
  EWMA as the floor incumbent; GARCH(1,1)-class and range-based estimators as candidates).
  Forecast target: forward realized volatility. Measured baseline to beat: 0.67 correlation
  achieved in probing (vs 0.60 naive).
- **4.2 Correlation.** Shrinkage-estimated covariance (Ledoit–Wolf-class) on total returns;
  correlation-regime awareness (rolling safe-asset correlation checks) feeds the defense
  logic in §4.4.
- **4.3 Regime detector.** HMM-class model over market-level features (realized/implied
  volatility, curve slopes, credit proxies, drawdown state, news-flow intensity from §3.4).
  Output is a regime posterior, not a hard label. Transition sensitivity is a tuned
  parameter, optimized for after-cost walk-forward performance (§1.4).
- **4.4 Allocator.** Maps (regime posterior, vol forecasts, correlations, event risk) →
  target weights. Benign: near-full equity, risk-weighted within sleeves. Stress: cash-heavy
  (BIL-class), GLD/TLT only if currently behaving as safe assets, optional inverse-ETF
  hedge sleeve. Turnover penalty in the objective; trades below a drift threshold are
  skipped.
- **4.5 Break-glass.** A severity score from live monitoring (vol/volume spikes, gap moves,
  extreme event scores). Above threshold → intraday de-risk toward the stress allocation.
  Expected frequency: a few per year; the threshold is tuned to that base rate. Fully
  functional without the LLM (price/volume rules alone).

---

## 5. Event and text risk pipeline

- **Scoring:** an LLM reads filings/news items and emits structured risk assessments
  (severity, scope: asset vs market, direction of risk). Served via an OpenAI-compatible
  endpoint; local model on the appliance is the default, hosted models are config swaps.
  Verified 2026-07-31: llama-swap at `http://127.0.0.1:12434/v1` serving `qwen3.6:35b-a3b`
  (chat) plus embedding models. Client abstraction is **chimeric** (published
  multi-provider library) so provider switching stays configuration, not code.
  Scores are versioned by (model, prompt) and stored beside — never inside — the raw archive.
- **Integration:** scores enter the risk engine as features (per-asset risk elevation,
  market news-flow intensity), influencing positions only through §4's sizing math.
- **Gate:** before any pipeline investment beyond the archive, an **event study** must show
  that filing/news signals precede elevated realized volatility on this universe. If the
  event study fails, the pipeline stays shelved and the spine ships without it.
  **Status (2026-07-31): passed for 8-K filings only** (see §10 step 6) — pipeline
  investment proceeds scoped to filing-event scoring at short horizons; news-volume
  features remain shelved unless content-scored variants pass this same gate.

---

## 6. Self-improvement

- **6.1 Gated auto-retrain.** All fitted components refit on a schedule (default monthly).
  Promotion rule: a refit replaces the incumbent only if it wins the walk-forward comparison.
  Retrain cost is minutes of CPU; the gate, not the cost, is the design constraint.
- **6.2 Episodic/analog memory.** A state vector (regime posterior, vol/correlation summary,
  drawdown state, news intensity) is matched via exact k-NN against the full daily history.
  Retrieved analogs (a) condition risk estimates as a nonparametric prior and (b) appear in
  the daily briefing as explanation ("nearest analogs: Feb 2018, Aug 2015"). The buffer
  grows as Meridian lives — improvement with zero weight updates.

---

## 7. Validation protocol

- **Walk-forward.** Expanding or rolling-origin evaluation over decades of daily data;
  every tuned parameter (including regime sensitivity) is fit only on data before each
  evaluation window. No component sees future data at any point; `fetched_at` is the
  availability clock for all text/event features.
- **Null harness.** SPY + volatility-target rule implemented in the same engine, same cost
  model, same calendar. All headline claims are relative to it.
- **Costs.** Commission-free assumptions but honest spreads/slippage per instrument, and
  turnover accounting. Taxable-account drag is reported alongside.
- **Promotion gates.** Challenger-vs-incumbent comparisons (§1.7, §6.1) use this same
  protocol — one harness for everything.
- **Known-limitation reporting.** Survivorship bias in the stock sleeve (Yahoo-sourced data
  lacks delisted names) is documented with every backtest; it inflates absolute equity
  returns but has limited effect on vol/regime estimation, which is what Meridian consumes.

---

## 8. Execution

Broker-API automation constrained by hard, model-external guardrails: per-position weight
caps, gross-exposure cap, daily turnover cap, order-size sanity checks, trading-hours
checks. Every action produces a notification; a kill switch halts trading immediately.
Failure of any upstream component degrades to the most conservative available behavior —
never to unbounded action. Initial guardrail values are maximally conservative and loosen
only as the trust ladder (§1.12) advances.

---

## 9. Infrastructure

| Concern | Where it runs |
|---|---|
| News archive collection | GitHub Actions cron (never gaps with the appliance) |
| Live market/event monitoring | Framework Desktop appliance |
| LLM scoring | Appliance (local model); OpenAI-compatible endpoint abstraction |
| Daily decision + execution | Appliance |
| Research, backtests, challenger training | RTX 3090 workstation |
| Retraining (classical spine) | Appliance CPU (minutes) |

No component requires rented datacenter GPUs. Hardware ceases to be a design input beyond
this table.

---

## 10. Roadmap (build order)

1. **Archive live** — push the feeds archiver + workflow so collection starts. *(implemented, live in CI)*
2. **Data layer** — historical loaders (upstream `adjClose` + dividends/splits cross-check), macro panel, point-in-time store. *(implemented: `python/market_data/`; ETF core verified at ≤2e-6 divergence)*
3. **Null baseline + harness** — SPY + vol-target in the walk-forward engine with costs. Nothing else is evaluated until this exists. *(implemented: `python/harness/`; OOS 1998–2026 after costs — buy-hold 9.3% CAGR/0.44 Sharpe, vol-target 8.2%/0.49)*
4. **Classical spine** — vol models, shrunk correlations, regime detector; allocator v0 (ETF core only). *(implemented: `python/spine/` — EWMA incumbent verifies the 0.67 probing bar (0.666 measured), Ledoit–Wolf shrinkage, 2-state Gaussian HMM filtered causally, allocator v0)*
5. **Backtest vs null** — first honest race. Stock sleeve added only after the ETF core wins. *(raced 2026-07-31, twice: spine v0 OOS Sharpe 0.47; tuned spine — allocator mapping selected per window on train data only, incl. a vol_only fallback — 0.44. Null vol-target: 0.49. **Below the bar; not shipping.** Diagnostic: regime-modulated mappings win in-sample but lose OOS — the 2-state HMM posterior over (realized vol, VIX, T10Y3M, drawdown) adds no OOS value beyond vol-targeting. The spine's tail profile is far better (15% maxdd vs 42/55%), but the charter bar is risk-adjusted return. Path forward: richer regime features — news-flow intensity (gated by step 6), credit spreads — as challengers through this same race. **Vol-forecaster race (pre-registered in `python/vol_race/`, run 2026-07-31): FAIL — but with a valuable split verdict.** Stage 1 (forecast quality, common sample): VIX 0.720 / blend_har_vix 0.713 / blend_ewma_vix 0.708 all clear the incumbent EWMA's 0.666 at 21d — implied vol carries real forecast information beyond the incumbent. Stage 2 (the race): every graduate *loses* to the null (Sharpe 0.41–0.45 vs 0.49, OOS 1998–2026 after costs). Better correlation did not convert into a better book: the VIX-sized target trades ~2× the null's turnover (5.5 vs 3.0 one-way/yr) and its exposure timing is worse after unconditional mean calibration — the variance risk premium is state-dependent, so a global rescale misprices calm vs stressed periods. Lesson recorded: forecast-quality metrics do not gate shipping; only the race does. A premium-adjusted or cost-aware VIX sizing rule would be a *new* pre-registration, not a retune.)*
6. **Event study** — do filings/news precede realized-vol elevation? Gates all text work. *(run 2026-07-31, pre-registered gate in `python/event_study/study.py`: **PASS, scoped to 8-K filings** — +6.0% median 5-day vol uplift over the EWMA forecast, p=1.7e-3, n=957, 15 large caps 2015–2026. 10-Qs show negative uplift (scheduled disclosure resolves uncertainty); FNSPID news-burst counts and market-level count intensity show no signal — volume isn't information, content might be. Signal is short-horizon (gone by 21d) → LLM scores belong as per-name short-horizon risk features and break-glass inputs, not slow regime features. Caveat: overlapping forward windows make p-values optimistic; effect size is modest.)*
7. **LLM event pipeline** — only if §6 passes; scores as risk-engine features + break-glass severity. *(core implemented 2026-07-31: `python/event_pipeline/` scores 8-K content via chimeric → local qwen3.6 with structured output (severity 0–10, scope, direction); scores versioned by model@prompt-hash in append-only `data/scores/`, ~9s/filing. Verified live on real filings with sensible calibration. Panel backfill complete 2026-07-31: 1,017 scored 8-Ks, committed in `data/scores/`. **Score study (pre-registered gate in `python/score_study/`): FAIL** — within 8-K event days, LLM severity does not rank forward-vol uplift (spearman +0.004, p=0.46, n=957; serious sev≥5 vs routine sev≤2 median uplift −6.7%, p=0.74). The step-6 8-K day effect is flat across scored severity: knowing *that* an 8-K was filed carries the vol information; the LLM's *how bad is it* adds none. Direction diagnostic ordered correctly (risk_up 0.973 vs risk_down 0.901) but not significant. Consequence: **no allocator race with severity features** — scores are retained as break-glass severity inputs only (intraday monitoring), and the filing-occurrence flag itself remains the candidate risk feature. Caveats: scored by a 2026 model over historical text (backtest-legal, content-only, but anachronistic); overlapping forward windows. **Occurrence-flag race (pre-registered in `python/eightk_race/`, run 2026-07-31): gate PASSES but does not ship.** Challenger — null vol-target with equity halved for 5 sessions after a panel 8-K intensity burst (causal z≥2; 9-name availability-limited panel; OOS 2022–2026) — after-cost Sharpe 0.57 vs null 0.49, and a matched-exposure constant de-lever placebo reproduces the null (0.48), so the edge is timing, not exposure level. But the post-hoc diagnostics say don't trust it yet: block-bootstrap Sharpe diff +0.08 with 90% CI [−0.14, +0.31] (P(≤0)=0.27); the pre-registered signal diagnostic is null (spearman −0.035); and the active-day edge comes from *return* differences (−0.6bp vs +7.4bp/day at similar vol) — return timing is off-charter as a mechanism and consistent with luck on n=344 active days. Disposition: **hold, don't ship** — EDGAR panel coverage widens every day the archive runs; re-race on the same frozen pre-registration as OOS accrues (a real signal strengthens, a fluke fades). No parameter retuning.)*
8. **Episodic memory** — analog retrieval + briefing explanations.
9. **Paper trading** — ≥3 months live, behavior-match criteria defined in advance.
10. **Execution + guardrails** — broker integration, then staged capital per the trust ladder.

---

## Appendix A — Verified data availability (finance-query v2.8.0, checked 2026-07-30)

| Need | Endpoint | Verified |
|---|---|---|
| Daily OHLCV, decades | `/v2/chart/{s}?range=max&interval=1d` | SPY 8,430 bars → 1993; KO → 1985; TLT/GLD/BIL/SH/QQQ → inception |
| Adjusted close (total return) | same chart endpoint, `adjClose` field | populated since 2026-07-30; verified 2026-07-31: SPY/AAPL full history, no nulls, dividend+split adjusted |
| Dividends (adjClose cross-check) | `/v2/dividends/{s}` | SPY: 135 payments, full history |
| Splits | `/v2/splits/{s}` | AAPL: 5 splits → 1987 |
| FRED series | `/v2/fred/series/{id}` | VIXCLS: 9,541 obs, 1990 → present |
| Treasury curve | `/v2/fred/treasury-yields?year=` | 2008 verified (251 rows) |
| EDGAR full-text search | `/v2/edgar/search?q&forms&startDate&endDate` | 2008 query: 2,155 hits with file dates |
| Filing history / XBRL | `/v2/edgar/submissions`, `/v2/edgar/facts` | up to 1,000 filings/company; multi-MB facts |
| Earnings transcripts | `/v2/transcripts/{s}/all` | AAPL: 61 transcripts, 2011–2026 |
| Live feeds | `/v2/feeds?sources=` | 30 sources incl. `sec-filings`; 607 entries/sweep |
| Live news + sentiment | `/v2/news` | 50 items with sentiment labels |
| Quotes / hours / intraday | `/v2/quotes`, `/v2/hours`, `/v2/chart` (1m–1h) | working |
| Known gap | historical news archive | none upstream — closed by §3.4 from day one |
