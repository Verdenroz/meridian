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
first, in the same harness. Below the bar Meridian deserves zero dollars, and that result
would be accepted.

**The bar (amended 2026-07-31).** A challenger ships only if, after costs, out-of-sample,
walk-forward, it beats the null outright on Sharpe — **or** satisfies all three of:

1. **Sharpe non-inferiority:** OOS Sharpe ≥ null Sharpe − 0.05.
2. **Materially better tail:** OOS max drawdown ≤ two-thirds of the null's.
3. **Timing, not timidity:** it beats, on Sharpe, a constant-exposure placebo — the null
   statically de-levered to match the challenger's realized volatility. A drawdown
   reduction that a static de-lever reproduces is exposure, not skill, and does not count.

The co-criterion widens the gate for risk-superior parity; it never narrows the original
Sharpe path. These thresholds are frozen — changing them is a new amendment, not a retune.

**Prospective-only clause.** This amendment was motivated by the 2026-07-31 credit-spread
race (the 5f spine reached Sharpe parity at a third of the null's drawdown), so no
candidate may pass on evidence available at amendment time alone — that would be grading
the data by a bar it inspired. Shipping under the co-criterion additionally requires
(a) the candidate's frozen pre-registration still passing at re-race as OOS accrues, and
(b) the pre-registered paper-trading behavior-match (§10 step 9) over ≥3 months of shadow
decisions. Rationale: the charter's product is risk management; a bar blind to the
drawdown path measured the mandate's inputs, not its output.

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

**Status (2026-08-01): implemented.** `meridian-bus` (`src/`, ~1k lines replacing the 12k-line
training stack, which is archived reference-only in `legacy/databus-v4/`) sweeps quotes +
today's 5m bars each minute in-session, evaluates the §4.5 severity rules against daily
baselines (EWMA(0.94) vol, 20-day volume, monthly intraday volume profile), and journals
every cycle (heartbeat-style, `logs/bus/`) plus a `data/bus/latest.json` snapshot. Alerts
append to git-tracked `data/breakglass/alerts.jsonl` — advisory records until §10 step 10
exists. Trigger thresholds were calibrated 2026-08-01 on SPY 1993–2026 EOD history to the
§4.5 base rate: watch union ≈4 episodes/yr, break-glass union
(ret_z≥3.5 | gap_z≥2.5 | ret_z≥3 & vol_ratio≥2.5) ≈1.9 episodes/yr. Triggers are
price/volume-only (LLM-free per §4.5); the SEC 8-K feed is polled as context recorded on
alerts, never a trigger (consistent with the held occurrence-flag race and the failed score
study). One alert per (session, level), escalation allowed, restart-safe via the journal.
Alerts notify (§1.11) via an optional `MERIDIAN_ALERT_WEBHOOK` env var — configuration
outside the model's reach; absent or failing webhook degrades to journal+log, never to a
blocked alert. Appliance deployment: `deploy/meridian-bus.service` (systemd, §9).

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
  **Status (2026-08-01): the content-scored variant passed** (`python/tone_study/`,
  pre-registered; FinBERT tone over the FNSPID corpus via the legacy sentiment cache,
  1.41M headlines, zero new inference). Both legs clear at the 5-day horizon:
  negative-tone days (≥3 headlines, mean ≤ −0.3; n=596) beat a *matched news-active
  neutral control* on forward-vol uplift (+6.5% median, p=0.0078) — while beating the
  plain baseline only weakly (+4.7%, p=0.035), i.e. the elevation is content, not
  volume, and neutral news-active days sit *below* baseline; market-wide negative tone
  rank-predicts SPY 5d residual vol (spearman 0.128, p≈1e-14, n=3,550 days,
  2009–2023). Same fingerprint as 8-Ks: signal gone by 21d (uplift turns negative) —
  tone belongs at short horizons, beside the filing-occurrence flag, never as a slow
  regime feature. Unlocked, each behind its own pre-registered race before touching
  allocation: per-name negative-tone intensity as a short-horizon risk feature, and
  market tone as a break-glass/monitoring input. Caveats pre-registered: FNSPID
  publisher timestamps (next-close availability convention), overlapping windows make
  p optimistic, 2019-era FinBERT scoring older text; the corpus ends 2023-12 — live
  continuation must come from the feeds archive scored by the same versioned scorer.
  **Status (2026-08-01, later): the tone-brake race FAILED** (`python/tone_race/`,
  pre-registered). Market-wide negative-tone burst brake (frozen brake family: z ≥ 2,
  scale 0.5, 5 sessions) on the null vol-target, OOS 2010-04-30 → 2023-12-18 after
  costs: Sharpe 0.786 vs null 0.803, and max drawdown *worse* (0.206 vs 0.189), so
  route (b) dies at the tail bound; the tone∪credit compound brake is worse still
  (0.760). The signal itself is the strongest diagnostic of any race — spearman
  +0.144 (p≈0) of the negative-tone z on SPY 5d residual vol — which sharpens the
  cross-race lesson: every binary brake on the null has now failed (8-K, credit,
  tone) because EWMA sizing already de-levers through the same episodes; braking the
  rebound days makes tails worse, not better. Orthogonal channels have only ever
  added value through the regime model (credit 4f→5f). Tone's market-level defense
  deployment is closed; remaining unlocked paths are per-name sleeve sizing (blocked
  on the sleeve engine) and break-glass/monitoring enrichment (advisory, no race).

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
  **Status (2026-07-31):** retrieval + briefing implemented (`python/episodic/`, §10
  step 8). Analogs are explanation-only: (a) — conditioning risk estimates — would change
  allocations and therefore awaits its own pre-registered race through the §1.7 gate.

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
  protocol — one harness for everything. The promotion bar itself is §1.3 (amended
  2026-07-31): outright Sharpe win, or Sharpe non-inferiority (−0.05) + max-drawdown
  co-criterion (≤⅔ null) + matched-vol placebo control, prospective-only.
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
5. **Backtest vs null** — first honest race. Stock sleeve added only after the ETF core wins. *(raced 2026-07-31, twice: spine v0 OOS Sharpe 0.47; tuned spine — allocator mapping selected per window on train data only, incl. a vol_only fallback — 0.44. Null vol-target: 0.49. **Below the bar; not shipping.** Diagnostic: regime-modulated mappings win in-sample but lose OOS — the 2-state HMM posterior over (realized vol, VIX, T10Y3M, drawdown) adds no OOS value beyond vol-targeting. The spine's tail profile is far better (15% maxdd vs 42/55%), but the charter bar is risk-adjusted return. Path forward: richer regime features — news-flow intensity (gated by step 6), credit spreads — as challengers through this same race. **Vol-forecaster race (pre-registered in `python/vol_race/`, run 2026-07-31): FAIL — but with a valuable split verdict.** Stage 1 (forecast quality, common sample): VIX 0.720 / blend_har_vix 0.713 / blend_ewma_vix 0.708 all clear the incumbent EWMA's 0.666 at 21d — implied vol carries real forecast information beyond the incumbent. Stage 2 (the race): every graduate *loses* to the null (Sharpe 0.41–0.45 vs 0.49, OOS 1998–2026 after costs). Better correlation did not convert into a better book: the VIX-sized target trades ~2× the null's turnover (5.5 vs 3.0 one-way/yr) and its exposure timing is worse after unconditional mean calibration — the variance risk premium is state-dependent, so a global rescale misprices calm vs stressed periods. Lesson recorded: forecast-quality metrics do not gate shipping; only the race does. A premium-adjusted or cost-aware VIX sizing rule would be a *new* pre-registration, not a retune. **That follow-up ran 2026-08-01 (`python/vrp_race/`, pre-registered): FAIL, decisively.** Three factorial challengers built from the frozen postmortem — per-tercile VRP calibration, wide rebalance band (0.05), both — all land at OOS Sharpe 0.41 vs the null's 0.49, indistinguishable from the failed global-scale reference (0.4125); the wide band cuts turnover 5.5→4.5/yr without moving Sharpe. Attribution is clean: neither diagnosed defect was the binding one — VIX-sized targeting is a worse book than EWMA targeting in this engine regardless of premium handling, and the VIX branch is closed unless a future challenger brings *new information*, not recalibration. **Credit-spread race (pre-registered in `python/credit_race/`, run 2026-07-31): gate passes on the letter, held on the evidence.** Both challengers nominally beat the null (brake 0.4995, spine-5f 0.4994 vs null 0.4889, OOS 1998–2026 after costs) but block-bootstrap margins are noise (brake P(≤0)=0.36; spine-5f CI [−0.22,+0.24], P(≤0)=0.50). The real finding is attribution: **BAA10Y is the first regime feature to add OOS value** — the 5-feature tuned spine jumps from the 4f reference's 0.44 to Sharpe parity with the null while keeping the spine's tail profile (maxdd 13.7% vs the null's 41.7%, vol 8.3% vs 13.3%). Sharpe-parity-at-a-third-of-the-drawdown is materially different risk, but under the Sharpe-only bar as raced: **hold, don't ship**; re-race as OOS accrues. The open charter question this raised — should the bar price tail risk — was resolved 2026-07-31: §1.3 amended, prospectively only, to add a Sharpe-non-inferiority + maxdd co-criterion with a matched-vol placebo control. The 5f spine is the leading candidate under the amended bar but must clear it on post-amendment evidence (re-races as OOS accrues + the step-9 shadow behavior-match), not on this backtest. **Quality-spread race (`python/quality_race/`, pre-registered and run 2026-08-01): FAIL.** Intended as ICE BofA HY OAS, substituted on availability (upstream truncates ICE BofA to ~3y) with the Moody's Baa−Aaa quality spread. The 6f spine (0.4485) is *worse* than the 5f reference (0.5021) — the default-premium channel is already in BAA10Y's level, and the extra feature dilutes the HMM; the brake route is Sharpe-non-inferior but leaves the tail untouched. First live exercise of the amended §1.3 gate: the 6f spine reached the (b) branch and the matched-vol placebo correctly killed it (placebo 0.486 at matched vol vs 0.4485) — the control catching exposure-masquerading-as-skill exactly as §1.3(3) intends.)* **Rate-vol race (`python/rate_race/`, pre-registered and run 2026-08-01): PASS, route (a) — the project's first outright Sharpe wins.** Feature: log EWMA vol of daily DGS10 changes (FRED, 1962+) — the bond-turbulence channel the 2022-style stress exposed. Both challengers beat the null outright (5f-rate 0.5216, 6f credit+ratevol 0.5278 vs null 0.4906, OOS 1998–2026 after costs) at a third of its drawdown (15.5%/17.0% vs 41.7%); unlike the quality spread, the channels *compound* — 6f beats both 5f parents. Non-gating bootstrap (21/2000/20260801, same registration as the deep race): margins are noise at daily resolution (5f-rate +0.031 CI90 [−0.16,+0.23] P(≤0)=0.37; 6f +0.037 [−0.17,+0.24] P(≤0)=0.37) — so, per the credit-race discipline: **held, not shipped**; monthly re-race + shadow accrual adjudicate. The 6f spine replaces 5f-credit as the leading candidate. **Hedge-veto race (`python/hedge_race/`, pre-registered and run 2026-08-01): PASS, route (a).** First hedge-leg hypothesis: TLT barred from the defensive sleeve for 21 sessions after a rate-vol burst (causal z ≥ 2) — the 60-session correlation check is right about the 2022 state but late by construction. The veto strictly improves its 5f-credit base (Sharpe 0.5091 vs 0.5021, maxdd 12.95% vs 13.71%) and beats the null outright; bootstrap noise like the others (+0.019, P(≤0)=0.43) — **held**. Mechanism note from the compound race: the veto's value is *conditional on the regime model lacking the rate channel*. **Stock-bond correlation race (`python/corr_race/`, pre-registered and run 2026-08-01): FAIL.** 60-session corr(SPY, −ΔDGS10) as a feature: alone it adds nothing (0.4495; the matched-vol placebo again correctly killed the (b)-route), and stacked on the 6f leader it dilutes (7f 0.4294 vs 0.5278) — the quality-race failure mode. The co-movement *sign* regime carries no allocator value beyond what rate vol already provides; correlation stays served by the safe-asset check and episodic state vector. **Compound race (`python/compound_race/`, pre-registered and run 2026-08-01): PASS route (a) vs the null (0.5206) but *below its 6f parent* (0.5278)** — the veto helps the credit-only spine yet hurts 6f: once rate vol is a regime feature the HMM already de-risks through rate turbulence and the veto double-counts. Honest disposition: the compound is not the leader; **6f alone leads**, the veto is the fallback for regime models without the rate channel. Cross-race lesson consolidated 2026-08-01: three consecutive brake failures (8-K deep, tone, tone∪credit) vs regime-feature and hedge-leg passes — orthogonal information helps through the *regime model* or *sleeve membership*, never as binary exposure brakes on top of EWMA sizing. **Combination diagnostics (2026-08-01, non-gating; scratch analysis over the recorded strategies):** (1) the three stress channels fire on largely *disjoint* day sets — credit∩rate Jaccard 0.15 (796 vs 1,168 active OOS days, 258 shared), credit∩tone 0.13, rate∩tone 0.17 — they are complementary detectors of different stress kinds, not confirmations of one another; this is why credit+rate compound in the HMM and why tone's information (strongest spearman of any channel) failed only in *deployment*. (2) The rate feature does drive de-risking where the veto fires (5f-rate cuts equity to 0.74× in veto windows, the deepest), but 6f's relative de-risking (0.90×) is not deeper than 5f-credit's (0.85×) — the veto-on-6f "double-count" story is plausible, not proven; the operative fact is simply that stacking adds nothing reliable. (3) Marginal contributions within 6f: credit's seat next to rate vol bootstraps to nothing (+0.006, P(≤0)=0.46), rate's next to credit is larger but unresolved (+0.026, P(≤0)=0.34); the two 5f books correlate only 0.727 — genuinely different behavior. Consolidated dispositions: 6f leads, 5f-rate is the parsimonious runner-up and now shadow-logs alongside it (`spine5f_rate`) so credit's seat is adjudicated on live evidence; the veto stays a held fallback for regime models lacking the rate channel; tone stays monitoring/per-name-future. The disjoint-episode finding motivated one derived registration: the regime-granularity race (`python/nstate_race/` — does a 3rd HMM state separate stress kinds?). **Regime-granularity race (`python/nstate_race/`, pre-registered and run 2026-08-01): PASS route (a) — the largest margin of any race.** A 3-state HMM on the unchanged 6f features (n=3 a single a-priori choice; allocator still reads the top-vol state as stress): Sharpe **0.5947 vs null 0.4906** (2-state 6f: 0.5278), CAGR 8.56% at 11.0% vol, maxdd 20.2% — half the null's. Mechanism matches the motivating diagnostic: a middle state lets the book stay invested through mild/localized stress instead of blending all stress kinds into one defensive response. Bootstrap (21/2000/20260801): +0.104 vs null CI90 [−0.06,+0.27] **P(≤0)=0.15** — unresolved at daily resolution but the strongest evidence of any candidate (all others 0.34–0.46); vs 2-state +0.067, P(≤0)=0.28. Disposition: **held — the new leading candidate**; wired into the monthly re-race and shadow (`spine6f_3state`). Registered risk stands: 3 Gaussian states on 6 features is ~3× the parameters — the monthly re-race and shadow behavior-match are exactly the guards for that. **Four-state race (`python/nstate4_race/`, pre-registered and run 2026-08-01): FAIL, decisively** — 0.4376, below even the null, maxdd 31%. The granularity curve is a sharp peak, not a slope: 2-state 0.528 → 3-state 0.595 → 4-state 0.438. This strengthens the 3-state result (a peak between two inferior neighbors is harder to attribute to luck than an endpoint) and, per the registration, closes n-scanning: further regime-capacity work goes through learned challengers (§1.7), not n+1 registrations. **Topic-salience race (`python/salience_race/`, pre-registered and run 2026-08-01): PASS on the letter, weakest of the held candidates.** First slow news-derived regime feature — inflation/rates anchor cosine over the legacy 1.13M-embedding cache (endpoint verified to reproduce cached vectors, cosine 0.9995), 21-session mean, causal z, zero-filled outside FNSPID coverage; scored on the availability-clipped span 2010-02→2024-01. The 7f (6f + salience) beats the null outright (0.8834 vs 0.8639; 6f ref 0.8651 on the same easy era) but salience *alone* fails (0.7995), and the bootstrap is noise (vs null +0.020 P(≤0)=0.27; vs 6f +0.018 P(≤0)=0.42). Disposition: **held, low confidence** — and *local-only adjudication*: the feature depends on the local embed cache (not in git, not reachable from CI), so it is excluded from the CI monthly re-race and re-raced locally; live continuation would embed feeds-archive titles through the appliance endpoint. **Deep filing-tone study (`python/filing_tone/`, pre-registered and run 2026-08-01): FAIL — the filing-text branch closes on all three fronts.** FinBERT over 4,856 8-K texts (RTX 3090, ~3 min; scores written to the standing sentiment cache; 15-name panel, 1994–2026, 4,871 filing days — deeper than the deep race). Only 61 filing days (1.3%) clear the negative-tone threshold — 8-K prose is neutral boilerplate, so the content leg cannot even reach its n≥100 events; the rank leg finds a real but sub-gate signal (spearman 0.051, p=2e-4 at 5d, gone by 21d). With severity FAIL + occurrence FAIL + content FAIL, filing *text* carries no deployable vol information at the index or per-name level; headline content (tone study) remains the only passing text signal. **Episodic news analogs shipped 2026-08-01** (explanation-only, no gate): per-day news centroids over the embedding cache with a live path that embeds feeds-archive titles via the endpoint and grows the cache; 2020-03-16 retrieves the COVID-crash news cluster. **Champion consolidation race (`python/champion_race/`, pre-registered and run 2026-08-01): PASS route (a), and the axes are additive** — on the salience-clipped span, 7f 3-state 0.9005 > 6f 3-state 0.8854 > 7f 2-state 0.8834 > null 0.8639: the feature axis and the state axis contribute independently (unlike the veto compound, which subtracted). Margins on this easy era are individually noise-scale; consolidated standing: **6f 3-state leads on the full-span evidence; 7f 3-state is the salience-era variant** — both held, salience's seat adjudicated by local re-races only (embed-cache dependency). **Sector-dispersion race (`python/dispersion_race/`, pre-registered and run 2026-08-01): PASS on the letter, channel not adopted.** First use of the nine 1998-vintage SPDR sector ETFs (fetched 2026-08-01 with the standard adjClose cross-check, zero mismatches): causal z of log EWMA cross-sectional sector-return dispersion, zero-filled pre-coverage — the direct measurement of stress *localization* the disjoint-episode diagnostic implied. Alone the channel fails exactly as the controls intend: 5f-disp (0.4779) reaches the (b) branch and the matched-vol placebo kills it (0.4871). Stacked, 7f-disp beats the null outright (0.5333) but is noise-distance above its 6f parent (0.5278), and on the 3-state leader it *dilutes* badly (0.5029 vs 0.5947, maxdd 41.3% — the quality/corr failure mode at the worst spot). Reading: whatever localization information sectors carry, the third state already extracts it from the index-level features; **dispersion is closed as a regime feature** (local sector data; excluded from the CI re-race). **Student-t emission race (`python/temission_race/`, pre-registered and run 2026-08-01): PASS vs the null; the emission family does not displace Gaussian.** First learned-regime challenger after the 4-state closure: diagonal Student-t emissions at a frozen ν=5 (estimating ν was deliberately excluded as a knob), EM/init/ordering/filter otherwise identical to the incumbent, via an additive `hmm_cls` hook on the spine (default Gaussian — frozen races unchanged, verified by the test suite). Both challengers beat the null outright (t5 2-state 0.5310, t5 3-state 0.5471 vs 0.4906) — but t5 2-state is only noise above Gaussian 2-state (0.5278) and t5 3-state is clearly *below* the Gaussian 3-state leader (0.5947) with a worse tail (26.9% vs 20.2%). Mechanism reading: the third state's value comes from the very tail days the t-density down-weights — robust emissions and a dedicated stress state are substitutes, and the state does the job better. **Gaussian emissions stand**; the race joins the monthly re-race so the substitution story is adjudicated on accruing OOS. **Turnover race (`python/turnover_race/`, pre-registered and run 2026-08-01): PASS route (a) across the board — the new leading candidate, on the best-resolved margin yet.** Feature: causal z of log SPY dollar volume (close×volume, 1993→present — data in the store untouched by any race; the trailing z absorbs the series' secular growth) — the *participation* channel no price or macro feature sees: capitulation trades multiples of normal dollar volume at the same realized vol. All three challengers beat the null outright: 5f-turn 0.5417, 7f-turn 0.5672, and 7f-turn 3-state **0.6295 vs null 0.4906** (CAGR 8.79% at 10.7% vol, maxdd 20.5% — half the null's 41.7%). Turnover improves *every* parent it is stacked on (4f→5f-turn +0.10 over the 4f reference's 0.44, 6f 0.5278→7f-turn 0.5672, 6f-3s 0.5947→7f-turn-3s 0.6295) — the credit+rate compounding signature, not the dilution signature. Non-gating bootstrap (21/2000/20260801): vs null +0.139 CI90 [−0.03,+0.31] **P(≤0)=0.088** — the strongest evidence of any candidate (3-state was 0.15) — and the marginal turnover seat on the 3-state leader is the best-resolved seat ever measured (+0.035 CI90 [−0.02,+0.09], P(≤0)=0.13; credit's seat 0.46, rate's 0.34); alone-vs-null is noise (+0.05, P(≤0)=0.36) — the channel's value concentrates when it compounds with the others. Disposition: **held, the new leading candidate** — `spine7f_turn_3state` shadow-logs beside its 6f 3-state parent (the turnover seat adjudicated live), and the race joins the monthly CI re-race (its data is inside the default fetch). **Octet consolidation race (`python/octet_race/`, pre-registered and run 2026-08-01): PASS vs the null on the letter, but the stack does not consolidate.** The full 8f (6f + turnover + salience) 3-state on the salience-clipped span beats the null (0.8796 vs 0.8639) yet lands *below both 7f parents* (7f-turn 0.8928, 7f-sal 0.9005) and even the 6f reference (0.8854) — the compound-race lesson replayed on the feature axis: two individually passing channels are not additive by right; here they interfere (plausibly overlapping information — news-flow composition and trading intensity co-move — plus 3 Gaussian states over 8 features approaching the capacity edge the quality race first exposed). Also recorded honestly: turnover's own margin on this easy clipped era is thin (+0.007 over 6f-3s) — its value concentrates in the pre-2010 stress eras the clip excludes, which is why the full span remains the standard of evidence. Disposition: **8f rejected; 7f-turn 3-state stays the full-span leading candidate; 7f-sal 3-state stays the salience-era variant** — the two 7f books are alternatives adjudicated by re-race and shadow accrual, not stacked. Local-only like every salience-dependent race (embed cache), excluded from the CI re-race. **Overnight-gap race (`python/gap_race/`, pre-registered and run 2026-08-01): PASS on the letter, channel not adopted.** First use of the chart store's open field (verified clean 1993→present; max |gap| 11% on 2020-03-16, the COVID limit-down open): log EWMA vol of overnight log-gaps — how much volatility arrives while the market cannot trade. All three challengers beat the null outright, but attribution says redundancy: 5f-gap (0.5320) sits at the 6f reference's level (0.5278) — the gap channel adds nothing the existing vol/credit/rate channels lack; 7f-gap *dilutes* (0.5251), and the 8f stack drags the leader (0.6175 vs 0.6295). Gap turbulence is closed as a regime feature; the open field stays available for other uses. **Tone-feature race (`python/tonefeat_race/`, pre-registered and run 2026-08-01): PASS — the tone channel finally lands, in exactly the shape the record predicted.** The loop-closing race: the tone study passed (the only text-content signal), the tone brake failed, and the three-brake postmortem said orthogonal information helps through the regime model only — yet tone had never been raced as a regime feature. Feature: session-aligned negative market-tone z, every constant inherited frozen from the tone race, zero-filled outside coverage; clipped span 2010-04→2023-12 (FNSPID). Alone and on 6f it FAILS (0.7624/0.7783 — under-diversified 2-state books on the easy era) — but the full stack **8f-turn-tone 3-state (0.8513) beats the null (0.8031), the 7f-turn leader (0.8304), and the 6f 3-state (0.8229)** on the clipped span: unlike salience (octet interference), the tone seat on the leader is positive. Non-gating bootstrap (21/2000/20260801, clipped): vs null +0.048 CI90 [−0.002,+0.095] **P(≤0)=0.0565 — the best-resolved vs-null margin of any candidate** (turnover full-span: 0.088); the tone seat vs 7f-turn +0.021, P(≤0)=0.26 (better than salience's 0.42 or rate's 0.34, still unresolved). Disposition: **held — 8f-turn-tone 3-state joins the candidate set as the clipped-span leader**; 7f-turn 3-state remains the full-span leading candidate. Adjudication is local-only (sentiment cache; no CI re-race, no shadow — CI cannot compute the feature). Standing limitation, honestly recorded: FNSPID coverage ends 2024-01, so the tone feature is zero on live dates until the feeds-archive headlines are FinBERT-scored — the natural eGPU continuation job, which would also let the candidate accrue true OOS. **Learned-regime race (`python/learned_race/`, pre-registered and run 2026-08-01 on the eGPU): FAIL, decisively — the first neural challenger loses to both the null and its classical twin.** Frozen spec: single-layer GRU (hidden 32, seed 20260801, deterministic CUDA verified bit-identical) on the leader's own 7f-turn features, trained per walk-forward window to predict forward-21d vol exceedance (risk, never returns), p_stress through the unchanged allocator/tuner. Result: 0.4476 vs null 0.4906, maxdd 35.0% — while the classical 7f-turn 3-state posts 0.6295 on *identical features*, so the comparison is attributable feature-for-feature: the supervised exceedance objective, refit on expanding windows, generalizes worse than the unsupervised HMM filter (30% base-rate labels + nonstationarity are a hard target; the HMM never has to name a threshold). Charter working as intended — §1.7's bar is real, the classical spine stays. The learned axis remains open (one architecture is not the axis), but the next neural challenger needs a different objective shape (e.g. density/filtering, not exceedance classification), registered fresh.
6. **Event study** — do filings/news precede realized-vol elevation? Gates all text work. *(run 2026-07-31, pre-registered gate in `python/event_study/study.py`: **PASS, scoped to 8-K filings** — +6.0% median 5-day vol uplift over the EWMA forecast, p=1.7e-3, n=957, 15 large caps 2015–2026. 10-Qs show negative uplift (scheduled disclosure resolves uncertainty); FNSPID news-burst counts and market-level count intensity show no signal — volume isn't information, content might be. Signal is short-horizon (gone by 21d) → LLM scores belong as per-name short-horizon risk features and break-glass inputs, not slow regime features. Caveat: overlapping forward windows make p-values optimistic; effect size is modest.)*
7. **LLM event pipeline** — only if §6 passes; scores as risk-engine features + break-glass severity. *(core implemented 2026-07-31: `python/event_pipeline/` scores 8-K content via chimeric → local qwen3.6 with structured output (severity 0–10, scope, direction); scores versioned by model@prompt-hash in append-only `data/scores/`, ~9s/filing. Verified live on real filings with sensible calibration. Panel backfill complete 2026-07-31: 1,017 scored 8-Ks, committed in `data/scores/`. **Score study (pre-registered gate in `python/score_study/`): FAIL** — within 8-K event days, LLM severity does not rank forward-vol uplift (spearman +0.004, p=0.46, n=957; serious sev≥5 vs routine sev≤2 median uplift −6.7%, p=0.74). The step-6 8-K day effect is flat across scored severity: knowing *that* an 8-K was filed carries the vol information; the LLM's *how bad is it* adds none. Direction diagnostic ordered correctly (risk_up 0.973 vs risk_down 0.901) but not significant. Consequence: **no allocator race with severity features** — scores are retained as break-glass severity inputs only (intraday monitoring), and the filing-occurrence flag itself remains the candidate risk feature. Caveats: scored by a 2026 model over historical text (backtest-legal, content-only, but anachronistic); overlapping forward windows. **Occurrence-flag race (pre-registered in `python/eightk_race/`, run 2026-07-31): gate PASSES but does not ship.** Challenger — null vol-target with equity halved for 5 sessions after a panel 8-K intensity burst (causal z≥2; 9-name availability-limited panel; OOS 2022–2026) — after-cost Sharpe 0.57 vs null 0.49, and a matched-exposure constant de-lever placebo reproduces the null (0.48), so the edge is timing, not exposure level. But the post-hoc diagnostics say don't trust it yet: block-bootstrap Sharpe diff +0.08 with 90% CI [−0.14, +0.31] (P(≤0)=0.27); the pre-registered signal diagnostic is null (spearman −0.035); and the active-day edge comes from *return* differences (−0.6bp vs +7.4bp/day at similar vol) — return timing is off-charter as a mechanism and consistent with luck on n=344 active days. Disposition: **hold, don't ship** — EDGAR panel coverage widens every day the archive runs; re-race on the same frozen pre-registration as OOS accrues (a real signal strengthens, a fluke fades). No parameter retuning. Automated 2026-07-31: a monthly CI re-race (`.github/workflows/monthly-rerace.yml` → `python/rerace/`) re-runs both held gates on their frozen parameters and appends verdicts to git-tracked, append-only `data/races/reraces.jsonl`. **Deep-history second look (`python/deep_eightk/`, new pre-registration, run 2026-08-01): FAIL — the high-power result.** Full per-CIK 8-K histories from SEC's submissions archive (the finance-query ~1,000-filing cap was the original binding constraint; predecessor CIKs for GOOGL/XOM handled explicitly), the event study's 15-name panel, OOS 2004–2026 (5.6× the span, 1,242 active days vs 344), coverage-aware rate intensity. The brake *loses*: Sharpe 0.633 vs null 0.666, bootstrap diff −0.033 CI90 [−0.10, +0.04] (P(≤0)=0.76), signal spearman 0.011 (p=0.16). Two independent pre-registrations now agree the occurrence signal has no per-day ranking power, and at scale the brake subtracts value; the original 2022–2026 pass reads as the fluke its own diagnostics flagged. The frozen monthly re-race continues untouched and adjudicates the original gate on its own terms.)*
8. **Episodic memory** — analog retrieval + briefing explanations. *(implemented 2026-07-31: `python/episodic/` — causal state vector (log EWMA vol, log VIX, curve slope, drawdown, BAA10Y credit spread, 60d mean pairwise SPY/TLT/GLD correlation; log rate vol added 2026-08-01 with the rate-race pass so bond-led stress retrieves its own kind — 2022-09-26 now analogizes to 2022/2007, explanation-only so no gate); exact k-NN in z-space standardized over only the buffer the query may see, with a 63-session exclusion window (≥ outcome horizon, so analog futures are known history) and 21-session separation so top-k spans distinct episodes. The briefing reports each analog's forward-21d return/vol/maxdd and the aggregate nonparametric prior beside the standing EWMA forecast. Two spec'd state inputs deferred, documented in the module: the regime posterior (a fitted transform of these same observables — raw features keep the matrix fit-free and point-in-time) and news intensity (archive too young; joins as history accrues). The correlation summary requires the full core universe, so the searchable buffer starts early 2005 (GLD + 60 sessions) and grows daily. Explanation-only by construction: analogs condition no allocation; wiring them into the allocator or risk estimates is a new challenger through the §1.7 gate. Sanity: querying 2020-03-16 retrieves Oct 2008, Aug/Oct 2011, Dec 2018.)*
9. **Paper trading** — ≥3 months live, behavior-match criteria defined in advance. *(shadow stage live 2026-07-31: `python/shadow/` logs daily target weights for the null vol-target, credit brake, and 5f spine to append-only, git-tracked `data/shadow/decisions.jsonl` (point-in-time, `fetched_at`-stamped, not rebuildable) via a daily post-close workflow. This is the evidence accumulator only — behavior-match criteria must be pre-registered before any evaluation reads the log. The live decisions double as true-OOS resolution for the two held races. **Behavior-match pre-registration frozen 2026-07-31** (`python/behavior_match/criteria.py`): window 2026-07-31 → 2026-10-30; evaluation opens 2026-11-02 and the evaluator refuses to run earlier — no peeking at partial windows, no override flag. Per-strategy PASS requires all of C1 coverage (≥60 logged trading days, no missing run >5 sessions, uniform `job_version`), C2 fidelity (every logged row re-derived by re-running the exact shadow procedure at evaluation time; ≥95% of rows within 1% per-symbol weight, none beyond 5%; mismatches from upstream data revisions count — no discretionary exclusions), and C3 behavior in-distribution (window mean gross exposure, annualized decision turnover, and brake-active frequency for the credit brake, each inside the 5–95th percentile band of same-length rolling windows from a shadow-equivalent reference replay ending 2026-07-30; the spine's tuner refits every 21 sessions in the replay — a recorded compute concession). It is not a performance gate: no return, Sharpe, or drawdown number participates. PASS satisfies §1.3 clause (b) only; FAIL retires and renames the contender and restarts its clock. Criteria may be applied to later same-length windows; the parameters never change.)*
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
