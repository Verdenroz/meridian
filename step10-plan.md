# Meridian — Step 10 plan: execution + guardrails

**Status:** decisions closed 2026-09-01 — no code written yet
**Opened:** 2026-08-30 · **Last updated:** 2026-09-01
**Scope:** roadmap §10 — broker integration, then staged capital per the trust ladder (§1.12).

Living document. Update the workstream checkboxes and the progress log as work lands;
append to the decision table rather than rewriting it. `meridian-spec.md` §1 Charter is
authoritative over this file. Nothing here may relax a charter invariant; where this plan
and the spec disagree, the spec wins and this file gets corrected.

---

## 0. Position going in

The research phase is saturated, not stalled. Thirty pre-registered races have closed every
channel that was open; the negative results are decisive and durable:

- Brakes on the null are a dead deployment family (three consecutive failures, including the
  strongest signal diagnostic measured).
- Learned challengers lose to classical twins on identical features, in two opposite
  objective shapes. Architecture is not the lever; new data is.
- Regime granularity peaks at 3 states (0.528 / 0.595 / 0.438 across 2/3/4).
- Text is closed on filings and on per-name breadth. Headline content is the only survivor.

No positive result yet clears the bar for committing capital. The leading candidate (7f-turn 3-state) beats the
null by +0.139 Sharpe with CI90 [−0.03, +0.31], P(≤0) = 0.088 — the best-resolved margin in
the program, and its interval still straddles zero, before any correction for having been
selected as the max of ~30 races on one panel.

Three measurements taken 2026-08-30 sharpen this considerably.

- **Divergence.** The leader's book differs from the null's on 41.8% of the 7,171 sessions
  since 1998 — but the distribution is bimodal, not Bernoulli: 98–99% in 2001–04 and
  2008–09, 0–4% in calm years (2006, 2013, 2014, 2017, 2023, 2024), 8.3% in 2026 YTD. The
  edge lives entirely inside regime episodes; off-episode the two books are the same book
  (calm-days-only Sharpe 1.136 leader vs 1.139 null, diff −0.004).
- **Leave-one-episode-out.** Nine episodes derived from the divergence series. Dropping any
  one of them leaves the edge positive: min +0.072 (dropping 1999–2005, 21% of the sample),
  median +0.143, max +0.171. Dropping 2007–2010 leaves +0.119. **The result does not rest on
  one crisis** — the single largest specific objection to the leader is now dead.
- **Shadow.** 21 of 21 logged sessions have every contender and the null at SPY 1.00, cash
  0.00. The window is accumulating calendar, not evidence.

What LOEO does *not* establish. Within-episode the leader beats the null in only **4 of 9**
episodes (in-episode Sharpe diffs range −0.592 to +0.192). The full-sample edge comes from
the aggregate effect on the compounding path — max drawdown 0.205 vs 0.417, 49% of the
null's, comfortably inside §1.3(b)'s two-thirds bound — not from being reliably right in any
given crisis. The leader is best characterized as a **drawdown reducer with an unreliable
per-episode return effect**. Selection across ~30 races on one panel also remains
uncorrected, and the episodes are endogenous to the leader (defined as the days it chose to
act), so this is a robustness check, not an independent out-of-sample test.

The behavior-match gate opens 2026-11-02 and, on this trajectory, will pass having tested a
book that was identical to buy-and-hold on ~56 of 60 days. It satisfies §1.3(b); it is not a
performance gate and should not be read as one.

Step 10 is gated on none of this. It is required for any book that ever runs, it is the
longest-lead and least-exercised part of the system, and it is the only work whose value does
not depend on which candidate eventually wins.

---

## 1. The decision that shapes the build: null-first

**Ship the null vol-target as the first production book.** Not the leader, not a held
candidate — the baseline everything is raced against.

The charter bar is *beat SPY plus a simple vol-target rule*. That rule has not been beaten
decisively, and it is a legitimate production strategy in its own right: 0.491 Sharpe at 19%
max drawdown against SPY's 55%. Running it live:

- exercises the entire execution path — orders, fills, drift, reconciliation, guardrail
  enforcement, kill switch, break-glass — with zero model risk and no dependence on a
  contested edge;
- makes the trust ladder real rather than hypothetical, and gives stages 3 and 4 something
  to demote *from*;
- converts the leader from a backtest artifact into a challenger against a **live
  incumbent**, which is §1.7's "classical spine runs production" applied one rung lower than
  it has been applied so far;
- and fixes the evidence problem structurally: when the next stress episode arrives, the
  comparison happens on a real book instead of a paper one.

Consequence for this plan: **the execution layer must be strategy-agnostic from day one.** It
consumes target weights and knows nothing about regimes. Promoting a candidate later is a
config change, never an execution-layer change.

---

## 2. Open decisions (blocking)

| # | Decision | Options | Decision | Status | Needed by |
|---|---|---|---|---|---|
| D1 | Broker | Alpaca, IBKR, Schwab, Tradier, tastytrade | **IBKR Lite, cash account.** Alpaca eliminated by D7: it has never offered cash accounts (every account is margin; unchanged as of 2026-03). Tradier was the low-ops runner-up (bearer-token REST, $10/mo) and is the fallback if IB Gateway proves unmaintainable on the appliance. | decided 2026-09-01 | — |
| D2 | Initial live capital | — | **$10,000.** Enough that spread, regulatory fees, and whole-share rounding are observable in W7; small enough that total loss is irrelevant. | decided 2026-09-01 | — |
| D3 | Live book identity at stage 3 | null only / null + leader split | **Null only.** Leader and 8f-tone stay shadow contenders against the live incumbent. Promotion is a config change plus a pre-registered decision. | decided 2026-09-01 | — |
| D4 | Where execution runs | appliance / GitHub Actions / both | **Appliance, CI as watchdog.** The appliance returns to daily production service; a CI cron alerts on a missed decision. | decided 2026-09-01 | — |
| D5 | Kill-switch mechanism | file flag / broker-side / both | **Halt file + documented manual broker-side liquidation.** No ntfy action button; fewer moving parts. | decided 2026-09-01 | — |
| D6 | Notification channel | ntfy / email / both | **ntfy topic**, reusing the `MERIDIAN_ALERT_WEBHOOK` wiring the Rust bus already has. | decided 2026-09-01 | — |
| D7 | Account type | cash / margin-never-borrow | **Cash account.** Borrowing is impossible at the broker, not only in our code. Consequence: T+1 settlement rules (good-faith violations) apply and are enforced as a guardrail (§4). | decided 2026-09-01 | — |
| D8 | Cash leg | idle cash / SGOV / broker interest | **SGOV.** The harness accrues T-bill return on the uninvested weight; SGOV (0–3 mo T-bills, ~$100/share) matches that almost exactly and is broker-agnostic. Exempt from any per-position cap. | decided 2026-09-01 | — |
| D9 | Equity vehicle | SPY fractional / SPY whole / SPLG whole | **SPLG, whole shares.** SPY at ~$600 makes one share ~6% of a $10k book, outside the null's 2% rebalance band. SPLG tracks the same index at ~$70 (0.7% granularity) with a lower expense ratio, so fractional support is irrelevant. Harness and shadow stay on SPY data; execution maps the SPY weight to SPLG. | decided 2026-09-01 | — |
| D10 | Decision → order timing | next open / same-day 15:45 / MOC | **Next session, 09:35 ET or later.** Live consumes the identical post-close decision the shadow log records, so live and shadow never diverge by construction. Overnight slippage vs the harness's close-price assumption is measured by W7, not assumed. | decided 2026-09-01 | — |
| D11 | IBKR API route | TWS API via IB Gateway / Client Portal Web API | **IB Gateway + IBC + ib_async.** The mature headless path; IBC handles restarts and the weekly re-login. Headless OAuth for the Web API is not open to standard individual accounts. | decided 2026-09-01 | — |
| D12 | Order type | marketable limit / market / midpoint+retry | **Marketable limit, 0.5% collar** (buy at ask +0.5%, sell at bid −0.5%). Fills like a market order in normal tape, rejects on a stale or dislocated quote. Unfilled remainder cancels at window end and alerts. | decided 2026-09-01 | — |

---

## 3. Workstreams

Acceptance criteria are what "done" means. Unchecked boxes are the live TODO list.

### W1 — Broker adapter · `python/execution/broker/`
- [ ] Open the IBKR Lite cash account; enable paper trading; IB Gateway + IBC on the appliance; credentials via environment only
- [x] Adapter interface: `account()`, `submit(order)`, `cancel(id)`, `open_orders()` (`execution/broker/`)
- [x] Local cash-account paper broker behind that interface, git-tracked journal (`data/execution/paper/`)
- [x] Paper-mode smoke: whole-share fills, rejects, settlement, idempotent resubmit (18 tests)
- [ ] IBKR adapter behind the same interface; broker names appear nowhere else in the tree
- [ ] Symbol allowlist `{SPLG, SGOV}` and the SPY → SPLG weight mapping live in execution config, not in the adapter
- **Accepts when:** a target-weight dict round-trips to fills on paper, and swapping the
  adapter would touch exactly one module.

### W2 — Guardrail layer · `python/execution/guardrails.py`
Model-external by construction (§1.11). The allocator cannot import, configure, or bypass it.
- [ ] Freeze initial values (§4 below) before the first live order
- [ ] Pure validator: `(current, target, account) -> approved_orders | Rejection`
- [ ] Reject, never silently clamp — a rejection is an alert, not a shrug
- [ ] Property tests: no output ever exceeds gross, per-position, turnover, or order-size caps
- [ ] Charter exclusions enforced here too: no leverage, no shorts, no crypto
- **Accepts when:** a deliberately hostile target-weight vector cannot produce a violating
  order in tests, and the guardrail module has no dependency on `spine` or any race package.

### W3 — Order lifecycle + reconciliation
- [x] Idempotent submission keyed by (decision_date, strategy, symbol, side) — a retry never double-trades (paper)
- [ ] Fill polling, partial-fill handling, end-of-day position reconciliation
- [ ] Divergence alarm: broker positions vs intended weights beyond tolerance
- [ ] Append-only execution journal, git-tracked, same contract as `data/shadow/`
- **Accepts when:** killing the process mid-cycle and restarting produces no duplicate order.

### W4 — Kill switch + notifications
- [ ] Halt flag checked before every submission; set = flatten-or-hold, never new risk
- [ ] Every action notifies (§1.11): submissions, fills, rejections, halts
- [ ] Documented manual broker-side liquidation procedure
- **Accepts when:** setting the flag stops trading within one decision cycle, verified live.

### W5 — Decision → order path
- [x] `shadow` target weights feed execution unchanged; SPY → SPLG and cash → SGOV mapping in `execution/targets.py`
- [ ] Degrade-safe: stale data, upstream failure, or missing decision holds current positions
      and alerts — never liquidates, never rebalances on bad input
- [ ] Trading-hours and market-calendar checks
- **Accepts when:** a full simulated outage produces a hold plus an alert, and no orders.

### W6 — Break-glass execution path
Today the Rust bus is advisory only.
- [ ] Restart the bus on the appliance (dormant since 2026-07-31)
- [ ] Define what an alert is *permitted* to do — proposal: de-risk toward cash only, never
      add exposure, subject to the same guardrails
- [ ] Rate-limit: at most one break-glass action per day, then require manual re-arm
- **Accepts when:** a replayed historical trigger produces a bounded, guardrailed de-risk in
  paper and nothing else.

### W7 — Observability
- [ ] Daily reconciliation report: intended vs actual weights, drift, costs, turnover
- [ ] Realized-vs-backtest cost tracking (validates the harness cost model against reality)
- [ ] CI watchdog alerting on a missed appliance decision

### W8 — Evidence clocks (background, already running)
| Clock | State 2026-08-30 | Target |
|---|---|---|
| Shadow log | 21 sessions, zero gaps | 60 by 2026-10-30 |
| Tone-live | 31 covered sessions | ~120 to warm live z (~Jan 2027) |
| Per-name archive | restarted 2026-08-31 after a 28-day outage | daily, no gaps |
| Monthly re-race | last 2026-08-02 | next 2026-09-01, now includes claims/oil/fx |

---

## 4. Proposed initial guardrail values

Maximally conservative per §1.11; loosen only as the trust ladder advances, and only by
editing this table with a dated entry in the progress log.

| Guardrail | Proposed | Basis |
|---|---|---|
| Gross exposure | ≤ 100% | Charter: no leverage |
| Allowed symbols | SPLG, SGOV only | Null holds the index and T-bills; any other symbol is rejected until a promotion decision lists it |
| Per-position weight | ≤ 100% SPLG, ≤ 100% SGOV | Either leg may be the whole book |
| Daily one-way turnover | ≤ 20% | Null measures 3.0 one-way/yr; 20%/day is ~17x headroom and still bounds a runaway |
| Single order size | ≤ 25% of book | Sanity, not strategy |
| Orders per day | ≤ 10 | Bounds any loop bug |
| Price collar | ≤ 0.5% from last quote | Rejects stale-price trades; matches the D12 limit |
| Settlement (cash account) | never sell a lot bought with unsettled funds; one order batch per session | T+1 good-faith-violation rule; break-glass may only sell settled lots |
| Trading window | 09:35–15:55 ET; daily batch submitted at window open | Avoids open/close dislocation |
| Shorts / leverage / crypto | prohibited | Charter §1.5 |

Not yet frozen. Freeze before the first live order and record the freeze here.

Open before freeze: the 20%/day turnover cap and the 25% order-size cap can bind on the
null's own de-risk. A vol jump from 15% to 30% halves the target in one step, a 50% one-way
move; March 2020 did that inside a week. Measure the null's historical maximum daily one-way
turnover in the harness before freezing, and either set the cap above it or exempt
risk-reducing sells from it. A cap that blocks the de-risk is a guardrail pointed the wrong way.

---

## 5. Calendar

| When | Milestone |
|---|---|
| Sep, week 1 | D1–D12 decided (done 2026-09-01); IBKR account + paper login; IB Gateway on the appliance; W1 adapter against paper |
| Sep, week 2–3 | W2 guardrails + property tests; W3 lifecycle |
| Sep, week 4 | W4 kill switch; W5 wired end to end on paper |
| Oct, week 1 | Paper running daily on the appliance; W7 reports |
| Oct, week 2 | Freeze guardrails |
| Oct, week 3–4 | Null goes live at trust-ladder stage 3 sizing |
| 2026-11-02 | Behavior-match evaluates on schedule — a process check on a system that now exists |
| Nov+ | Leader and 8f-tone shadow against the live book; W6 break-glass |

---

## 6. Invariants for this build

1. Execution is strategy-agnostic. It consumes target weights and knows nothing about regimes.
2. Guardrails live outside the model and reject rather than clamp.
3. Every failure degrades toward less risk. No failure produces unbounded action.
4. The execution journal is append-only and point-in-time, like every other Meridian record.
5. Promotion of any candidate to the live book is a separate, pre-registered decision. Shipping
   execution does not ship a model.

---

## 7. Progress log

Newest last. One line per material change; note what moved and why.

- **2026-08-30** — Plan opened. Established that step 10 is gated on nothing and that the
  Nov 2 behavior-match gate will likely pass vacuously (0/21 divergent shadow sessions).
  Adopted null-first as the shaping decision. Per-name archive restarted after a 28-day
  outage caused by the Aug-2 commits sitting unpushed until 2026-08-30.
- **2026-08-30** — Leave-one-episode-out run on the leader. Edge survives dropping any single
  episode (min +0.072, median +0.143); the "it's just 2008" hypothesis is dead. Per-episode
  hit rate is 4/9, so the mechanism is drawdown reduction, not crisis timing. Null-first
  stands, but the leader is now a materially stronger stage-4 candidate than it was this
  morning.
- **2026-09-01** — All twelve decisions closed in one sitting (D1–D12). Cash-only is the
  shaping constraint: it eliminated Alpaca outright (margin-only) and pushed the book to an
  IBKR Lite cash account. SPLG replaces SPY as the traded vehicle so a $10k book rebalances
  in whole shares inside the 2% band; SGOV is the cash leg so live cash accrues what the
  harness assumes. Live consumes the post-close shadow decision unchanged and executes next
  session at 09:35 ET. Guardrail table gained a symbol allowlist and a cash-account
  settlement rule. Next: open the account, stand up IB Gateway on the appliance, W1.
- **2026-09-01** — Paper book built, CI-orchestrated like the races rather than on the
  appliance: `python/execution/` (frozen order types, `Broker` protocol, cash-account
  `PaperBroker` with T+1 settlement, whole-share planner, daily cycle) and
  `.github/workflows/paper-trade.yml` running at 09:40 ET against the previous close's
  `null_voltarget` shadow row. Journal is git-tracked under `data/execution/paper/`.
  Paper account opened at $10,000. Guardrails (W2) are the next layer and slot between
  `plan_orders` and `submit`.
- **2026-09-01** — Cron held. The local paper broker fills against our own cost table
  and the null has sat at 100% SPY for 22 sessions, so a simulated book would learn
  plumbing only. Code, tests, and interface committed; `paper-trade.yml` and the opened
  account stay out of history until the IBKR paper adapter produces real fills. The
  local broker's job until then is test double for the W2 guardrail property tests.
