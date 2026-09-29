"""Intraday day-trader challenger (pre-registration).

Registered 2026-09-29; frozen from the registering commit on. Sessions from
2026-09-30 are out of sample. Nothing here touches the shadow
log, the allocator, or the IBKR null book.

Motivation. The daily engine forecasts risk, not returns, and trades a few
times a month. This registers the one intraday return hypothesis the
2026-09-29 feasibility probe left standing, as a challenger to the charter's
no-return-prediction thesis. The probe already examined (and is therefore not
evidence for anything below): first-hour vs last-half-hour momentum on
2024-09..2026-09 hourly SPY/QQQ/IWM (corr ~0, loses after 2 bp); daily
overnight/intraday decomposition and gap buckets on SPY/XLK 2000-2026 (gap
effects flip sign across symbols). No archived 1m bar was inspected before
registration.

Hypothesis. In calm regimes, a large intraday stretch of SPY away from its
session VWAP partially reverts before the close. Stress regimes are excluded
because stretches there are trend, not noise.

Strategy (all constants in GATE):
- Regime filter: trade a session only if, at the prior close, the filtered
  posterior of the lowest-variance state of the spine7f_turn_3state HMM
  exceeds ``calm_posterior``. Recomputed causally; never read from the shadow
  log.
- Signal: session VWAP from 1m bars up to and including minute t;
  sigma = std of 1m log returns over the prior ``sigma_sessions`` archived
  sessions; band = ``k`` * sigma * sqrt(minutes since 09:30).
- Entry, once per session (cash-account settlement allows one full-size
  round trip), inside [``entry_start``, ``entry_end``]: SPY close <= VWAP - band
  buys SPY; SPY close >= VWAP + band buys SH. Fill at the next bar's open.
- Exit: first SPY close back through VWAP, else ``flat_by``. No stop.
- Size: the whole book in whole shares.

Simulation. Decisions see bars through t only; fills at bar t+1 open plus the
harness one-way cost (SPY 1.0 bp, SH 3.0 bp) plus IBKR tiered commission,
with a stress run at $1.00/order (the observed paper fills). Local book of
``book`` dollars; journal append-only under data/execution/daytrade/.

Controls (same engine, same sessions, same costs):
1. Random entry: for every real trade, a seeded uniform entry minute in the
   entry window, same side, same holding minutes (truncated at flat_by).
2. SPY open-to-close hold on the same sessions.

Forward clock. Every session from 2026-09-30 on is out of sample, replayed
post-close from archive/bars/. Sessions archived before registration are
in-sample smoke tests, excluded from the gate. One descriptive backtest runs
once the archive holds ``backtest_sessions`` sessions; it can kill the
hypothesis early, never promote it.

Gate, judged once at ``gate_trades`` forward trades:
PASS iff (a) mean net bps per trade minus the random control's, bootstrapped
over trades, has P(<=0) < ``p_max``; and (b) daily book P&L beats SPY
open-to-close on Sharpe via harness.stats.sharpe_diff_bootstrap with the
registered block length, resamples, and seed, with P(<=0) < ``p_max``.
FAIL retires the hypothesis. Interim reports are descriptive; no retuning.
PASS earns an IBKR-paper stage, not capital.
"""

GATE: dict = {
    "calm_posterior": 0.5,
    "sigma_sessions": 20,
    "k": 2.0,
    "entry_start": "10:00",
    "entry_end": "15:00",
    "flat_by": "15:55",
    "book": 10_000.0,
    "stress_commission": 1.00,
    "backtest_sessions": 120,
    "gate_trades": 500,
    "p_max": 0.10,
    "trade_resamples": 10_000,
    "block_len": 5,
    "n_resamples": 5_000,
    "seed": 20260929,
}
