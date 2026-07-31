"""Behavior-match pre-registration for the shadow-trading stage.

FROZEN 2026-07-31 — committed before any evaluation reads
`data/shadow/decisions.jsonl`. Editing any threshold below after this date
is a new pre-registration (new module, new spec note), never an amendment
in place. The evaluator refuses to run before `earliest_evaluation` and
there is deliberately no override flag: peeking at partial windows is
exactly the contamination this freeze exists to prevent.

What is being tested
--------------------
Whether the live daily pipeline behaves the way the backtests said it
would. This is NOT a performance gate: three months of returns is noise,
and no Sharpe, return, or drawdown number participates in any verdict
below. A PASS satisfies clause (b) of the amended charter bar's
prospective requirement (spec §1.3) for that strategy; shipping still
requires clause (a) — the strategy's own frozen race pre-registration
continuing to pass as out-of-sample data accrues.

Evaluation window
-----------------
Decisions with `decision_date` in [window_start, window_end] — three
calendar months from the first logged decision. Evaluation may run on or
after `earliest_evaluation` (the first NYSE session after the window
closes). The same frozen criteria may be applied to any later same-length
window (e.g. after a failed contender is retired, renamed, and its clock
restarted); the parameters never change.

Per-strategy criteria — PASS requires all three
-----------------------------------------------
C1  Coverage. At least `min_logged_days` logged decision dates within the
    window's trading days (taken from the panel calendar), no missing run
    longer than `max_gap_sessions`, and every in-window row carrying
    `job_version == log_job_version`. A dead cron is an operational fail.

C2  Fidelity. Every logged row is re-derived at evaluation time by
    re-running the exact shadow procedure (fresh contender fit at that
    date's HistoryView over the same store). A row matches when every
    per-symbol weight differs by ≤ `weight_tol`. PASS requires a match
    rate ≥ `min_match_rate` AND no row with any per-symbol difference
    > `weight_tol_hard`. Mismatches caused by upstream data revisions
    still count against the budget — sensitivity to revisions is real
    operational risk, and there are no discretionary exclusions.

C3  Behavior in-distribution. Statistics of the logged window — mean
    gross exposure and annualized one-way decision turnover for every
    strategy, plus brake-active frequency for credit_brake — must fall
    inside the [band_lo_pct, band_hi_pct] percentile band of the same
    statistic over every same-length stride-1 rolling window of a
    shadow-equivalent reference replay (SPY inception + warmup through
    `reference_end`, data the shadow window never touches). C3 is a
    distribution-shift alarm: it can fail honestly if the live quarter is
    unlike anything in the reference period — that outcome still blocks
    promotion until a later window passes.

Reference replay (C3) — frozen procedure
----------------------------------------
Daily decision semantics identical to the shadow job: fresh band state
every day, targets emitted every day. The spine's tuner refit runs every
`reference_fit_stride` sessions instead of daily — a compute concession,
recorded here, that leaves the daily banding and regime-filter semantics
untouched. The replay starts `reference_warmup_sessions` after SPY
history begins and ends at `reference_end` (the last session before the
first logged decision). Percentiles use numpy linear interpolation.

Consequences
------------
PASS → §1.3 clause (b) is satisfied for that strategy.
FAIL → the contender is retired and renamed (per shadow/contenders.py)
once the root cause is fixed, and its clock restarts. The decisions log
itself is never edited or regenerated.
"""

from __future__ import annotations

CRITERIA: dict[str, str | int | float] = {
    "log_job_version": "shadow-v1",
    "window_start": "2026-07-31",
    "window_end": "2026-10-30",
    "earliest_evaluation": "2026-11-02",
    "min_logged_days": 60,
    "max_gap_sessions": 5,
    "weight_tol": 0.01,
    "weight_tol_hard": 0.05,
    "min_match_rate": 0.95,
    "reference_end": "2026-07-30",
    "reference_fit_stride": 21,
    "reference_warmup_sessions": 252,
    "band_lo_pct": 5.0,
    "band_hi_pct": 95.0,
}
