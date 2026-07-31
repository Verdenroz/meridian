# Meridian

Meridian is a **regime-adaptive risk engine that manages real money**. It forecasts risk —
volatility, correlation, market regime — and allocates an ETF core plus a small risk-weighted
stock sleeve accordingly: near-full equity exposure in benign regimes, cash-heavy defense in
stress. It does **not** forecast returns (measured IC ≈ 0; a settled decision).

The bar: **beat SPY plus a simple volatility-target rule, after costs, out-of-sample,
walk-forward** — the null baseline is implemented first, in the same harness, and raced
honestly. Below that bar the system deserves zero dollars.

[`meridian-spec.md`](meridian-spec.md) is the canonical spec; its Charter (§1) is
authoritative over everything else in this repository.

## Principles

- **Risk, not returns.** No component predicts returns.
- **Challengers, never incumbents.** A learned component ships only after beating its
  classical counterpart in the walk-forward harness. The classical spine runs production.
- **Point-in-time honesty.** Text/event features key on `fetched_at` — when Meridian could
  have known — never publisher timestamps. The news archive is append-only.
- **Guardrails outside the model.** Position, turnover, and exposure limits live in code the
  model cannot touch, with a kill switch.
- **Excluded:** crypto, margin/leverage, short selling (inverse-ETF hedges allowed).

## Layout

```text
archive/feeds/        append-only news/feeds record (git-tracked; JSONL is truth)
python/feeds_archive/ archiver: fetch / loop / build-db / status (stdlib-only)
python/tests/         invariant tests (point-in-time, dedupe, idempotency)
src/                  Rust DataBus — being repurposed as the live market/event poller
legacy/               archived v4 model/training stack — reference only, do not extend
.github/workflows/    archive-feeds.yml: 10-minute CI cron so collection never gaps
```

Data comes from the unmodified upstream [finance-query](https://finance-query.com) v2 API.
Total-return series are constructed locally from chart + dividends + splits (Appendix A of
the spec records verified availability). `data/`, `checkpoints/`, `reports/`, `logs/` are
derived and gitignored.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check python && mypy python && pytest python/tests

PYTHONPATH=python python -m feeds_archive fetch     # one-shot poll (what CI runs)
PYTHONPATH=python python -m feeds_archive build-db  # JSONL -> SQLite index
PYTHONPATH=python python -m feeds_archive status    # archive freshness
```

## Build order

Per spec §10: archive live → data layer → **null baseline + walk-forward harness** →
classical spine → backtest vs. null → event study (gates all text/LLM work) → LLM event
pipeline → episodic memory → paper trading → execution with guardrails.
