"""Cross-check upstream adjClose against a local dividends-based reconstruction.

Upstream closes and dividend amounts are already split-adjusted (verified
empirically: max relative divergence ~1e-6 over SPY/AAPL full history), so the
reconstruction applies dividend back-adjustment only. Splits are stored for
audit but do not enter this math. adjClose was null upstream until 2026-07-30;
regression there is a live risk, hence the loud failure mode.
"""

from __future__ import annotations

from market_data.series import utc_date

REL_TOLERANCE = 1e-4


class CrossCheckError(Exception):
    """Upstream adjClose is missing or diverges from local reconstruction."""


def dividends_by_date(dividends: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dividends:
        ts, amount = d.get("timestamp"), d.get("amount")
        if ts is None or not amount or amount <= 0:
            continue
        date = utc_date(ts)
        out[date] = out.get(date, 0.0) + float(amount)
    return out


def reconstruct_adj_close(rows: list[dict], div_by_date: dict[str, float]) -> list[float]:
    """Backward pass: the latest close is its own adjusted close; each ex-div
    date scales all earlier closes by (1 - dividend / previous close)."""
    n = len(rows)
    factors = [1.0] * n
    factor = 1.0
    for i in range(n - 1, 0, -1):
        dividend = div_by_date.get(rows[i]["date"], 0.0)
        if dividend:
            prev_close = rows[i - 1]["close"]
            if dividend >= prev_close:
                raise CrossCheckError(
                    f"dividend {dividend} >= previous close {prev_close} "
                    f"on {rows[i]['date']} — corrupt upstream data"
                )
            factor *= 1.0 - dividend / prev_close
        factors[i - 1] = factor
    return [row["close"] * factors[i] for i, row in enumerate(rows)]


def verify_adj_close(
    symbol: str,
    rows: list[dict],
    dividends: list[dict],
    tolerance: float = REL_TOLERANCE,
) -> dict:
    """Raise CrossCheckError on missing adjClose or divergence; return a report."""
    if not rows:
        raise CrossCheckError(f"{symbol}: no rows to verify")
    missing = sum(1 for r in rows if r["adj_close"] is None)
    if missing:
        raise CrossCheckError(
            f"{symbol}: adjClose missing on {missing}/{len(rows)} rows — "
            "possible upstream regression (field was null until 2026-07-30)"
        )
    div_by_date = dividends_by_date(dividends)
    recon = reconstruct_adj_close(rows, div_by_date)
    worst_err, worst_date = 0.0, rows[0]["date"]
    for row, expected in zip(rows, recon):
        err = abs(expected - row["adj_close"]) / max(abs(row["adj_close"]), 1e-12)
        if err > worst_err:
            worst_err, worst_date = err, row["date"]
    if worst_err > tolerance:
        raise CrossCheckError(
            f"{symbol}: adjClose diverges from dividends-based reconstruction "
            f"(max rel err {worst_err:.2e} on {worst_date}, tolerance {tolerance:.0e})"
        )
    row_dates = {r["date"] for r in rows}
    applied = sum(1 for date in div_by_date if date in row_dates)
    return {
        "symbol": symbol,
        "rows": len(rows),
        "dividends_applied": applied,
        "max_rel_err": worst_err,
        "worst_date": worst_date,
    }
