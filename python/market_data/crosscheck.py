"""Cross-check upstream adjClose against the dividends endpoint, step by step.

Upstream closes and dividend amounts are already split-adjusted, so each
back-adjustment step in adjClose/close should equal (1 - dividend/prev_close)
on ex-div days and 1.0 elsewhere. Comparing *steps* localizes errors: an
isolated endpoint gap (observed: KO missing its 2001-09-10 payout; MSFT
shipping a duplicated row) is recorded but tolerated, while systematic
inconsistency or steps no dividend could explain fail loudly. adjClose was
null upstream until 2026-07-30, so regression stays a live risk.
"""

from __future__ import annotations

from market_data.series import utc_date

# A step differing from its dividend-implied value by more than this is a
# mismatch (dividend rounding sits orders of magnitude below).
STEP_TOLERANCE = 2e-4
# An implied one-day payout above this fraction of price is not a dividend.
MAX_PLAUSIBLE_YIELD = 0.25
# Systematic failure: too many mismatched steps, in fraction or count.
MAX_MISMATCH_FRACTION = 0.05
MAX_MISMATCHES = 5


class CrossCheckError(Exception):
    """Upstream adjClose is missing or inconsistent with dividend history."""


def dividends_by_date(dividends: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    seen: set[tuple[int, float]] = set()
    for d in dividends:
        ts, amount = d.get("timestamp"), d.get("amount")
        if ts is None or not amount or amount <= 0:
            continue
        # Upstream sometimes duplicates a payout row verbatim (seen: MSFT
        # 2026-02-19). Identical (ts, amount) pairs are data bugs, not two
        # payouts; distinct same-day payouts (regular + special) differ.
        if (ts, float(amount)) in seen:
            continue
        seen.add((ts, float(amount)))
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
    tolerance: float = STEP_TOLERANCE,
) -> dict:
    """Raise CrossCheckError on missing adjClose or step-level inconsistency;
    return a report including any isolated, tolerated mismatches."""
    if not rows:
        raise CrossCheckError(f"{symbol}: no rows to verify")
    missing = sum(1 for r in rows if r["adj_close"] is None)
    if missing:
        raise CrossCheckError(
            f"{symbol}: adjClose missing on {missing}/{len(rows)} rows — "
            "possible upstream regression (field was null until 2026-07-30)"
        )
    last = rows[-1]
    if abs(last["adj_close"] / last["close"] - 1.0) > 1e-3:
        raise CrossCheckError(
            f"{symbol}: adjClose does not converge to close on the latest bar"
        )

    div_by_date = dividends_by_date(dividends)
    ratios = [r["adj_close"] / r["close"] for r in rows]
    event_days = 0
    mismatches: list[dict] = []
    for i in range(1, len(rows)):
        step = ratios[i - 1] / ratios[i]
        prev_close = rows[i - 1]["close"]
        dividend = div_by_date.get(rows[i]["date"], 0.0)
        expected = 1.0 - dividend / prev_close if dividend else 1.0
        if dividend or abs(step - 1.0) > tolerance:
            event_days += 1
        if abs(step - expected) <= tolerance:
            continue
        implied = (1.0 - step) * prev_close
        if implied < -0.01 or implied > MAX_PLAUSIBLE_YIELD * prev_close:
            raise CrossCheckError(
                f"{symbol}: adjustment step on {rows[i]['date']} implies "
                f"payout {implied:.4f} vs close {prev_close:.2f} — not a "
                "plausible dividend; adjClose corruption"
            )
        mismatches.append(
            {
                "date": rows[i]["date"],
                "implied_dividend": round(implied, 6),
                "endpoint_dividend": round(dividend, 6),
            }
        )
    if mismatches and (
        len(mismatches) > MAX_MISMATCHES
        or len(mismatches) / max(event_days, 1) > MAX_MISMATCH_FRACTION
    ):
        sample = ", ".join(m["date"] for m in mismatches[:5])
        raise CrossCheckError(
            f"{symbol}: {len(mismatches)}/{event_days} adjustment steps "
            f"inconsistent with dividend history (e.g. {sample})"
        )
    return {
        "symbol": symbol,
        "rows": len(rows),
        "adjustment_events": event_days,
        "mismatches": mismatches,
    }
