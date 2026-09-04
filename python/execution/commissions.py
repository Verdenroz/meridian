"""IBKR Pro Tiered commission on US stocks: a per-share rate with a per-order
floor, capped at one percent of trade value. Exchange and regulatory
pass-throughs are cents on this book's order sizes and are left out; the
broker's own commission report replaces this model on live fills."""

from __future__ import annotations

TIERED_PER_SHARE = 0.0035
TIERED_MIN_PER_ORDER = 0.35
TIERED_MAX_OF_VALUE = 0.01


def tiered_commission(qty: int, price: float) -> float:
    per_share = max(qty * TIERED_PER_SHARE, TIERED_MIN_PER_ORDER)
    return round(min(per_share, TIERED_MAX_OF_VALUE * qty * price), 4)
