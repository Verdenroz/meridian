"""Allocator: equity weight → full ETF-core book.

The equity fraction arrives already sized (vol target × regime mapping, see
AllocatorConfig). The defensive remainder is cash by default, with TLT/GLD
sleeves only while they currently behave as safe assets (negative trailing
correlation with equity). Residual weight is cash — the engine pays the T-bill
rate on it. No inverse-ETF sleeve in v0.
"""

from __future__ import annotations

TARGET_VOL = 0.15
# Fraction of the defensive budget each qualifying safe asset receives.
SAFE_SLEEVE_FRACTION = 0.20


def allocate(
    equity_weight: float,
    safe_assets: dict[str, bool],
    equity_symbol: str = "SPY",
) -> dict[str, float]:
    equity = max(0.0, min(1.0, equity_weight))
    defense = 1.0 - equity
    weights = {equity_symbol: equity}
    for symbol, is_safe in safe_assets.items():
        if is_safe:
            weights[symbol] = SAFE_SLEEVE_FRACTION * defense
    return weights
