"""Transaction cost model: one-way cost in bps of traded notional.

Commission-free brokerage assumed; costs are half-spread plus slippage,
deliberately on the conservative side. Unknown instruments get a punitive
default so forgetting to configure a name can only hurt the strategy, never
flatter it.
"""

from __future__ import annotations

# One-way bps: half-spread + slippage for liquid ETFs at daily-close scale.
DEFAULT_COST_BPS: dict[str, float] = {
    "SPY": 1.0,
    "SPLG": 1.5,
    "SGOV": 1.0,
    "TLT": 1.5,
    "GLD": 1.5,
    "BIL": 1.0,
    "SH": 3.0,
}
UNKNOWN_COST_BPS = 5.0


def one_way_cost(symbol: str, cost_bps: dict[str, float] | None = None) -> float:
    table = DEFAULT_COST_BPS if cost_bps is None else cost_bps
    return table.get(symbol, UNKNOWN_COST_BPS) / 1e4
