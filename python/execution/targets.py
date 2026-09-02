"""Target weights to whole-share orders. Pure: no broker, no strategy.

The shadow log speaks in research symbols (SPY, cash). The live book trades
their executable counterparts: SPLG for the index at a share price that
keeps whole-share rounding inside the null's rebalance band, SGOV for cash
so the book earns the T-bill accrual the harness assumes.
"""

from __future__ import annotations

import math

from execution.orders import Account, OrderRequest, Quote

SYMBOL_MAP = {"SPY": "SPLG"}
CASH_SYMBOL = "SGOV"
COLLAR = 0.005


def live_targets(weights: dict[str, float], cash_weight: float) -> dict[str, float]:
    mapped = {SYMBOL_MAP.get(s, s): w for s, w in weights.items() if w > 0}
    return {**mapped, CASH_SYMBOL: mapped.get(CASH_SYMBOL, 0.0) + cash_weight}


def plan_orders(
    account: Account,
    targets: dict[str, float],
    quotes: dict[str, Quote],
    key: str,
    collar: float = COLLAR,
) -> tuple[OrderRequest, ...]:
    """Sells first so a cash account has the proceeds before its buys."""
    symbols = sorted(set(targets) | {p.symbol for p in account.positions})
    requests = [
        _order_for(symbol, account, targets.get(symbol, 0.0), quotes[symbol], key, collar)
        for symbol in symbols
    ]
    present = [r for r in requests if r is not None]
    return tuple(sorted(present, key=lambda r: r.side != "sell"))


def _order_for(
    symbol: str, account: Account, weight: float, quote: Quote, key: str, collar: float
) -> OrderRequest | None:
    held = account.qty(symbol)
    notional = account.equity * weight
    buy_limit = round(quote.ask * (1 + collar), 2)
    sell_limit = round(quote.bid * (1 - collar), 2)
    sell_floor = math.floor(notional / quote.last)
    buy_ceiling = math.floor(notional / buy_limit)
    if held > sell_floor:
        return OrderRequest(f"{key}:{symbol}:sell", symbol, "sell", held - sell_floor, sell_limit)
    if held < buy_ceiling:
        return OrderRequest(f"{key}:{symbol}:buy", symbol, "buy", buy_ceiling - held, buy_limit)
    return None
