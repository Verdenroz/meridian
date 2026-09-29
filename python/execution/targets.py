"""Target weights to whole-share orders. Pure: no broker, no strategy.

The shadow log speaks in research symbols (SPY, cash). The live book trades
their executable counterparts: SPYM (formerly SPLG) for the index at a share
price that keeps whole-share rounding inside the null's rebalance band, SGOV
for cash so the book earns the T-bill accrual the harness assumes.
"""

from __future__ import annotations

import math
from dataclasses import replace

from execution.guardrails import LIMITS
from execution.orders import Account, OrderRequest, Quote

SYMBOL_MAP = {"SPY": "SPYM"}
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
    """Sells first so a cash account has the proceeds before its buys.
    Risk-adding buys are ramped at the guardrail cap per session, so an entry
    from all-cash takes four sessions instead of one rejected order. Buys are
    trimmed to the cash the batch has, so a misvalued equity cannot size a
    batch the guardrail must reject."""
    symbols = sorted(set(targets) | {p.symbol for p in account.positions})
    requests = [
        _order_for(symbol, account, targets.get(symbol, 0.0), quotes[symbol], key, collar)
        for symbol in symbols
    ]
    present = sorted((r for r in requests if r is not None), key=lambda r: r.side != "sell")
    return _within_cash(present, account.cash)


def _within_cash(requests: list[OrderRequest], cash: float) -> tuple[OrderRequest, ...]:
    budget = cash + sum(r.qty * r.limit for r in requests if r.side == "sell")
    kept: list[OrderRequest] = []
    for request in requests:
        if request.side == "sell":
            kept.append(request)
            continue
        qty = min(request.qty, math.floor(budget / request.limit))
        if qty > 0:
            kept.append(replace(request, qty=qty))
            budget -= qty * request.limit
    return tuple(kept)


def _order_for(
    symbol: str, account: Account, weight: float, quote: Quote, key: str, collar: float
) -> OrderRequest | None:
    held = account.qty(symbol)
    notional = account.equity * weight
    buy_limit = math.floor(quote.ask * (1 + collar) * 100) / 100
    sell_limit = math.ceil(quote.bid * (1 - collar) * 100) / 100
    sell_floor = math.floor(notional / quote.last)
    buy_ceiling = math.floor(notional / buy_limit)
    if held > sell_floor:
        return OrderRequest(f"{key}:{symbol}:sell", symbol, "sell", held - sell_floor, sell_limit)
    if held < buy_ceiling:
        qty = buy_ceiling - held
        if symbol != LIMITS["cash_symbol"]:
            ramp = math.floor(LIMITS["max_risk_add"] * account.equity / buy_limit)
            qty = min(qty, ramp)
        return OrderRequest(f"{key}:{symbol}:buy", symbol, "buy", qty, buy_limit)
    return None
