"""Model-external guardrails: the last check before any order leaves.

A pure validator over (account, planned orders, quotes). It rejects, never
clamps: one violating order rejects the whole batch, and a rejection is an
alert. It imports nothing from spine, shadow, or any race package, so no
strategy can configure or bypass it.

Risk-reducing orders (selling the index leg, buying the cash leg) carry no
turnover cap: the null's own defense moved 38% of the book in one session
in 1997, and a cap that blocks the de-risk is a guardrail pointed the wrong
way. Risk-adding orders are capped per session; the null's largest re-risk
day outside its initial entry is 5%.
"""

from __future__ import annotations

from dataclasses import dataclass

from execution.orders import Account, OrderRequest, Quote

LIMITS: dict = {
    "allowed_symbols": frozenset({"SPYM", "SGOV"}),
    "cash_symbol": "SGOV",
    "max_gross": 1.0,
    "max_risk_add": 0.25,
    "max_orders": 10,
    "max_collar": 0.005,
    "gross_tolerance": 0.01,
}


@dataclass(frozen=True)
class Rejection:
    reason: str
    order: OrderRequest | None = None


def validate(
    account: Account,
    requests: tuple[OrderRequest, ...],
    quotes: dict[str, Quote],
    limits: dict = LIMITS,
    allow_risk_adding: bool = True,
) -> tuple[OrderRequest, ...] | Rejection:
    if len(requests) > limits["max_orders"]:
        return Rejection(f"{len(requests)} orders exceeds {limits['max_orders']}")
    for request in requests:
        reason = _order_violation(request, account, quotes, limits, allow_risk_adding)
        if reason is not None:
            return Rejection(reason, request)
    risk_add = sum(
        r.qty * r.limit for r in requests if _risk_adding(r, limits["cash_symbol"])
    )
    if account.equity > 0 and risk_add / account.equity > limits["max_risk_add"]:
        return Rejection(
            f"risk-adding {risk_add / account.equity:.3f} of equity exceeds "
            f"{limits['max_risk_add']}"
        )
    gross, cash = _after(account, requests, quotes)
    if cash < 0:
        return Rejection(f"batch would leave cash at {cash:.2f}")
    if account.equity > 0 and gross / account.equity > (
        limits["max_gross"] + limits["gross_tolerance"]
    ):
        return Rejection(f"gross {gross / account.equity:.3f} exceeds {limits['max_gross']}")
    return requests


def _risk_adding(request: OrderRequest, cash_symbol: str) -> bool:
    return request.side == "buy" and request.symbol != cash_symbol


def _order_violation(
    request: OrderRequest,
    account: Account,
    quotes: dict[str, Quote],
    limits: dict,
    allow_risk_adding: bool,
) -> str | None:
    if request.symbol not in limits["allowed_symbols"]:
        return f"{request.symbol} is not an allowed symbol"
    if request.side not in ("buy", "sell"):
        return f"{request.side} is not a side"
    if request.qty <= 0 or request.qty != int(request.qty):
        return f"{request.symbol} qty {request.qty} is not a positive whole number"
    if request.limit <= 0:
        return f"{request.symbol} limit {request.limit} is not positive"
    quote = quotes[request.symbol]
    reference = quote.ask if request.side == "buy" else quote.bid
    if abs(request.limit - reference) / reference > limits["max_collar"]:
        return f"{request.symbol} limit {request.limit} outside collar of {reference}"
    if request.side == "sell":
        position = next((p for p in account.positions if p.symbol == request.symbol), None)
        if position is None or request.qty > position.qty:
            return f"{request.symbol} sell {request.qty} exceeds holding"
        if position.unsettled_funded:
            return f"{request.symbol} was bought with unsettled funds"
        return None
    risk_adding = _risk_adding(request, limits["cash_symbol"])
    if risk_adding and not allow_risk_adding:
        return f"{request.symbol} buy adds risk while risk-adding is disabled"
    if risk_adding and account.equity > 0:
        share = request.qty * request.limit / account.equity
        if share > limits["max_risk_add"]:
            return f"{request.symbol} buy {share:.3f} of equity exceeds {limits['max_risk_add']}"
    return None


def _after(
    account: Account, requests: tuple[OrderRequest, ...], quotes: dict[str, Quote]
) -> tuple[float, float]:
    held = {p.symbol: p.qty for p in account.positions}
    cash = account.cash
    for r in requests:
        sign = 1 if r.side == "buy" else -1
        held[r.symbol] = held.get(r.symbol, 0) + sign * r.qty
        cash -= sign * r.qty * r.limit
    gross = sum(qty * quotes[symbol].last for symbol, qty in held.items())
    return gross, cash
