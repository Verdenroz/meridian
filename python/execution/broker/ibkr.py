"""Interactive Brokers adapter behind the Broker protocol.

Drives an already-connected ib_async client for an IBKR Lite cash account:
whole-share DAY limit orders on SMART-routed US stocks. The request's
client_id travels as the IBKR orderRef, so a restarted process finds its own
orders in the gateway's open and completed order lists instead of placing
them twice. Fills are read from executions rather than order status, since
orders recovered after a restart carry no fills of their own.

UNVERIFIED against a live IB Gateway: every mapping here follows the
ib_async documentation and is exercised only by a fake client. Verification
against the IBKR paper account is a pending W1 checkbox.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from execution.orders import Account, Order, OrderRequest, Position, Quote, Status

STATUS: dict[str, Status] = {
    "Filled": "filled",
    "Cancelled": "cancelled",
    "ApiCancelled": "cancelled",
    "Inactive": "rejected",
}


def connect(host: str, port: int, client_id: int) -> Any:
    import ib_async

    ib = ib_async.IB()
    ib.connect(host, port, clientId=client_id)
    return ib


class IbkrBroker:
    def __init__(
        self,
        client: Any,
        account_id: str,
        journal: Callable[[dict], None],
        api: Any = None,
    ) -> None:
        self.client = client
        self.account_id = account_id
        self.journal = journal
        self.api = api if api is not None else importlib.import_module("ib_async")
        self.pending: frozenset[str] = frozenset()

    def _record(self, event: str, order: Order) -> None:
        self.journal({"kind": event, "account": self.account_id, "order": order.to_dict()})

    def account(self, quotes: dict[str, Quote]) -> Account:
        summary = {v.tag: v.value for v in self.client.accountSummary(self.account_id)}
        held = (
            Position(item.contract.symbol, int(item.position), float(item.averageCost))
            for item in self.client.portfolio(self.account_id)
            if int(item.position)
        )
        return Account(
            cash=float(summary["TotalCashValue"]),
            settled_cash=float(summary["AvailableFunds"]),
            equity=float(summary["NetLiquidation"]),
            positions=tuple(sorted(held, key=lambda p: p.symbol)),
        )

    def _mine(self, trades: Any) -> list[Any]:
        return [t for t in trades if t.order.account == self.account_id]

    def open_orders(self) -> tuple[Order, ...]:
        return tuple(self._order(t) for t in self._mine(self.client.openTrades()))

    def _known(self, client_id: str) -> Any:
        trades = list(self.client.trades()) + list(self.client.reqCompletedOrders(True))
        return next((t for t in trades if t.order.orderRef == client_id), None)

    def submit(self, request: OrderRequest, now: str) -> Order:
        existing = self._known(request.client_id)
        if existing is not None:
            return self._watch(self._order(existing))
        contract = self.api.Stock(request.symbol, "SMART", "USD")
        limit = self.api.LimitOrder(
            request.side.upper(),
            request.qty,
            request.limit,
            tif="DAY",
            orderRef=request.client_id,
            account=self.account_id,
        )
        trade = self.client.placeOrder(contract, limit)
        order = replace(self._order(trade), submitted_at=now)
        self._record("submit", order)
        return self._watch(order)

    def _watch(self, order: Order) -> Order:
        if order.status == "open":
            self.pending = self.pending | {order.id}
        return order

    def cancel(self, order_id: str, now: str, reason: str) -> Order:
        trade = next(t for t in self.client.openTrades() if _order_id(t.order) == order_id)
        cancelled = replace(self._order(self.client.cancelOrder(trade.order)), reason=reason)
        self.pending = self.pending - {order_id}
        self._record("cancel", cancelled)
        return cancelled

    def process(self) -> tuple[Order, ...]:
        fills = tuple(
            self._order(t)
            for t in self.client.trades()
            if _order_id(t.order) in self.pending and t.orderStatus.status == "Filled"
        )
        for order in fills:
            self._record("fill", order)
        self.pending = self.pending - {o.id for o in fills}
        return fills

    def _fills(self, client_id: str) -> list[Any]:
        return [f for f in self.client.fills() if f.execution.orderRef == client_id]

    def _order(self, trade: Any) -> Order:
        order, status = trade.order, STATUS.get(trade.orderStatus.status, "open")
        fill = self._fills(order.orderRef)[-1] if status == "filled" else None
        ended = status in ("cancelled", "rejected")
        return Order(
            id=_order_id(order),
            client_id=order.orderRef,
            symbol=trade.contract.symbol,
            side=order.action.lower(),
            qty=int(order.totalQuantity),
            limit=float(order.lmtPrice),
            status=status,
            submitted_at=_stamp(trade.log[:1]) or "",
            fill_price=fill.execution.avgPrice if fill is not None else None,
            filled_at=fill.time.isoformat() if fill is not None else _stamp(trade.log[-1:]),
            reason=_message(trade.log) if ended else None,
        )


def _order_id(order: Any) -> str:
    # permId is 0 until the gateway acks; orderId is 0 on orders another client placed.
    return f"IB{order.orderId}" if order.orderId else f"IBP{order.permId}"


def _stamp(entries: list[Any]) -> str | None:
    return entries[0].time.isoformat() if entries else None


def _message(entries: list[Any]) -> str | None:
    return next((e.message for e in reversed(entries) if e.message), None)
