"""Local cash-account simulation behind the Broker protocol.

Whole shares, DAY limit orders, no borrowing: a buy needs cash on hand
(settled or unsettled), a sell needs the shares. Fills happen when a quote
is processed: a buy fills at the ask once the limit reaches it, a sell at
the bid. Sale proceeds settle T+1 and are reported separately so a
guardrail can refuse to trade on unsettled money.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from execution.broker.paper_state import PaperState, PaperStore, Settlement
from execution.orders import Account, Order, OrderRequest, Position, Quote


def settlement_date(trade_date: date) -> date:
    settled = trade_date + timedelta(days=1)
    while settled.weekday() >= 5:
        settled += timedelta(days=1)
    return settled


class PaperBroker:
    def __init__(self, store: PaperStore) -> None:
        self.store = store
        self.state = store.load()

    def _commit(self, state: PaperState, event: str, order: Order | None = None) -> None:
        self.state = state
        self.store.save(state)
        row = {"kind": event, "cash": round(state.cash, 2)}
        if order is not None:
            row["order"] = order.to_dict()
        self.store.journal(row)

    def account(self, quotes: dict[str, Quote]) -> Account:
        held = sum(p.qty * quotes[p.symbol].last for p in self.state.positions)
        unsettled = sum(s.amount for s in self.state.unsettled)
        return Account(
            cash=self.state.cash,
            settled_cash=self.state.cash - unsettled,
            equity=self.state.cash + held,
            positions=self.state.positions,
        )

    def open_orders(self) -> tuple[Order, ...]:
        return tuple(o for o in self.state.orders if o.status == "open")

    def submit(self, request: OrderRequest, now: str) -> Order:
        existing = next(
            (o for o in self.state.orders if o.client_id == request.client_id), None
        )
        if existing is not None:
            return existing
        order = Order(
            id=f"P{self.state.next_seq:06d}",
            client_id=request.client_id,
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            limit=request.limit,
            status="open",
            submitted_at=now,
        )
        reason = self._rejection(order)
        if reason is not None:
            order = replace(order, status="rejected", reason=reason)
        state = replace(self.state, next_seq=self.state.next_seq + 1).with_order(order)
        self._commit(state, "submit", order)
        return order

    def _rejection(self, order: Order) -> str | None:
        if order.qty <= 0 or order.qty != int(order.qty):
            return "qty must be a positive whole number"
        if order.limit <= 0:
            return "limit must be positive"
        if order.side == "buy":
            committed = sum(
                o.qty * o.limit for o in self.open_orders() if o.side == "buy"
            )
            if order.qty * order.limit > self.state.cash - committed:
                return "insufficient cash"
            return None
        pending = sum(
            o.qty
            for o in self.open_orders()
            if o.side == "sell" and o.symbol == order.symbol
        )
        if order.qty > self.state.position(order.symbol).qty - pending:
            return "insufficient shares"
        return None

    def cancel(self, order_id: str, now: str, reason: str) -> Order:
        order = next(o for o in self.state.orders if o.id == order_id)
        if order.status != "open":
            return order
        cancelled = replace(order, status="cancelled", reason=reason, filled_at=now)
        self._commit(self.state.with_order(cancelled), "cancel", cancelled)
        return cancelled

    def process(self, quotes: dict[str, Quote], now: str, today: date) -> tuple[Order, ...]:
        fills = []
        for order in self.open_orders():
            price = _marketable_price(order, quotes[order.symbol])
            if price is None:
                continue
            filled = replace(order, status="filled", fill_price=price, filled_at=now)
            self._commit(self._apply_fill(filled, price, today), "fill", filled)
            fills.append(filled)
        return tuple(fills)

    def _apply_fill(self, order: Order, price: float, today: date) -> PaperState:
        held = self.state.position(order.symbol)
        notional = order.qty * price
        state = self.state.with_order(order)
        if order.side == "buy":
            qty = held.qty + order.qty
            avg = (held.qty * held.avg_cost + notional) / qty
            return replace(state, cash=state.cash - notional).with_position(
                Position(order.symbol, qty, avg, unsettled_funded=bool(state.unsettled))
            )
        settle = Settlement(settlement_date(today).isoformat(), notional)
        return replace(
            state, cash=state.cash + notional, unsettled=state.unsettled + (settle,)
        ).with_position(Position(order.symbol, held.qty - order.qty, held.avg_cost))

    def settle(self, today: date) -> None:
        remaining = tuple(
            s for s in self.state.unsettled if date.fromisoformat(s.settle_date) > today
        )
        if remaining == self.state.unsettled:
            return
        positions = self.state.positions
        if not remaining:
            positions = tuple(replace(p, unsettled_funded=False) for p in positions)
        self._commit(
            replace(self.state, unsettled=remaining, positions=positions), "settle"
        )


def _marketable_price(order: Order, quote: Quote) -> float | None:
    if order.side == "buy":
        return quote.ask if order.limit >= quote.ask else None
    return quote.bid if order.limit <= quote.bid else None
