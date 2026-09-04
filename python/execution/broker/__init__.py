"""The one interface every broker adapter implements.

Broker names appear only inside this package; the cycle and the guardrails
see this protocol and nothing else.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from execution.orders import Account, Order, OrderRequest, Quote


class Broker(Protocol):
    def account(self, quotes: dict[str, Quote]) -> Account: ...

    def settle(self, today: date) -> None: ...

    def submit(self, request: OrderRequest, now: str) -> Order: ...

    def process(self, quotes: dict[str, Quote], now: str, today: date) -> tuple[Order, ...]: ...

    def cancel(self, order_id: str, now: str, reason: str) -> Order: ...

    def open_orders(self) -> tuple[Order, ...]: ...
