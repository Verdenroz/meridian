"""Order, fill, position, and account records. All immutable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Side = Literal["buy", "sell"]
Status = Literal["open", "filled", "cancelled", "rejected"]


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    market_state: str
    fetched_at: str
    prev_close: float | None = None


@dataclass(frozen=True)
class OrderRequest:
    client_id: str
    symbol: str
    side: Side
    qty: int
    limit: float


@dataclass(frozen=True)
class Order:
    id: str
    client_id: str
    symbol: str
    side: Side
    qty: int
    limit: float
    status: Status
    submitted_at: str
    fill_price: float | None = None
    filled_at: str | None = None
    reason: str | None = None
    commission: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    avg_cost: float
    unsettled_funded: bool = False


@dataclass(frozen=True)
class Account:
    cash: float
    settled_cash: float
    equity: float
    positions: tuple[Position, ...]

    def qty(self, symbol: str) -> int:
        return next((p.qty for p in self.positions if p.symbol == symbol), 0)
