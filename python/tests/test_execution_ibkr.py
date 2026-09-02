"""Mapping tests for the IBKR adapter against a fake ib_async client:
order construction, orderRef idempotency, cancel, status and account mapping."""

import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import pytest
from execution.broker.ibkr import IbkrBroker
from execution.orders import OrderRequest

NOW = "2026-09-02T13:40:00+00:00"
T0 = datetime(2026, 9, 2, 13, 39, 55, tzinfo=UTC)
T1 = datetime(2026, 9, 2, 13, 41, 0, tzinfo=UTC)
ACCOUNT = "DU1"
DONE = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


@dataclass(frozen=True)
class FakeContract:
    symbol: str
    exchange: str
    currency: str


@dataclass(frozen=True)
class FakeOrder:
    action: str
    totalQuantity: float
    lmtPrice: float
    tif: str = ""
    orderRef: str = ""
    account: str = ""
    orderId: int = 0
    permId: int = 0


@dataclass(frozen=True)
class FakeStatus:
    status: str


@dataclass(frozen=True)
class FakeLogEntry:
    time: datetime
    status: str
    message: str = ""


@dataclass(frozen=True)
class FakeTrade:
    contract: FakeContract
    order: FakeOrder
    orderStatus: FakeStatus
    fills: list = field(default_factory=list)
    log: list = field(default_factory=list)


@dataclass(frozen=True)
class FakeExecution:
    orderRef: str
    avgPrice: float


@dataclass(frozen=True)
class FakeFill:
    execution: FakeExecution
    time: datetime


@dataclass(frozen=True)
class FakeValue:
    tag: str
    value: str


@dataclass(frozen=True)
class FakeItem:
    contract: FakeContract
    position: float
    averageCost: float


class FakeApi:
    Stock = FakeContract

    @staticmethod
    def LimitOrder(action: str, qty: float, limit: float, **kw: Any) -> FakeOrder:
        return FakeOrder(action, qty, limit, **kw)


class FakeIB:
    def __init__(
        self,
        trades: tuple[FakeTrade, ...] = (),
        completed: tuple[FakeTrade, ...] = (),
        fills: tuple[FakeFill, ...] = (),
        summary: tuple[FakeValue, ...] = (),
        portfolio: tuple[FakeItem, ...] = (),
    ) -> None:
        self._trades = list(trades)
        self._completed = list(completed)
        self._fills = list(fills)
        self._summary = list(summary)
        self._portfolio = list(portfolio)
        self.placed: list[tuple[FakeContract, FakeOrder]] = []
        self.cancelled: list[FakeOrder] = []

    def placeOrder(self, contract: FakeContract, order: FakeOrder) -> FakeTrade:
        self.placed.append((contract, order))
        numbered = replace(order, orderId=len(self.placed))
        trade = FakeTrade(contract, numbered, FakeStatus("PreSubmitted"), [], [FakeLogEntry(T0, "PendingSubmit")])
        self._trades = self._trades + [trade]
        return trade

    def cancelOrder(self, order: FakeOrder) -> FakeTrade:
        self.cancelled.append(order)
        return self.set_status(order.orderRef, "Cancelled", "cancelled by client")

    def set_status(self, client_id: str, status: str, message: str = "") -> FakeTrade:
        trade = next(t for t in self._trades if t.order.orderRef == client_id)
        updated = replace(trade, orderStatus=FakeStatus(status), log=trade.log + [FakeLogEntry(T1, status, message)])
        self._trades = [updated if t is trade else t for t in self._trades]
        return updated

    def trades(self) -> list[FakeTrade]:
        return list(self._trades)

    def openTrades(self) -> list[FakeTrade]:
        return [t for t in self._trades if t.orderStatus.status not in DONE]

    def reqCompletedOrders(self, apiOnly: bool) -> list[FakeTrade]:
        return list(self._completed)

    def fills(self) -> list[FakeFill]:
        return list(self._fills)

    def accountSummary(self, account: str) -> list[FakeValue]:
        return list(self._summary)

    def portfolio(self, account: str) -> list[FakeItem]:
        return list(self._portfolio)


BUY = OrderRequest("k:SPLG:buy", "SPLG", "buy", 10, 80.5)


def _broker(fake: FakeIB) -> tuple[IbkrBroker, list[dict]]:
    rows: list[dict] = []
    return IbkrBroker(fake, ACCOUNT, rows.append, FakeApi), rows


class TestSubmit:
    def test_builds_a_smart_usd_day_limit_carrying_the_client_id(self) -> None:
        fake = FakeIB()
        broker, rows = _broker(fake)
        order = broker.submit(BUY, NOW)
        ((contract, placed),) = fake.placed
        assert contract == FakeContract("SPLG", "SMART", "USD")
        assert placed == FakeOrder("BUY", 10, 80.5, "DAY", "k:SPLG:buy", ACCOUNT)
        assert (order.id, order.client_id, order.side, order.status) == ("IB1", "k:SPLG:buy", "buy", "open")
        assert (order.qty, order.limit, order.submitted_at) == (10, 80.5, NOW)
        assert [r["kind"] for r in rows] == ["submit"]
        assert rows[0]["order"] == order.to_dict()

    def test_resubmit_returns_the_known_order_without_placing_again(self) -> None:
        fake = FakeIB()
        broker, _ = _broker(fake)
        first = broker.submit(BUY, NOW)
        again = broker.submit(BUY, NOW)
        restarted, rows = _broker(fake)
        after_restart = restarted.submit(BUY, NOW)
        assert again.id == after_restart.id == first.id
        assert len(fake.placed) == 1
        assert rows == []

    def test_completed_order_from_a_previous_session_is_found_by_order_ref(self) -> None:
        done = FakeOrder("BUY", 10, 80.5, "DAY", "k:SPLG:buy", ACCOUNT, permId=777)
        fake = FakeIB(
            completed=(FakeTrade(FakeContract("SPLG", "SMART", "USD"), done, FakeStatus("Filled")),),
            fills=(FakeFill(FakeExecution("k:SPLG:buy", 80.01), T1),),
        )
        broker, rows = _broker(fake)
        order = broker.submit(BUY, NOW)
        assert (order.id, order.status, order.fill_price) == ("IBP777", "filled", 80.01)
        assert order.filled_at == T1.isoformat()
        assert fake.placed == [] and rows == []


class TestCancel:
    def test_cancel_goes_through_the_client_and_journals(self) -> None:
        fake = FakeIB()
        broker, rows = _broker(fake)
        order = broker.submit(BUY, NOW)
        cancelled = broker.cancel(order.id, NOW, "unfilled at cycle end")
        assert fake.cancelled == [replace(fake.placed[0][1], orderId=1)]
        assert (cancelled.status, cancelled.reason) == ("cancelled", "unfilled at cycle end")
        assert cancelled.filled_at == T1.isoformat()
        assert broker.open_orders() == ()
        assert [r["kind"] for r in rows] == ["submit", "cancel"]


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("ib_status", "expected"),
        [
            ("PendingSubmit", "open"),
            ("PreSubmitted", "open"),
            ("Submitted", "open"),
            ("PendingCancel", "open"),
            ("Filled", "filled"),
            ("Cancelled", "cancelled"),
            ("ApiCancelled", "cancelled"),
            ("Inactive", "rejected"),
        ],
    )
    def test_gateway_status_maps_to_order_status(self, ib_status: str, expected: str) -> None:
        fake = FakeIB(fills=(FakeFill(FakeExecution("k:SPLG:buy", 80.01), T1),))
        broker, _ = _broker(fake)
        broker.submit(BUY, NOW)
        fake.set_status("k:SPLG:buy", ib_status, "gateway says so")
        order = broker.submit(BUY, NOW)
        assert order.status == expected
        assert order.reason == ("gateway says so" if expected in ("cancelled", "rejected") else None)
        assert order.fill_price == (80.01 if expected == "filled" else None)

    def test_open_orders_are_filtered_to_this_account(self) -> None:
        other = FakeTrade(
            FakeContract("SGOV", "SMART", "USD"),
            FakeOrder("SELL", 5, 100.0, "DAY", "x", "DU2", orderId=9),
            FakeStatus("Submitted"),
        )
        fake = FakeIB(trades=(other,))
        broker, _ = _broker(fake)
        broker.submit(BUY, NOW)
        assert [o.client_id for o in broker.open_orders()] == ["k:SPLG:buy"]

    def test_process_journals_each_fill_once(self) -> None:
        fake = FakeIB(fills=(FakeFill(FakeExecution("k:SPLG:buy", 80.01), T1),))
        broker, rows = _broker(fake)
        broker.submit(BUY, NOW)
        assert broker.process() == ()
        fake.set_status("k:SPLG:buy", "Filled")
        (filled,) = broker.process()
        assert (filled.status, filled.fill_price, filled.filled_at) == ("filled", 80.01, T1.isoformat())
        assert broker.process() == ()
        assert [r["kind"] for r in rows] == ["submit", "fill"]


class TestAccount:
    def test_summary_and_whole_share_positions(self) -> None:
        fake = FakeIB(
            summary=(
                FakeValue("TotalCashValue", "1234.50"),
                FakeValue("SettledCash", "1000.00"),
                FakeValue("NetLiquidation", "9000.25"),
                FakeValue("AccountType", "CASH"),
            ),
            portfolio=(
                FakeItem(FakeContract("SGOV", "SMART", "USD"), 50.0, 100.1),
                FakeItem(FakeContract("SPLG", "SMART", "USD"), 0.0, 80.0),
                FakeItem(FakeContract("AAPL", "SMART", "USD"), 3.0, 190.0),
            ),
        )
        broker, _ = _broker(fake)
        account = broker.account({})
        assert (account.cash, account.settled_cash, account.equity) == (1234.5, 1000.0, 9000.25)
        assert [(p.symbol, p.qty, p.avg_cost) for p in account.positions] == [
            ("AAPL", 3, 190.0),
            ("SGOV", 50, 100.1),
        ]
        assert account.qty("SPLG") == 0


def test_adapter_imports_without_ib_async() -> None:
    assert "ib_async" not in sys.modules
