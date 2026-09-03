from dataclasses import dataclass, replace
from typing import Any

from execution.broker.verify import UNSET_DOUBLE, verify

ACCOUNT = "DU1"


@dataclass(frozen=True)
class FakeContract:
    symbol: str
    exchange: str
    currency: str
    conId: int = 0


@dataclass(frozen=True)
class FakeOrder:
    action: str
    totalQuantity: float
    lmtPrice: float
    tif: str = ""
    orderRef: str = ""
    account: str = ""


@dataclass(frozen=True)
class FakeValue:
    tag: str
    value: str


@dataclass(frozen=True)
class FakeItem:
    contract: FakeContract
    position: float
    averageCost: float


@dataclass(frozen=True)
class FakeState:
    status: str = "PreSubmitted"
    commission: float = 0.35
    initMarginChange: str = "0.0"
    warningText: str = ""


CLEAN_STATE = FakeState()


class FakeApi:
    Stock = FakeContract

    @staticmethod
    def LimitOrder(action: str, qty: float, limit: float, **kw: Any) -> FakeOrder:
        return FakeOrder(action, qty, limit, **kw)


class FakeIB:
    def __init__(
        self,
        summary: tuple[FakeValue, ...],
        portfolio: tuple[FakeItem, ...] = (),
        state: FakeState = CLEAN_STATE,
    ) -> None:
        self._summary = summary
        self._portfolio = portfolio
        self.state = state

    def isConnected(self) -> bool:
        return True

    def managedAccounts(self) -> list[str]:
        return [ACCOUNT]

    def accountSummary(self, account: str) -> list[FakeValue]:
        return list(self._summary)

    def portfolio(self, account: str) -> list[FakeItem]:
        return list(self._portfolio)

    def qualifyContracts(self, *contracts: FakeContract) -> list[FakeContract]:
        return [replace(c, conId=i + 100) for i, c in enumerate(contracts)]

    def whatIfOrder(self, contract: FakeContract, order: FakeOrder) -> FakeState:
        return self.state

    def trades(self) -> list[Any]:
        return []

    def openTrades(self) -> list[Any]:
        return []

    def reqCompletedOrders(self, apiOnly: bool) -> list[Any]:
        return []

    def fills(self) -> list[Any]:
        return []


SUMMARY = (
    FakeValue("TotalCashValue", "10000.0"),
    FakeValue("SettledCash", "9500.0"),
    FakeValue("NetLiquidation", "10000.0"),
    FakeValue("Currency", "USD"),
)
HELD = (FakeItem(FakeContract("SPLG", "SMART", "USD"), 12.0, 80.25),)


def test_verify_passes_every_check_on_a_conforming_gateway() -> None:
    checks = verify(FakeIB(SUMMARY, HELD), ACCOUNT, FakeApi)
    assert [c.name for c in checks] == [
        "connected",
        "account managed",
        "summary tags",
        "account mapping",
        "contracts qualify",
        "orders readable",
        "what-if order",
    ]
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert "SPLG:12" in checks[3].detail
    assert "SPLG=100" in checks[4].detail and "SGOV=101" in checks[4].detail


def test_verify_reports_failures_without_aborting() -> None:
    fake = FakeIB(SUMMARY[:2], state=FakeState(commission=UNSET_DOUBLE))
    by_name = {c.name: c for c in verify(fake, "DU2", FakeApi)}
    assert not by_name["account managed"].ok and "DU2" in by_name["account managed"].detail
    assert not by_name["summary tags"].ok and "NetLiquidation" in by_name["summary tags"].detail
    assert not by_name["account mapping"].ok
    assert not by_name["what-if order"].ok
    assert by_name["contracts qualify"].ok and by_name["orders readable"].ok
