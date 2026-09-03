"""Read-only checks of a live IB Gateway against the adapter's assumptions.

Each check exercises one mapping the adapter depends on: the login manages
the account, the summary carries the cash and equity tags, the executable
symbols qualify as SMART/USD stocks, open and completed orders parse, and a
one-share DAY limit what-if returns a commission. Nothing is placed.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from execution.broker.ibkr import IbkrBroker
from execution.targets import CASH_SYMBOL, SYMBOL_MAP

UNSET_DOUBLE = 1.7976931348623157e308
SUMMARY_TAGS = ("TotalCashValue", "SettledCash", "NetLiquidation")
VERIFY_SYMBOLS = (*SYMBOL_MAP.values(), CASH_SYMBOL)
WHAT_IF_LIMIT = 1.0


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _managed(client: Any, account_id: str) -> str:
    accounts = client.managedAccounts()
    if account_id not in accounts:
        raise ValueError(f"{account_id} not in managed accounts {accounts}")
    return ", ".join(accounts)


def _summary(client: Any, account_id: str) -> str:
    summary = {v.tag: v.value for v in client.accountSummary(account_id)}
    missing = [t for t in SUMMARY_TAGS if t not in summary]
    if missing:
        raise ValueError(f"summary lacks {missing}")
    return " ".join(f"{t}={float(summary[t]):.2f}" for t in SUMMARY_TAGS)


def _account(broker: IbkrBroker) -> str:
    account = broker.account({})
    held = " ".join(f"{p.symbol}:{p.qty}" for p in account.positions) or "flat"
    return f"cash={account.cash:.2f} settled={account.settled_cash:.2f} equity={account.equity:.2f} {held}"


def _contracts(client: Any, api: Any) -> str:
    contracts = [api.Stock(s, "SMART", "USD") for s in VERIFY_SYMBOLS]
    qualified = client.qualifyContracts(*contracts)
    unqualified = [c.symbol for c in qualified if not c.conId]
    if len(qualified) != len(contracts) or unqualified:
        raise ValueError(f"unqualified {unqualified or 'some contracts'}")
    return " ".join(f"{c.symbol}={c.conId}" for c in qualified)


def _orders(broker: IbkrBroker) -> str:
    open_orders = broker.open_orders()
    broker._known("verify:none")
    return f"{len(open_orders)} open, order and fill history parse"


def _what_if(client: Any, api: Any, account_id: str) -> str:
    contract = api.Stock(VERIFY_SYMBOLS[0], "SMART", "USD")
    order = api.LimitOrder("BUY", 1, WHAT_IF_LIMIT, tif="DAY", account=account_id)
    state = client.whatIfOrder(contract, order)
    commission = state.commission
    if state.warningText:
        raise ValueError(state.warningText)
    if not math.isfinite(commission) or commission >= UNSET_DOUBLE:
        raise ValueError(f"no commission in what-if state {state.status!r}")
    return f"commission={commission:.4f} margin_change={state.initMarginChange}"


def _checks(client: Any, account_id: str, api: Any) -> Iterator[tuple[str, Callable[[], str]]]:
    broker = IbkrBroker(client, account_id, lambda _: None, api)
    yield "connected", lambda: str(client.isConnected())
    yield "account managed", lambda: _managed(client, account_id)
    yield "summary tags", lambda: _summary(client, account_id)
    yield "account mapping", lambda: _account(broker)
    yield "contracts qualify", lambda: _contracts(client, api)
    yield "orders readable", lambda: _orders(broker)
    yield "what-if order", lambda: _what_if(client, api, account_id)


def verify(client: Any, account_id: str, api: Any = None) -> tuple[Check, ...]:
    api = api if api is not None else importlib.import_module("ib_async")
    results = []
    for name, probe in _checks(client, account_id, api):
        try:
            results.append(Check(name, True, probe()))
        except Exception as exc:  # noqa: BLE001
            results.append(Check(name, False, f"{type(exc).__name__}: {exc}"))
    return tuple(results)
