"""A fixed-notional book carved out of a larger account.

The IBKR paper account holds a million in simulated cash; the live account
will hold ten thousand. Sized against the whole account, the paper book
would trade a hundred times the intended size, so the cycle sizes against
a book instead: its cash is the notional less the net cash its own fills
consumed, read back from the append-only journal; its positions are the
account's, since the book is the only trader there. With no notional the
account is the book.
"""

from __future__ import annotations

from execution.orders import Account

BOOK_NOTIONAL = 10_000.0


def net_spent(rows: list[dict]) -> float:
    fills = {r["order"]["id"]: r["order"] for r in rows if r.get("kind") == "fill"}
    return sum(_cash_out(o) for o in fills.values())


def _cash_out(order: dict) -> float:
    notional = order["qty"] * order["fill_price"]
    if order["side"] == "buy":
        return notional + order["commission"]
    return -(notional - order["commission"])


def book_account(account: Account, rows: list[dict], notional: float | None) -> Account:
    if notional is None:
        return account
    cash = min(notional - net_spent(rows), account.cash)
    held = account.equity - account.cash
    return Account(
        cash=cash,
        settled_cash=min(cash, account.settled_cash),
        equity=cash + held,
        positions=account.positions,
    )
