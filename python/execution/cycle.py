"""The daily paper cycle: latest shadow decision to orders, fills, journal.

Runs after the open of the session that follows the decision, so live and
shadow consume the identical post-close row. Anything missing holds: a
closed market, a stale decision, a missing quote, the halt flag, a guardrail
rejection. Orders are keyed by (decision_date, strategy, symbol, side), so a
rerun after a crash finds its own orders instead of placing them twice.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from shadow.job import check_staleness
from shadow.log import load_decisions

from execution.broker.paper import PaperBroker
from execution.broker.paper_state import PaperStore
from execution.guardrails import Rejection, validate
from execution.halt import HALT_PATH, halted
from execution.notify import alert
from execution.orders import Account, Order, Quote
from execution.quotes import fetch_quotes
from execution.targets import live_targets, plan_orders

STRATEGY = "null_voltarget"
JOB_VERSION = "paper-v1"
DRIFT_TOLERANCE = 0.02


def latest_decision(log_path: Path, strategy: str) -> dict:
    rows = [r for r in load_decisions(log_path) if r["strategy"] == strategy]
    if not rows:
        raise RuntimeError(f"no shadow decisions for {strategy} at {log_path}")
    return max(rows, key=lambda r: r["decision_date"])


def cycled_keys(store: PaperStore) -> set[tuple[str, str]]:
    return {
        (r["decision_date"], r["strategy"])
        for r in store.journal_rows()
        if r["kind"] == "cycle"
    }


def run_cycle(
    store: PaperStore,
    log_path: Path,
    base_url: str,
    strategy: str = STRATEGY,
    today: date | None = None,
    halt_path: Path = HALT_PATH,
) -> dict:
    today = today or datetime.now(UTC).date()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    decision = latest_decision(log_path, strategy)
    decision_date = decision["decision_date"]
    check_staleness(decision_date, today)
    base = {"decision_date": decision_date, "strategy": strategy, "fetched_at": now}
    if (decision_date, strategy) in cycled_keys(store):
        return {**base, "status": "skipped"}
    if halted(halt_path):
        return _record(store, "halt", {**base, "status": "halted"})

    targets = live_targets(decision["weights"], decision["cash_weight"])
    broker = PaperBroker(store)
    symbols = sorted(set(targets) | {p.symbol for p in broker.state.positions})
    quotes = fetch_quotes(base_url, symbols)
    closed = [s for s, q in quotes.items() if q.market_state != "REGULAR"]
    if closed:
        return {**base, "status": "market_closed", "symbols": closed}

    broker.settle(today)
    key = f"{decision_date}:{strategy}"
    planned = plan_orders(broker.account(quotes), targets, quotes, key)
    approved = validate(broker.account(quotes), planned, quotes)
    if isinstance(approved, Rejection):
        order = None if approved.order is None else approved.order.__dict__
        row = {**base, "status": "rejected", "reason": approved.reason, "order": order}
        return _record(store, "rejection", row)

    for side in ("sell", "buy"):
        for request in (r for r in approved if r.side == side):
            broker.submit(request, now)
        broker.process(quotes, now, today)
    for open_order in broker.open_orders():
        broker.cancel(open_order.id, now, "unfilled at cycle end")

    after = broker.account(quotes)
    orders = tuple(o for o in broker.state.orders if o.client_id.startswith(key))
    row = {
        **base,
        "kind": "cycle",
        "job_version": JOB_VERSION,
        "targets": targets,
        "orders": [o.to_dict() for o in orders],
        "cash": round(after.cash, 2),
        "equity": round(after.equity, 2),
        "positions": {p.symbol: p.qty for p in after.positions},
        "drift": round(max_drift(after, targets, quotes), 4),
        "cost_bps": round(realized_cost_bps(orders, quotes), 2),
        "status": "done",
    }
    store.journal(row)
    alert(f"paper cycle {decision_date}: {len(orders)} orders", row)
    if row["drift"] > DRIFT_TOLERANCE:
        alert(f"paper drift {row['drift']:.3f} exceeds {DRIFT_TOLERANCE}", row)
    return row


def _record(store: PaperStore, kind: str, row: dict) -> dict:
    store.journal({**row, "kind": kind})
    alert(f"paper {kind}: {row['status']}", row)
    return row


def max_drift(account: Account, targets: dict[str, float], quotes: dict[str, Quote]) -> float:
    symbols = set(targets) | {p.symbol for p in account.positions}
    actual = {p.symbol: p.qty * quotes[p.symbol].last / account.equity for p in account.positions}
    return max(abs(actual.get(s, 0.0) - targets.get(s, 0.0)) for s in symbols)


def realized_cost_bps(orders: tuple[Order, ...], quotes: dict[str, Quote]) -> float:
    """Notional-weighted slippage of fills against the previous close, the
    price the harness assumed the decision traded at. Positive is a cost."""
    fills = [o for o in orders if o.status == "filled" and o.fill_price is not None]
    priced = [(o, quotes[o.symbol].prev_close) for o in fills]
    priced = [(o, ref) for o, ref in priced if ref]
    if not priced:
        return 0.0
    notional = sum(o.qty * ref for o, ref in priced if ref)
    cost = sum(
        o.qty * (o.fill_price - ref) * (1 if o.side == "buy" else -1)
        for o, ref in priced
        if ref and o.fill_price
    )
    return cost / notional * 1e4
