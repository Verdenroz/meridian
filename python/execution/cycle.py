"""The daily paper cycle: latest shadow decision to orders, fills, journal.

Runs after the open of the session that follows the decision, so live and
shadow consume the identical post-close row. Anything missing holds: a
closed market, a stale decision, a missing quote. Orders are keyed by
(decision_date, strategy, symbol, side), so a rerun after a crash finds
its own orders instead of placing them twice.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from shadow.job import check_staleness
from shadow.log import load_decisions

from execution.broker.paper import PaperBroker
from execution.broker.paper_state import PaperStore
from execution.quotes import fetch_quotes
from execution.targets import live_targets, plan_orders

STRATEGY = "null_voltarget"
JOB_VERSION = "paper-v1"


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
) -> dict:
    today = today or datetime.now(UTC).date()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    decision = latest_decision(log_path, strategy)
    decision_date = decision["decision_date"]
    check_staleness(decision_date, today)
    if (decision_date, strategy) in cycled_keys(store):
        return {"status": "skipped", "decision_date": decision_date}

    targets = live_targets(decision["weights"], decision["cash_weight"])
    broker = PaperBroker(store)
    symbols = sorted(set(targets) | {p.symbol for p in broker.state.positions})
    quotes = fetch_quotes(base_url, symbols)
    closed = [s for s, q in quotes.items() if q.market_state != "REGULAR"]
    if closed:
        return {"status": "market_closed", "decision_date": decision_date, "symbols": closed}

    broker.settle(today)
    key = f"{decision_date}:{strategy}"
    requests = plan_orders(broker.account(quotes), targets, quotes, key)
    for side in ("sell", "buy"):
        for request in (r for r in requests if r.side == side):
            broker.submit(request, now)
        broker.process(quotes, now, today)
    for order in broker.open_orders():
        broker.cancel(order.id, now, "unfilled at cycle end")

    after = broker.account(quotes)
    row = {
        "kind": "cycle",
        "job_version": JOB_VERSION,
        "decision_date": decision_date,
        "strategy": strategy,
        "fetched_at": now,
        "targets": targets,
        "orders": [
            o.to_dict() for o in broker.state.orders if o.client_id.startswith(key)
        ],
        "cash": round(after.cash, 2),
        "equity": round(after.equity, 2),
        "positions": {p.symbol: p.qty for p in after.positions},
        "status": "done",
    }
    store.journal(row)
    return row
