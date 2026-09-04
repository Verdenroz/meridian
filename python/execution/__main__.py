"""CLI: the paper book behind the execution layer.

    uv run python -m execution init --cash 10000   # open the paper account once
    uv run python -m execution cycle               # trade the latest shadow decision (local sim)
    uv run python -m execution cycle --broker ibkr # same decision on the IBKR paper gateway
    uv run python -m execution status
    uv run python -m execution watchdog          # exit 1 if a clock missed last session
    uv run python -m execution verify-broker     # read-only checks against a running IB Gateway
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.client import DEFAULT_BASE_URL
from shadow.log import DEFAULT_LOG_PATH

from execution.book import BOOK_NOTIONAL
from execution.broker.paper_state import (
    DEFAULT_IBKR_DIR,
    DEFAULT_PAPER_DIR,
    PaperState,
    PaperStore,
)
from execution.cycle import STRATEGY, run_cycle
from execution.notify import alert
from execution.watchdog import last_session, paper_findings, shadow_findings


def cmd_init(args: argparse.Namespace) -> int:
    store = PaperStore(args.paper_dir)
    if store.exists():
        print(f"paper account already exists at {store.state_path}")
        return 1
    store.save(PaperState.initial(args.cash))
    store.journal({"kind": "open", "cash": args.cash})
    print(f"opened paper account with {args.cash:.2f} at {store.state_path}")
    return 0


def cmd_cycle(args: argparse.Namespace) -> int:
    book = args.book or None
    if args.broker == "ibkr":
        report = _ibkr_cycle(args, book)
    else:
        store = PaperStore(args.paper_dir)
        report = run_cycle(store, args.log_path, args.base_url, args.strategy, book=book)
    print(f"{report['status']}: decision {report['decision_date']}")
    for order in report.get("orders", []):
        fill = f"@ {order['fill_price']:.2f}" if order["fill_price"] else order["reason"]
        print(f"  {order['side']:<4} {order['qty']:>5} {order['symbol']:<5} {order['status']:<9} {fill}")
    if report["status"] == "done":
        print(
            f"  equity {report['equity']:.2f}  cash {report['cash']:.2f}  {report['positions']}"
            f"  commission {report['commission']:.2f}"
        )
    return 0


def _ibkr_cycle(args: argparse.Namespace, book: float | None) -> dict:
    from execution.broker.ibkr import IbkrBroker, connect

    store = PaperStore(args.journal_dir)
    client = connect(args.host, args.port, args.client_id)
    try:
        account_id = args.account or client.managedAccounts()[0]
        broker = IbkrBroker(client, account_id, store.journal)
        return run_cycle(
            store, args.log_path, args.base_url, args.strategy, broker=broker, book=book
        )
    finally:
        client.disconnect()


def cmd_status(args: argparse.Namespace) -> int:
    store = PaperStore(args.paper_dir)
    if not store.exists():
        print(f"no paper account at {store.state_path}")
        return 0
    state = store.load()
    unsettled = sum(s.amount for s in state.unsettled)
    print(f"cash {state.cash:.2f} (unsettled {unsettled:.2f})")
    for p in state.positions:
        print(f"  {p.symbol:<5} {p.qty:>6} @ {p.avg_cost:.2f}")
    cycles = [r for r in store.journal_rows() if r["kind"] == "cycle"]
    if cycles:
        last = cycles[-1]
        print(
            f"{len(cycles)} cycles, last decision {last['decision_date']} "
            f"equity {last['equity']:.2f} drift {last['drift']:.3f} cost {last['cost_bps']:.1f}bps"
        )
    return 0


def cmd_watchdog(args: argparse.Namespace) -> int:
    session = last_session(args.base_url)
    strategies = tuple(args.strategy)
    findings = shadow_findings(args.log_path, session, strategies)
    if not findings:
        findings = paper_findings(PaperStore(args.paper_dir), session) + paper_findings(
            PaperStore(DEFAULT_IBKR_DIR), session
        )
    print(f"last session {session}: {'ok' if not findings else 'MISSED'}")
    for finding in findings:
        print(f"  {finding}")
        alert(f"watchdog: {finding[:80]}", {"kind": "watchdog", "session": session, "finding": finding})
    return 1 if findings else 0


def cmd_verify_broker(args: argparse.Namespace) -> int:
    from execution.broker.ibkr import connect
    from execution.broker.verify import verify

    try:
        client = connect(args.host, args.port, args.client_id)
    except OSError as exc:
        print(f"no IB Gateway at {args.host}:{args.port} ({exc})")
        return 1
    try:
        account_id = args.account or client.managedAccounts()[0]
        checks = verify(client, account_id)
    finally:
        client.disconnect()
    for check in checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name:<18} {check.detail}")
    failed = [c for c in checks if not c.ok]
    print(f"{account_id}: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def _gateway_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="4002 paper gateway, 4001 live")
    parser.add_argument("--client-id", type=int, default=9)
    parser.add_argument("--account", default=None, help="defaults to the first managed account")


def main() -> int:
    parser = argparse.ArgumentParser(prog="execution")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="open the paper account")
    init.add_argument("--cash", type=float, required=True)
    init.set_defaults(func=cmd_init)
    cycle = sub.add_parser("cycle", help="trade the latest shadow decision on paper")
    cycle.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    cycle.add_argument("--base-url", default=DEFAULT_BASE_URL)
    cycle.add_argument("--strategy", default=STRATEGY)
    cycle.add_argument("--broker", choices=("paper", "ibkr"), default="paper")
    cycle.add_argument("--book", type=float, default=BOOK_NOTIONAL, help="0 = whole account")
    cycle.add_argument("--journal-dir", type=Path, default=DEFAULT_IBKR_DIR)
    _gateway_args(cycle)
    cycle.set_defaults(func=cmd_cycle)
    status = sub.add_parser("status", help="paper account summary")
    status.set_defaults(func=cmd_status)
    watchdog = sub.add_parser("watchdog", help="alert if last session was not logged")
    watchdog.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    watchdog.add_argument("--base-url", default=DEFAULT_BASE_URL)
    watchdog.add_argument("--strategy", action="append", default=[STRATEGY])
    watchdog.set_defaults(func=cmd_watchdog)
    verify = sub.add_parser("verify-broker", help="read-only checks against a running IB Gateway")
    _gateway_args(verify)
    verify.set_defaults(func=cmd_verify_broker)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
