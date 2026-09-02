"""CLI: the paper book behind the execution layer.

    uv run python -m execution init --cash 10000   # open the paper account once
    uv run python -m execution cycle               # trade the latest shadow decision
    uv run python -m execution status
    uv run python -m execution watchdog          # exit 1 if a clock missed last session
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.client import DEFAULT_BASE_URL
from shadow.log import DEFAULT_LOG_PATH

from execution.broker.paper_state import DEFAULT_PAPER_DIR, PaperState, PaperStore
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
    report = run_cycle(PaperStore(args.paper_dir), args.log_path, args.base_url, args.strategy)
    print(f"{report['status']}: decision {report['decision_date']}")
    for order in report.get("orders", []):
        fill = f"@ {order['fill_price']:.2f}" if order["fill_price"] else order["reason"]
        print(f"  {order['side']:<4} {order['qty']:>5} {order['symbol']:<5} {order['status']:<9} {fill}")
    if report["status"] == "done":
        print(f"  equity {report['equity']:.2f}  cash {report['cash']:.2f}  {report['positions']}")
    return 0


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
        findings = paper_findings(PaperStore(args.paper_dir), session)
    print(f"last session {session}: {'ok' if not findings else 'MISSED'}")
    for finding in findings:
        print(f"  {finding}")
        alert(f"watchdog: {finding[:80]}", {"kind": "watchdog", "session": session, "finding": finding})
    return 1 if findings else 0


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
    cycle.set_defaults(func=cmd_cycle)
    status = sub.add_parser("status", help="paper account summary")
    status.set_defaults(func=cmd_status)
    watchdog = sub.add_parser("watchdog", help="alert if last session was not logged")
    watchdog.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    watchdog.add_argument("--base-url", default=DEFAULT_BASE_URL)
    watchdog.add_argument("--strategy", action="append", default=[STRATEGY])
    watchdog.set_defaults(func=cmd_watchdog)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
