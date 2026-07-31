"""CLI: daily shadow decisions for the paper-trading record.

    uv run python -m shadow run          # fetch latest data, log decisions
    uv run python -m shadow run --no-fetch
    uv run python -m shadow status
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from market_data.client import DEFAULT_BASE_URL
from market_data.store import DEFAULT_STORE_DIR

from shadow.job import run_decision
from shadow.log import DEFAULT_LOG_PATH, load_decisions


def cmd_run(args: argparse.Namespace) -> int:
    report = run_decision(
        args.store_dir, args.log_path, args.base_url, fetch=not args.no_fetch
    )
    print(f"decision date {report['decision_date']}")
    for row in report["written"]:
        weights = ", ".join(f"{s}={w:.3f}" for s, w in sorted(row["weights"].items()))
        print(
            f"  {row['strategy']:<16} {weights or 'all cash'}  "
            f"cash={row['cash_weight']:.3f}  {row['meta']}"
        )
    if report["skipped"]:
        print(f"  already logged: {', '.join(report['skipped'])}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rows = load_decisions(args.log_path)
    if not rows:
        print(f"no decisions yet at {args.log_path}")
        return 0
    by_strategy: Counter[str] = Counter(r["strategy"] for r in rows)
    last = {r["strategy"]: r["decision_date"] for r in rows}
    print(f"{len(rows)} decisions at {args.log_path}")
    for name, count in sorted(by_strategy.items()):
        print(f"  {name:<16} {count} rows, last {last[name]}")
    dates = sorted({r["decision_date"] for r in rows})
    print(f"  span: {dates[0]}..{dates[-1]} ({len(dates)} decision days)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="shadow")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="fetch latest data and log today's decisions")
    run.add_argument("--no-fetch", action="store_true")
    run.set_defaults(func=cmd_run)
    status = sub.add_parser("status", help="decision log summary")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
