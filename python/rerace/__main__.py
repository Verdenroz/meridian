"""CLI: monthly re-race of the held gates.

    uv run python -m rerace run          # fetch latest data, re-run held races
    uv run python -m rerace run --no-fetch
    uv run python -m rerace status
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_data.client import DEFAULT_BASE_URL
from market_data.store import DEFAULT_STORE_DIR
from shadow.log import load_decisions

from rerace.job import DEFAULT_LOG_PATH, run_reraces


def cmd_run(args: argparse.Namespace) -> int:
    report = run_reraces(
        args.store_dir, args.log_path, args.base_url, fetch=not args.no_fetch
    )
    print(f"re-race month {report['month']}")
    for row in report["written"]:
        null_sharpe = row["verdict"]["null_sharpe"]
        print(
            f"  {row['race']:<14} {'PASS' if row['race_pass'] else 'FAIL'}  "
            f"null sharpe {null_sharpe:.3f}  "
            + "  ".join(
                f"{label}={m['sharpe']:.3f}"
                for label, m in row["metrics"].items()
                if "null" not in label and "buy" not in label
            )
        )
    if report["skipped"]:
        print(f"  already logged this month: {', '.join(report['skipped'])}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rows = load_decisions(args.log_path)
    if not rows:
        print(f"no re-race verdicts yet at {args.log_path}")
        return 0
    print(f"{len(rows)} verdicts at {args.log_path}")
    for row in rows:
        print(
            f"  {row['fetched_at'][:7]}  {row['race']:<14} "
            f"{'PASS' if row['race_pass'] else 'FAIL'}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="rerace")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="re-run the held races and log verdicts")
    run.add_argument("--no-fetch", action="store_true")
    run.set_defaults(func=cmd_run)
    status = sub.add_parser("status", help="verdict log summary")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
