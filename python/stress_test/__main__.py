"""CLI: python -m stress_test status

Reports the episode state and, once the first episode's window has closed,
judges it and appends the verdict to the git-tracked record exactly once.
Nothing about either book is shown while an episode is open.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from credit_race.race import RACE_SYMBOLS
from harness.panel import build_panel
from market_data.client import DEFAULT_BASE_URL
from market_data.store import DEFAULT_STORE_DIR
from shadow.job import refresh
from shadow.log import DEFAULT_LOG_PATH, load_decisions

from stress_test import GATE
from stress_test.evaluate import book_returns, find_episode, path_stats, verdict

DEFAULT_RECORD = Path("data/races/stress_test.jsonl")


def _carried(values: list[float | None]) -> list[float]:
    out, last = [], 0.0
    for v in values:
        last = v if v is not None else last
        out.append(last)
    return out


def evaluate(store_dir: Path, log_path: Path, today: str) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    episode = find_episode(panel.dates, _carried(panel.adj_close["SPY"]))
    if episode is None or panel.dates[episode.trigger] > GATE["deadline"]:
        status = "no_evidence" if today > GATE["deadline"] else "no_episode"
        return {"status": status}
    dates = {k: panel.dates[getattr(episode, k)] for k in ("peak", "trigger")}
    if not episode.closed:
        return {"status": "open", **dates}
    rows = load_decisions(log_path)
    # A peak older than the shadow log starts the window at the log's first row.
    first_row = min(r["decision_date"] for r in rows)
    start = max(episode.peak, panel.dates.index(first_row) + 1)
    books = {
        role: path_stats(book_returns(rows, GATE[role], panel, start, episode.end))
        for role in ("leader", "null")
    }
    return {
        "status": "judged",
        **dates,
        "window": [panel.dates[start], panel.dates[episode.end]],
        "books": books,
        "verdict": verdict(books["leader"], books["null"]),
    }


def record_once(path: Path, report: dict) -> bool:
    if report["status"] not in ("judged", "no_evidence"):
        return False
    if path.exists() and path.read_text().strip():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**report, "gate": GATE, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return True


def cmd_status(args: argparse.Namespace) -> int:
    store = Path(args.store_dir)
    if not args.no_fetch:
        refresh(args.base_url, store)
    today = datetime.now(UTC).date().isoformat()
    report = evaluate(store, Path(args.log_path), today)
    print(json.dumps(report, indent=2))
    if record_once(Path(args.record), report):
        print(f"verdict recorded at {args.record}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="stress_test")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--record", default=str(DEFAULT_RECORD))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    p_status = sub.add_parser("status", help="episode state; judges once the window closes")
    p_status.add_argument("--no-fetch", action="store_true")
    p_status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
