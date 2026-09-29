"""CLI: python -m daytrade {replay|status}

replay walks every archived 1m session not yet journaled, oldest first, and
appends one row per session to the git-tracked, append-only journal. Rows
are never edited or regenerated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from bars_archive.store import coverage, read_session, session_path
from market_data.client import DEFAULT_BASE_URL
from shadow.job import refresh

from daytrade import GATE
from daytrade.regime import calm_posteriors
from daytrade.signal import find_trade, session_sigma
from daytrade.sim import open_to_close, random_control, trade_fill

JOB_VERSION = 1
OUT_OF_SAMPLE_FROM = date(2026, 9, 30)
DEFAULT_BARS = Path("archive/bars")
DEFAULT_JOURNAL = Path("data/execution/daytrade/journal.jsonl")
DEFAULT_STORE = Path("data/market")
SYMBOLS = ("SPY", "SH")


def journaled(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["session"] for line in path.read_text().splitlines()}


def pending_sessions(bars_root: Path, done: set[str]) -> list[date]:
    have = coverage(bars_root, "1m")
    complete = sorted(s for s, symbols in have.items() if set(SYMBOLS) <= symbols)
    eligible = complete[GATE["sigma_sessions"] :]
    return [date.fromisoformat(s) for s in eligible if s not in done]


def session_row(bars_root: Path, session: date, calm: float, replayed_at: str) -> dict:
    bars = {s: read_session(session_path(bars_root, "1m", session, s)) for s in SYMBOLS}
    spy = bars["SPY"]
    sigma = session_sigma(_prior_spy(bars_root, session))
    trade = find_trade(spy, sigma, GATE) if calm > GATE["calm_posterior"] else None
    return {
        "kind": "session",
        "session": session.isoformat(),
        "in_sample": session < OUT_OF_SAMPLE_FROM,
        "calm_posterior": round(calm, 6),
        "sigma": sigma,
        "trade": None if trade is None else trade_fill(trade, bars, GATE),
        "random": None
        if trade is None
        else trade_fill(random_control(trade, spy, session, GATE), bars, GATE),
        "open_close": open_to_close(spy, GATE),
        "job_version": JOB_VERSION,
        "replayed_at": replayed_at,
    }


def _prior_spy(bars_root: Path, session: date) -> list[list[dict]]:
    have = coverage(bars_root, "1m")
    prior = sorted(d for d, symbols in have.items() if "SPY" in symbols and d < session.isoformat())
    return [
        read_session(session_path(bars_root, "1m", date.fromisoformat(d), "SPY"))
        for d in prior[-GATE["sigma_sessions"] :]
    ]


def run_replay(args: argparse.Namespace) -> int:
    bars_root, journal = Path(args.bars_dir), Path(args.journal)
    sessions = pending_sessions(bars_root, journaled(journal))
    if not sessions:
        print("nothing to replay")
        return 0
    store = Path(args.store_dir)
    if args.fetch:
        refresh(args.base_url, store)
    calm = calm_posteriors(store, sessions)
    replayed_at = datetime.now(UTC).isoformat(timespec="seconds")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a") as fh:
        for session in sessions:
            row = session_row(bars_root, session, calm[session], replayed_at)
            fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            traded = row["trade"]["symbol"] if row["trade"] else "-"
            print(f"{row['session']} calm={row['calm_posterior']:.3f} trade={traded}")
    return 0


def run_status(args: argparse.Namespace) -> int:
    """Descriptive only: the gate is judged once, at the registered count."""
    path = Path(args.journal)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    oos = [r for r in rows if not r["in_sample"]]
    trades = [r for r in oos if r["trade"]]
    print(f"sessions={len(rows)} out_of_sample={len(oos)} oos_trades={len(trades)}/{GATE['gate_trades']}")
    if trades:
        mean = sum(r["trade"]["net_bps"] for r in trades) / len(trades)
        ctrl = sum(r["random"]["net_bps"] for r in trades) / len(trades)
        print(f"mean_net_bps={mean:.2f} random_control_bps={ctrl:.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="daytrade")
    parser.add_argument("--bars-dir", default=str(DEFAULT_BARS))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    sub = parser.add_subparsers(dest="command", required=True)
    p_replay = sub.add_parser("replay", help="journal every archived session not yet replayed")
    p_replay.add_argument("--store-dir", default=str(DEFAULT_STORE))
    p_replay.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p_replay.add_argument("--no-fetch", dest="fetch", action="store_false")
    p_replay.set_defaults(func=run_replay)
    sub.add_parser("status", help="descriptive journal summary").set_defaults(func=run_status)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
