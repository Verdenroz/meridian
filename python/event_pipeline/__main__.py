"""CLI: score 8-K filings with the local LLM.

    uv run python -m event_pipeline score --limit 5      # newest unscored 8-Ks
    uv run python -m event_pipeline status
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from event_study.study import PANEL
from market_data.client import DEFAULT_BASE_URL as FINANCE_QUERY_BASE_URL
from market_data.store import DEFAULT_STORE_DIR

from event_pipeline.filings import (
    fetch_document_text,
    load_8k_metadata,
    refresh_8k_metadata,
)
from event_pipeline.scorer import DEFAULT_BASE_URL, DEFAULT_MODEL, EventScorer
from event_pipeline.store import (
    DEFAULT_SCORES_PATH,
    append_score,
    load_scored_keys,
    load_scores,
)


def _pending(args: argparse.Namespace, scorer_version: str) -> list[dict]:
    scored = load_scored_keys(args.scores_path)
    pending: list[dict] = []
    for symbol in args.symbols:
        meta = load_8k_metadata(args.store_dir, symbol)
        if meta is None or args.refresh:
            meta = refresh_8k_metadata(FINANCE_QUERY_BASE_URL, args.store_dir, symbol)
        for filing in meta["filings"]:
            if args.since and filing["date"] < args.since:
                continue
            if not filing["document"]:
                continue
            if (filing["accession"], scorer_version) in scored:
                continue
            pending.append({**filing, "symbol": symbol, "cik": meta["cik"]})
    pending.sort(key=lambda f: f["date"], reverse=True)
    return pending


def cmd_score(args: argparse.Namespace) -> int:
    scorer = EventScorer(model=args.model, base_url=args.llm_base_url)
    pending = _pending(args, scorer.scorer_version)
    if not pending:
        print("nothing to score")
        return 0
    batch = pending[: args.limit]
    print(f"{len(pending)} unscored 8-Ks; scoring {len(batch)} (scorer {scorer.scorer_version})")
    failures = 0
    for filing in batch:
        started = time.monotonic()
        try:
            text = fetch_document_text(filing["cik"], filing["accession"], filing["document"])
            score = scorer.score(filing["symbol"], filing["date"], text)
        except Exception as exc:  # noqa: BLE001 — batch keeps going, exit code reports
            failures += 1
            print(f"  {filing['symbol']} {filing['date']}: FAILED — {str(exc)[:160]}", file=sys.stderr)
            continue
        append_score(
            args.scores_path,
            filing["symbol"],
            filing["accession"],
            filing["date"],
            scorer.scorer_version,
            score.model_dump(),
        )
        elapsed = time.monotonic() - started
        print(
            f"  {filing['symbol']} {filing['date']}: severity={score.severity} "
            f"scope={score.scope} direction={score.direction} ({elapsed:.0f}s)\n"
            f"    {score.rationale}"
        )
    return 1 if failures else 0


def cmd_status(args: argparse.Namespace) -> int:
    rows = load_scores(args.scores_path)
    if not rows:
        print(f"no scores yet at {args.scores_path}")
        return 0
    by_version: dict[str, int] = {}
    for row in rows:
        by_version[row["scorer_version"]] = by_version.get(row["scorer_version"], 0) + 1
    print(f"{len(rows)} scores at {args.scores_path}")
    for version, count in sorted(by_version.items()):
        print(f"  {version}: {count}")
    severities = [r["severity"] for r in rows if "severity" in r]
    if severities:
        from collections import Counter

        dist = Counter(severities)
        print("  severity distribution:", dict(sorted(dist.items())))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="event_pipeline")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--scores-path", type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument(
        "--symbols",
        type=lambda s: tuple(x.strip().upper() for x in s.split(",") if x.strip()),
        default=PANEL,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score unscored 8-Ks, newest first")
    score.add_argument("--limit", type=int, default=5)
    score.add_argument("--since", default=None, help="only filings on/after this date")
    score.add_argument("--refresh", action="store_true", help="refetch 8-K metadata")
    score.add_argument("--model", default=DEFAULT_MODEL)
    score.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    score.set_defaults(func=cmd_score)
    status = sub.add_parser("status", help="score counts by scorer version")
    status.set_defaults(func=cmd_status)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
