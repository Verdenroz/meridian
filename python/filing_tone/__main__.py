"""CLI: deep filing-tone study (fetch index, score text, run study).

    uv run python -m filing_tone fetch
    uv run --with torch --with transformers python -m filing_tone score
    uv run python -m filing_tone run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_study.study import PANEL
from market_data.client import DEFAULT_BASE_URL
from market_data.store import DEFAULT_STORE_DIR

from filing_tone.fetch import fetch_filing_index, load_filing_index, text_path
from filing_tone.score import processed_text, read_cached_tone, score_missing


def _covered_8k_texts(store_dir: Path) -> list[str]:
    texts = []
    for symbol in PANEL:
        for row in load_filing_index(store_dir, symbol) or []:
            if row["form"] == "8-K" and text_path(row["accession"]).is_file():
                texts.append(processed_text(text_path(row["accession"])))
    return texts


def cmd_fetch(args: argparse.Namespace) -> int:
    for symbol in PANEL:
        info = fetch_filing_index(args.base_url, args.store_dir, symbol)
        print(f"{info['symbol']}: {info['n_filings']} filings indexed", flush=True)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    texts = _covered_8k_texts(args.store_dir)
    print(f"{len(texts)} covered 8-K texts")
    written = score_missing(texts, batch_size=args.batch_size)
    print(f"wrote {written} new scores")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    for symbol in PANEL:
        rows = [
            r
            for r in (load_filing_index(args.store_dir, symbol) or [])
            if r["form"] == "8-K"
        ]
        covered = [r for r in rows if text_path(r["accession"]).is_file()]
        scored = sum(
            1
            for r in covered
            if read_cached_tone(processed_text(text_path(r["accession"]))) is not None
        )
        span = f"{rows[0]['date']}..{rows[-1]['date']}" if rows else "-"
        print(f"{symbol:>6}: 8-K {len(rows)}, text {len(covered)}, scored {scored}  [{span}]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from filing_tone.study import run_study

    report = run_study(args.store_dir)
    print(json.dumps(report, indent=2))
    v = report["verdict"]
    print(
        f"\nverdict: content {'PASS' if v['content_pass'] else 'FAIL'}, "
        f"rank {'PASS' if v['rank_pass'] else 'FAIL'} → "
        f"{'PASS' if v['overall_pass'] else 'FAIL'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="filing_tone")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="fetch per-symbol filing indexes").set_defaults(
        func=cmd_fetch
    )
    score = sub.add_parser("score", help="FinBERT-score covered 8-K texts")
    score.add_argument("--batch-size", type=int, default=64)
    score.set_defaults(func=cmd_score)
    sub.add_parser("status", help="coverage and scoring counts").set_defaults(
        func=cmd_status
    )
    sub.add_parser("run", help="run the pre-registered study").set_defaults(
        func=cmd_run
    )
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
