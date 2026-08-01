"""CLI: retrieve the nearest historical analogs and print the briefing.

    uv run python -m episodic run                       # latest close
    uv run python -m episodic run --date 2020-03-16     # any panel date
    uv run python -m episodic build-news                # news centroid store
    uv run python -m episodic news --date 2020-03-16    # similar news days
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harness.panel import build_panel
from market_data.store import DEFAULT_STORE_DIR

from episodic.briefing import build_briefing, format_briefing
from episodic.knn import DEFAULT_K, retrieve_analogs
from episodic.state import STATE_SYMBOLS, build_state_matrix


def cmd_run(args: argparse.Namespace) -> int:
    panel = build_panel(args.store_dir, STATE_SYMBOLS)
    states = build_state_matrix(panel, args.store_dir)
    if args.date is not None:
        try:
            query_index = panel.dates.index(args.date)
        except ValueError:
            print(f"{args.date} is not a panel date (span {panel.dates[0]}..{panel.dates[-1]})")
            return 1
    else:
        query_index = panel.n_days - 1
    analogs = retrieve_analogs(states, panel.dates, query_index, k=args.k)
    print(format_briefing(build_briefing(panel, states, query_index, analogs)))
    return 0


def cmd_build_news(args: argparse.Namespace) -> int:
    from episodic.news_analogs import build_centroids

    info = build_centroids()
    print(f"news centroids: {info['n_days']} days, span {info['span']}")
    return 0


def cmd_news(args: argparse.Namespace) -> int:
    from episodic.news_analogs import nearest_news_days

    result = nearest_news_days(args.date, k=args.k)
    if result is None:
        print(
            f"no news centroid for {args.date} (needs the centroid store, and "
            "for live dates >=30 archived titles + the local embedding endpoint)"
        )
        return 1
    print(f"news-similar days to {args.date} (centroid cosine):")
    for date, cos in result:
        print(f"  {date}  {cos:.3f}")
    print("explanation only — news analogs do not touch allocation")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="episodic")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="nearest analogs + briefing for a date")
    run.add_argument("--date", help="panel date to query (default: latest close)")
    run.add_argument("-k", type=int, default=DEFAULT_K, help="number of analogs")
    run.set_defaults(func=cmd_run)
    build_news = sub.add_parser("build-news", help="build the news centroid store")
    build_news.set_defaults(func=cmd_build_news)
    news = sub.add_parser("news", help="news-similar historical days for a date")
    news.add_argument("--date", required=True, help="UTC date to query")
    news.add_argument("-k", type=int, default=5, help="number of similar days")
    news.set_defaults(func=cmd_news)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
