"""CLI: run the pre-registered score study.

    uv run python -m score_study run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from event_pipeline.store import DEFAULT_SCORES_PATH
from market_data.store import DEFAULT_STORE_DIR

from score_study.study import run_study


def cmd_run(args: argparse.Namespace) -> int:
    report = run_study(args.store_dir, args.scores_path)
    gate = report["gate"]
    print(
        f"pre-registered gate (h={gate['horizon_days']}d, both legs must pass):\n"
        f"  A: spearman(severity, log ratio) ≥ {gate['min_spearman']} @ p ≤ {gate['max_p_value']}\n"
        f"  B: median uplift sev≥{gate['serious_min_severity']} vs sev≤{gate['routine_max_severity']}"
        f" ≥ {gate['min_median_uplift']*100:.0f}% @ same p"
    )
    for horizon, legs in report["horizons"].items():
        rank, bands, direction = legs["rank"], legs["bands"], legs["direction"]
        print(f"\nhorizon {horizon}d:")
        print(
            f"  rank       n={rank['n_days']:>5}  spearman={rank['spearman']:+.3f}  "
            f"p={rank['p_value']:.2e}"
        )
        if bands is None:
            print("  bands      insufficient events")
        else:
            print(
                f"  bands      serious n={bands['n_serious']}, routine n={bands['n_routine']}  "
                f"uplift={bands['median_uplift']*100:+6.1f}%  "
                f"effect={bands['effect_size']:+.3f}  p={bands['p_value']:.2e}"
            )
        if direction is None:
            print("  direction  insufficient events (diagnostic)")
        else:
            print(
                f"  direction  up n={direction['n_risk_up']} med={direction['median_up']:.3f} vs "
                f"down n={direction['n_risk_down']} med={direction['median_down']:.3f}  "
                f"p={direction['p_value']:.2e}  (diagnostic, non-gating)"
            )
    verdict = report["verdict"]
    print(
        f"\nverdict: rank={'PASS' if verdict['rank_pass'] else 'fail'}, "
        f"bands={'PASS' if verdict['bands_pass'] else 'fail'}"
    )
    print(
        "overall: "
        + (
            "PASS — severity unlocked as challenger feature in the harness race"
            if verdict["overall_pass"]
            else "FAIL — scores stay break-glass diagnostics only"
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="score_study")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--scores-path", type=Path, default=DEFAULT_SCORES_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered score study")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
