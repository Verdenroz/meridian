"""CLI: run the pre-registered tone-brake race.

    uv run python -m tone_race run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.racing import metrics_table, plain
from market_data.store import DEFAULT_STORE_DIR

from tone_race.race import CHALLENGERS, run_race


def cmd_run(args: argparse.Namespace) -> int:
    report = run_race(args.store_dir)
    gate = report["gate"]
    lo, hi = report["oos_span"]
    active = report["active_days"]
    print(
        f"pre-registered race: tone brake (z ≥ {gate['burst_z']}, "
        f"scale {gate['brake_scale']}, {gate['brake_days']}d) vs untouched "
        f"vol-target null; OOS {lo} → {hi}, after costs"
    )
    print(
        f"brake active (of {active['span_days']} OOS days): "
        f"tone {active['tone']}, credit {active['credit']}, "
        f"union {active['union']}\n"
    )
    print(metrics_table(report["metrics"], label_width=32))
    signal = report["signal"]
    print(
        f"\nsignal diagnostic (non-gating): n={signal['n_days']}  "
        f"spearman={signal['spearman']:+.3f}  p={signal['p_value']:.2e}"
    )
    verdict = report["verdict"]
    for label in CHALLENGERS:
        v = verdict["per_challenger"][label]
        status = f"PASS route {v['route']}" if v["passes"] else "FAIL"
        print(f"\n{label}: {status}")
        print(f"  {json.dumps(plain(v))}")
    print(f"\nrace verdict: {'PASS' if verdict['race_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tone_race")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pre-registered tone-brake race")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
