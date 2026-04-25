"""Pre-flight check: refuse to augment if any real shard is too short.

`load_windows_from_shards` silently drops shards with fewer rows than
`window_len`, which means an undersized real shard can slip through
`augment-data` → training and only surface as a confusing acceptance-rate
drop much later. Fail here instead, with filenames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train.feature_schema import MARKET_FEATURE_DIM, PER_ASSET_FEATURE_DIM

ROW_DIM = PER_ASSET_FEATURE_DIM + MARKET_FEATURE_DIM  # 280 floats per row
BYTES_PER_ROW = ROW_DIM * 4                            # f32


def count_rows(path: Path) -> int:
    size = path.stat().st_size
    if size % BYTES_PER_ROW != 0:
        raise ValueError(
            f"{path.name} byte size ({size}) is not a multiple of row size "
            f"({BYTES_PER_ROW}); shard is malformed."
        )
    return size // BYTES_PER_ROW


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True, help="directory of *.f32bin shards")
    ap.add_argument("--min-rows", type=int, default=65,
                    help="reject any shard with fewer rows than this (default: seq_len+1 = 65)")
    args = ap.parse_args()

    if not args.dir.is_dir():
        sys.stderr.write(f"not a directory: {args.dir}\n")
        raise SystemExit(2)

    shards = sorted(args.dir.glob("*.f32bin"))
    if not shards:
        print(f"[shard-check] no *.f32bin shards under {args.dir} — nothing to check.")
        return

    offenders: list[tuple[Path, int]] = []
    for p in shards:
        try:
            rows = count_rows(p)
        except ValueError as e:
            sys.stderr.write(f"{e}\n")
            raise SystemExit(1)
        if rows < args.min_rows:
            offenders.append((p, rows))

    if offenders:
        sys.stderr.write(
            f"[shard-check] {len(offenders)}/{len(shards)} shards under "
            f"--min-rows={args.min_rows}:\n"
        )
        for p, rows in offenders[:20]:
            sys.stderr.write(f"  {p.name}: {rows} rows\n")
        if len(offenders) > 20:
            sys.stderr.write(f"  ... ({len(offenders) - 20} more)\n")
        sys.stderr.write(
            "These shards cannot produce any training windows at seq_len=64. "
            "Fix or remove before merging into the augmented dataset.\n"
        )
        raise SystemExit(1)
    print(f"[shard-check] all {len(shards)} shards ≥ {args.min_rows} rows. OK.")


if __name__ == "__main__":
    main()
