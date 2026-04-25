"""Upload a shard directory tree to a HuggingFace dataset repo.

Used by `make upload` to push real `.f32bin` shards under data/mmaps/ to
the configured HF_DATA_REPO. Authentication is read from the environment
(HF_TOKEN or HUGGING_FACE_HUB_TOKEN); the Makefile exports these from .env.

CLI:
    python -m tools.hf_upload_shards --src data/mmaps --repo user/my-data --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.hf_common import upload_folder_to_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="directory containing shards")
    ap.add_argument("--repo", type=str, required=True, help="HF dataset repo id (user/name)")
    ap.add_argument("--private", action="store_true", help="create repo as private")
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["**/*.f32bin"],
        help="glob patterns to include (relative to --src)",
    )
    args = ap.parse_args()

    try:
        url = upload_folder_to_dataset(
            args.src,
            args.repo,
            private=args.private,
            allow_patterns=args.patterns,
        )
    except RuntimeError as e:
        sys.stderr.write(f"{e}\n")
        raise SystemExit(2)
    print(f"✓ uploaded → {url}")


if __name__ == "__main__":
    main()
