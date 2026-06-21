"""Pack FDA synthetic .f32bin shards into WebDataset tar archives.

Each tar contains many per-window .f32bin entries keyed by name
`FDA_SYNTH_NNNNNNNN.f32bin` so the downstream ShardDataset's
SYNTHETIC_SHARD_PREFIX detection continues to trigger.

The output format is plain uncompressed tar (WebDataset-compatible):
sorted entries, stored without compression so the payload is already
float32 and would not benefit from gzip.

CLI:
    python -m python.tools.pack_fda_shards \\
        --src data/mmaps_synth/train \\
        --out /tmp/fda_tars \\
        --target-mib 100

    python -m python.tools.pack_fda_shards \\
        --src data/mmaps_synth/train \\
        --out /tmp/fda_tars \\
        --upload-repo user/meridian-data --upload-path-in-repo fda_tars
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PackResult:
    tar_paths: list[Path]
    window_counts: list[int]
    total_windows: int
    total_bytes: int


def _plan_groups(files: list[Path], target_bytes: int) -> list[list[Path]]:
    """Greedy group files into chunks of ~target_bytes each."""
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in files:
        size = path.stat().st_size
        if current and current_bytes + size > target_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def pack_shards(
    src_dir: Path,
    out_dir: Path,
    target_bytes: int = 100 * 1024 * 1024,
    prefix: str = "FDA_SYNTH_",
    tar_name_fmt: str = "fda-synth-{idx:05d}.tar",
) -> PackResult:
    """Tar shards in `src_dir` matching `prefix` into `out_dir`.

    Entries are renamed to a canonical zero-padded sequence so the consumer
    sees a stable ordering regardless of the source filename numbering.
    The FDA_SYNTH_ prefix is preserved on every entry.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source dir not found: {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.name.startswith(prefix) and p.suffix == ".f32bin")
    if not files:
        raise RuntimeError(f"no {prefix}*.f32bin files under {src_dir}")

    groups = _plan_groups(files, target_bytes)
    tar_paths: list[Path] = []
    window_counts: list[int] = []
    total_bytes = 0
    window_idx = 0
    for g_idx, group in enumerate(groups):
        tar_path = out_dir / tar_name_fmt.format(idx=g_idx)
        with tarfile.open(tar_path, "w") as tf:
            for src in group:
                entry_name = f"{prefix}{window_idx:08d}.f32bin"
                tf.add(src, arcname=entry_name)
                window_idx += 1
        tar_paths.append(tar_path)
        window_counts.append(len(group))
        total_bytes += tar_path.stat().st_size
    return PackResult(
        tar_paths=tar_paths,
        window_counts=window_counts,
        total_windows=window_idx,
        total_bytes=total_bytes,
    )


def write_manifest(
    out_dir: Path,
    result: PackResult,
    source_manifest: Path | None,
    extra: dict | None = None,
) -> Path:
    manifest: dict = {
        "created_unix": int(time.time()),
        "total_windows": result.total_windows,
        "total_bytes": result.total_bytes,
        "tars": [
            {"name": p.name, "windows": n, "bytes": p.stat().st_size}
            for p, n in zip(result.tar_paths, result.window_counts)
        ],
    }
    if source_manifest and source_manifest.is_file():
        with source_manifest.open() as f:
            src = json.load(f)
        manifest["source_config"] = src.get("config")
        manifest["source_acceptance_rate"] = src.get("acceptance_rate")
    if extra:
        manifest.update(extra)
    path = out_dir / "fda_manifest_tars.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return path


def upload_to_hf(
    out_dir: Path,
    repo_id: str,
    path_in_repo: str = "",
    private: bool = True,
) -> str:
    """Upload tars + manifest to an HF dataset repo. Returns repo URL."""
    from tools.hf_common import upload_folder_to_dataset

    return upload_folder_to_dataset(
        out_dir,
        repo_id,
        private=private,
        allow_patterns=["*.tar", "fda_manifest_tars.json"],
        path_in_repo=path_in_repo,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="directory of *.f32bin shards")
    ap.add_argument("--out", type=Path, required=True, help="output directory for tars")
    ap.add_argument("--target-mib", type=int, default=100, help="target tar size (MiB)")
    ap.add_argument("--prefix", type=str, default="FDA_SYNTH_", help="source filename prefix to pack")
    ap.add_argument("--source-manifest", type=Path, default=None, help="path to fda_manifest.json to copy config into new manifest")
    ap.add_argument("--upload-repo", type=str, default=None, help="HF dataset repo id to upload tars to (optional)")
    ap.add_argument("--upload-path-in-repo", type=str, default="", help="subdirectory within the repo to upload into (e.g. fda_tars)")
    ap.add_argument("--upload-private", action="store_true", help="create the HF dataset as private")
    ap.add_argument("--dry-run", action="store_true", help="plan only, skip packing and upload")
    args = ap.parse_args()

    target_bytes = int(args.target_mib) * 1024 * 1024

    if args.dry_run:
        files = sorted(p for p in args.src.iterdir() if p.is_file() and p.name.startswith(args.prefix))
        groups = _plan_groups(files, target_bytes)
        print(f"[dry-run] {len(files)} input files → {len(groups)} tars (target {args.target_mib} MiB each)")
        for i, g in enumerate(groups[:3]):
            print(f"  tar {i:05d}: {len(g)} windows, {sum(p.stat().st_size for p in g) / 1024 / 1024:.1f} MiB")
        if len(groups) > 3:
            print(f"  ... ({len(groups) - 3} more)")
        return

    t0 = time.time()
    result = pack_shards(args.src, args.out, target_bytes=target_bytes, prefix=args.prefix)
    dt = time.time() - t0
    print(
        f"[pack] wrote {len(result.tar_paths)} tars, "
        f"{result.total_windows} windows, "
        f"{result.total_bytes / 1024 / 1024:.1f} MiB in {dt:.1f}s"
    )

    manifest_path = write_manifest(args.out, result, args.source_manifest)
    print(f"[pack] manifest: {manifest_path}")

    if args.upload_repo:
        url = upload_to_hf(
            args.out,
            args.upload_repo,
            path_in_repo=args.upload_path_in_repo,
            private=args.upload_private,
        )
        print(f"[upload] {url}")


if __name__ == "__main__":
    main()
