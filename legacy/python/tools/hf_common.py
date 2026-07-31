"""Shared HuggingFace upload helpers.

Both `pack_fda_shards` (synth tars) and `hf_upload_shards` (real f32bin)
push a directory tree to a dataset repo. This module holds the one copy of
auth + create + upload logic so those two CLIs stay thin wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_hf_token(explicit: str | None = None) -> str:
    """Return the first non-empty HF token in this priority order:

        1. `explicit` argument
        2. HF_TOKEN environment variable
        3. HUGGING_FACE_HUB_TOKEN environment variable

    Raises RuntimeError with a clear message if none is set.
    """
    candidates = (
        explicit,
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    for t in candidates:
        if t:
            return t
    raise RuntimeError(
        "No HuggingFace token found. Set HF_TOKEN in .env, or export "
        "HUGGING_FACE_HUB_TOKEN, or pass one explicitly."
    )


def upload_folder_to_dataset(
    src: Path,
    repo: str,
    *,
    private: bool,
    allow_patterns: list[str],
    token: str | None = None,
    path_in_repo: str = "",
) -> str:
    """Create (or reuse) an HF dataset repo and upload a folder into it.

    Returns the dataset URL. Thin wrapper around `huggingface_hub` so the
    callers don't each have to do auth + create_repo + upload_folder.
    """
    from huggingface_hub import HfApi, create_repo

    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")
    tok = resolve_hf_token(token)

    create_repo(
        repo, repo_type="dataset", private=private, exist_ok=True, token=tok
    )
    HfApi().upload_folder(
        folder_path=str(src),
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        token=tok,
    )
    return f"https://huggingface.co/datasets/{repo}"
