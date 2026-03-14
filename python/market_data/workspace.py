from __future__ import annotations

from pathlib import Path


WORKSPACE_DIRS: tuple[str, ...] = (
    "data/raw",
    "data/processed",
    "data/mmaps",
    "checkpoints",
    "logs",
    "reports",
    "experiments",
)


def initialize_workspace(root: Path) -> list[Path]:
    created: list[Path] = []
    for relative_dir in WORKSPACE_DIRS:
        directory = root / relative_dir
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return created

