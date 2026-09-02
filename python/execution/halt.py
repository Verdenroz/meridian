"""The kill switch: a git-tracked flag file. Present means hold.

Hold, never flatten: flattening is the manual broker-side procedure, so the
switch can never itself become the runaway.
"""

from __future__ import annotations

from pathlib import Path

HALT_PATH = Path("data/execution/HALT")


def halted(path: Path = HALT_PATH) -> bool:
    return path.exists()
