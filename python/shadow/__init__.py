"""Daily shadow decisions: the append-only paper-trading record."""

from shadow.job import run_decision
from shadow.log import DEFAULT_LOG_PATH, JOB_VERSION

__all__ = ["DEFAULT_LOG_PATH", "JOB_VERSION", "run_decision"]
