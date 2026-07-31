"""Deterministic train / validation / test split assignment (Section 10.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

Split = Literal["train", "validation", "test"]

# Boundaries are end-of-day inclusive (UTC)
_TRAIN_END = datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
_VAL_END = datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def assign_split(timestamp: datetime) -> Split:
    """Return the split for a given timestamp.

    Deterministic: depends only on the timestamp value, never on randomness.
    train:      2010-01-01 → 2022-12-31
    validation: 2023-01-01 → 2023-12-31
    test:       2024-01-01 → present
    """
    ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
    if ts <= _TRAIN_END:
        return "train"
    if ts <= _VAL_END:
        return "validation"
    return "test"


def is_train(timestamp: datetime) -> bool:
    return assign_split(timestamp) == "train"


def is_validation(timestamp: datetime) -> bool:
    return assign_split(timestamp) == "validation"


def is_test(timestamp: datetime) -> bool:
    return assign_split(timestamp) == "test"
