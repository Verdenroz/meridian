from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DataSplitRange:
    name: str
    start: datetime
    end: datetime

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp <= self.end


TRAIN_RANGE = DataSplitRange(
    name="train",
    start=datetime(2020, 1, 1),
    end=datetime(2024, 6, 30, 23, 59, 59),
)

VALIDATION_RANGE = DataSplitRange(
    name="validation",
    start=datetime(2024, 7, 1),
    end=datetime(2024, 12, 31, 23, 59, 59),
)

TEST_RANGE = DataSplitRange(
    name="test",
    start=datetime(2025, 1, 1),
    end=datetime(2025, 6, 30, 23, 59, 59),
)

ALL_SPLITS: tuple[DataSplitRange, ...] = (TRAIN_RANGE, VALIDATION_RANGE, TEST_RANGE)


def resolve_split(timestamp: datetime) -> str:
    for split in ALL_SPLITS:
        if split.contains(timestamp):
            return split.name
    raise ValueError("Timestamp is outside configured Meridian training split windows")

