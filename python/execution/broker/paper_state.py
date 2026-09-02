"""Paper account state: an atomic JSON snapshot plus an append-only journal.

The journal is the record and is git-tracked like the shadow log; never edit
rows. The snapshot is convenience state derived from the same events.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from execution.orders import Order, Position

DEFAULT_PAPER_DIR = Path("data/execution/paper")


@dataclass(frozen=True)
class Settlement:
    settle_date: str
    amount: float


@dataclass(frozen=True)
class PaperState:
    cash: float
    unsettled: tuple[Settlement, ...]
    positions: tuple[Position, ...]
    orders: tuple[Order, ...]
    next_seq: int

    @staticmethod
    def initial(cash: float) -> PaperState:
        return PaperState(cash, (), (), (), 1)

    def position(self, symbol: str) -> Position:
        return next(
            (p for p in self.positions if p.symbol == symbol), Position(symbol, 0, 0.0)
        )

    def with_position(self, position: Position) -> PaperState:
        others = tuple(p for p in self.positions if p.symbol != position.symbol)
        kept = others + ((position,) if position.qty else ())
        return replace(self, positions=tuple(sorted(kept, key=lambda p: p.symbol)))

    def with_order(self, order: Order) -> PaperState:
        others = tuple(o for o in self.orders if o.id != order.id)
        return replace(self, orders=others + (order,))

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "unsettled": [s.__dict__ for s in self.unsettled],
            "positions": [p.__dict__ for p in self.positions],
            "orders": [o.to_dict() for o in self.orders],
            "next_seq": self.next_seq,
        }

    @staticmethod
    def from_dict(d: dict) -> PaperState:
        return PaperState(
            cash=float(d["cash"]),
            unsettled=tuple(Settlement(**s) for s in d["unsettled"]),
            positions=tuple(Position(**p) for p in d["positions"]),
            orders=tuple(Order(**o) for o in d["orders"]),
            next_seq=int(d["next_seq"]),
        )


class PaperStore:
    def __init__(self, directory: Path = DEFAULT_PAPER_DIR) -> None:
        self.directory = directory
        self.state_path = directory / "state.json"
        self.journal_path = directory / "journal.jsonl"

    def exists(self) -> bool:
        return self.state_path.is_file()

    def load(self) -> PaperState:
        return PaperState.from_dict(json.loads(self.state_path.read_text()))

    def save(self, state: PaperState) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.state_path)

    def journal(self, row: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    def journal_rows(self) -> list[dict]:
        if not self.journal_path.is_file():
            return []
        with self.journal_path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]
