"""Re-race log invariants: JSON-safe rows and per-month idempotency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rerace.job import compact_row, logged_months
from shadow.log import append_decision


def fake_report() -> dict:
    return {
        "verdict": {
            "race_pass": np.bool_(True),
            "null_sharpe": np.float64(0.49),
            "challenger_sharpe": np.float64(0.57),
        },
        "metrics": {
            "null_voltarget": {"sharpe": np.float64(0.49), "max_drawdown": -0.42},
            "brake": {"sharpe": 0.57, "max_drawdown": np.float64(-0.38)},
        },
    }


def test_compact_row_serializes_numpy_scalars() -> None:
    row = compact_row("eightk_race", fake_report(), "2026-08-01T22:15:00+00:00")
    encoded = json.dumps(row)  # raises if numpy types leak through
    decoded = json.loads(encoded)
    assert decoded["race_pass"] is True
    assert decoded["metrics"]["brake"]["max_drawdown"] == -0.38


def test_logged_months_keys_by_month_and_race(tmp_path: Path) -> None:
    log = tmp_path / "reraces.jsonl"
    assert logged_months(log) == set()
    append_decision(
        log, compact_row("eightk_race", fake_report(), "2026-08-01T22:15:00+00:00")
    )
    append_decision(
        log, compact_row("credit_race", fake_report(), "2026-08-01T22:20:00+00:00")
    )
    assert logged_months(log) == {
        ("2026-08", "eightk_race"),
        ("2026-08", "credit_race"),
    }
