"""Deterministic train / validation / test split definitions.

Split boundaries (Section 10.1 of meridian-spec.md):
  train:       2010-01-01 → 2022-12-31
  validation:  2023-01-01 → 2023-12-31
  test:        2024-01-01 → present

Rules:
  - Splits are defined by timestamp only — no randomness.
  - A row with timestamp T belongs to exactly one split.
  - The split assignment is deterministic and must not change between runs.
  - No data from validation or test may be used during training or QAT.
"""

from __future__ import annotations

# TODO: implement split assignment function (Section 10.1 of meridian-spec.md)
