"""JSONL dataset ingestion — raw rows → memory-mapped f32 shards.

Input JSONL schema (one row per asset-minute):
  {
    "timestamp": "2024-06-30T10:00:00",
    "asset_class": "equity" | "etf" | "crypto",
    "price_volume":        float[10],
    "technical":           float[42],
    "options":             float[12] | null,   # null → use asset-class default
    "fundamental":         float[30],
    "risk_analytics":      float[8],
    "event_state":         float[10],
    "cross_asset_context": float[8],
    "text":                float[64]
  }

Output: binary shard files at data/mmaps/{split}/{asset}.f32bin
  Format: raw little-endian f32 values, 288 floats per row, no header.
  Struct: struct.Struct("<288f")

Split assignment delegated to train.splits.
Ingestion manifest written to reports/ingestion/{asset}.json.
"""

from __future__ import annotations

import struct

from train.feature_schema import PER_ASSET_FEATURE_DIM

ROW_STRUCT = struct.Struct(f"<{PER_ASSET_FEATURE_DIM}f")

# TODO: implement JSONL ingestion and shard writing
# (Section 10.1 of meridian-spec.md)
