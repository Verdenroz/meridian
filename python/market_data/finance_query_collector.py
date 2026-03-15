"""Historical data collector — thin Python bridge to the Rust DataBus binary.

The Rust DataBus (src/main.rs) is authoritative for data collection. This module
provides a Python interface for the `meridian data collect` command to invoke it
for historical batch downloads during training corpus assembly.

Usage (historical mode):
  cargo run -- --symbol AAPL --start 2020-01-01 --end 2025-06-30

The binary emits JSONL rows to stdout matching the full v3.0 schema:
  price_volume (10), technical (42), options (12), fundamental (30),
  risk_analytics (8), event_state (10), cross_asset_context (8), text (64)

For live inference, the DataBus runs as a persistent Rust process — this module
is not involved.
"""

from __future__ import annotations

# TODO: implement Python bridge for historical collection
# (Section 6.1 of meridian-spec.md — DataBus historical mode)
