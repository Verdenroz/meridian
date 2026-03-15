"""Temporal Cascade — Section 7.5 of meridian-spec.md.

Multi-resolution GDN stack with surprise-gated downward injection.

Three cascade levels (all RA-GDN-based):
  Level 0: minute-scale  — runs every minute (main backbone)
  Level 1: hourly-scale  — updates every 60 steps, receives downsampled state
  Level 2: daily-scale   — updates every 390 steps (full trading day)
  Level 3: weekly-scale  — updates every ~1950 steps

Surprise-gated injection (downward):
  At each level boundary, the slower context is injected into the faster stream
  gated by the surprise signal: injection_gate = sigmoid(W · surprise_signal)

  This means: if the current minute is surprising (large prediction error),
  the longer-horizon context is injected more strongly to help reorient.

Parameters: ~4.2M, ternary.
"""

from __future__ import annotations

# TODO: implement temporal cascade (Section 7.5 of meridian-spec.md)
