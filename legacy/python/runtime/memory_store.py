"""Recurrent state and FAISS store management — Section 15.

Manages all persistent in-memory state on the Pi 5:

  GDN hidden states:
    100 assets × 8 layers × D² state matrix (FP16)
    Size: ~75.5 MB
    Persisted to disk on clean shutdown; restored on startup.

  Temporal cascade states:
    3 cascade levels × 100 assets × D² (FP16)
    Size: ~16.4 MB

  Conv1d caches (temporal cascade boundary):
    Size: ~3.9 MB

  Episodic memory buffer:
    50K entries × D floats (FP16)
    FAISS IVF index (~6.4 MB)
    Write: surprise-gated; Read: k=8 nearest neighbors

  Rolling statistics:
    Per-asset per-feature (mean, variance) for z-score normalization
    42 indicators × 100 assets × 252-day window
    Size: ~8.0 MB

  Event state buffer:
    Per-asset recency and magnitude for 10 event-state features
    Size: ~2.0 MB

  Text embedding cache:
    100 assets × 64 dims × recent entries (time-decayed)
    Size: ~1.0 MB

Total normal inference: ~156 MB (well within 8 GB Pi 5 RAM)
"""

from __future__ import annotations

# TODO: implement memory store (Section 15 of meridian-spec.md)
