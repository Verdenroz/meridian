"""Financial Episodic Memory — Section 7.7 of meridian-spec.md.

FAISS-backed key-value buffer of surprising historical market states.

Write policy (surprise-gated):
  A state is written to the buffer when aggregate_surprise > threshold (2σ).
  Buffer capacity: 50K entries (13 MB at FP16).
  Eviction: oldest entry when full.

Read policy (k-NN retrieval):
  At each forward pass, retrieve k=8 nearest neighbors by L2 distance.
  Similarity computed in D=768 embedding space (projected from backbone state).
  Retrieved values are appended as extra tokens to cross-asset fusion input.

FAISS index: IVF with nlist=256, nprobe=32 for approximate search.
Target retrieval latency: < 0.5ms on Pi 5 ARM.

Parameters: ~200K (projection layers), FP16.
"""

from __future__ import annotations

# TODO: implement FAISS-backed episodic memory (Section 7.7 of meridian-spec.md)
