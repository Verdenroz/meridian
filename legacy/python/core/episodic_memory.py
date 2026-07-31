"""Financial Episodic Memory — FAISS-backed key-value store (Section 7.7)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAISS_AVAILABLE = False


class EpisodicMemory(nn.Module):
    """Surprise-gated episodic memory with k-NN retrieval.

    Write policy:
      Write a new entry when aggregate_surprise > threshold (2σ).
      FIFO eviction when the buffer is full.

    Read policy:
      Retrieve k=8 nearest neighbours by L2 distance in the key space.
      The retrieved value tokens are appended to the cross-asset fusion input.

    FAISS index: IVFFlat with nlist=256, nprobe=32.

    Parameters: ~200K (key/value projection layers), FP16 at inference.
    """

    def __init__(
        self,
        d_model: int = 768,
        capacity: int = 50_000,
        k: int = 8,
        surprise_threshold: float = 2.0,
        nlist: int = 256,
        nprobe: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.k = k
        self.surprise_threshold = surprise_threshold
        self.nlist = nlist
        self.nprobe = nprobe

        # Learnable projections: backbone state → memory key / value
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        # Scalar gate for write: Linear(1 → 1) on surprise signal
        self.write_gate = nn.Linear(1, 1)

        # Runtime buffer — not nn.Parameters; managed manually
        self._keys:   np.ndarray | None = None  # (N, D) float32
        self._values: np.ndarray | None = None  # (N, D) float32
        self._write_ptr: int = 0
        self._size: int = 0
        self._index: faiss.Index | None = None  # type: ignore[name-defined]
        self._index_dirty: bool = True

    # ── Index management ─────────────────────────────────────────────────────

    def _init_buffers(self) -> None:
        self._keys   = np.zeros((self.capacity, self.d_model), dtype=np.float32)
        self._values = np.zeros((self.capacity, self.d_model), dtype=np.float32)

    def _rebuild_index(self) -> None:
        if not _FAISS_AVAILABLE:
            return
        quantizer = faiss.IndexFlatL2(self.d_model)
        index = faiss.IndexIVFFlat(quantizer, self.d_model, self.nlist, faiss.METRIC_L2)
        index.nprobe = self.nprobe
        if self._size >= self.nlist:
            keys = self._keys[: self._size]
            index.train(keys)
            index.add(keys)
        self._index = index
        self._index_dirty = False

    # ── Write ────────────────────────────────────────────────────────────────

    def write(
        self,
        state: torch.Tensor,    # (D,) — market-wide backbone state
        surprise: torch.Tensor, # () scalar surprise signal
    ) -> None:
        """Write entry if surprise exceeds threshold."""
        if self._keys is None:
            self._init_buffers()

        gate = torch.sigmoid(self.write_gate(surprise.unsqueeze(0))).item()
        normalised_surprise = (surprise / (self.surprise_threshold + 1e-8)).item()
        if normalised_surprise < 1.0:
            return  # below threshold — skip write

        with torch.no_grad():
            key_np   = self.key_proj(state).detach().cpu().float().numpy()
            value_np = self.value_proj(state).detach().cpu().float().numpy()

        idx = self._write_ptr % self.capacity
        self._keys[idx]   = key_np
        self._values[idx] = value_np
        self._write_ptr += 1
        self._size = min(self._size + 1, self.capacity)
        self._index_dirty = True

    # ── Read ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: torch.Tensor,  # (D,)
    ) -> torch.Tensor:        # (k, D) — retrieved value tokens
        """Return k nearest-neighbour value tokens for cross-asset fusion."""
        if self._size == 0 or self._keys is None:
            return torch.zeros(self.k, self.d_model, dtype=query.dtype, device=query.device)

        with torch.no_grad():
            q_np = self.key_proj(query).detach().cpu().float().numpy()

        if _FAISS_AVAILABLE:
            if self._index_dirty:
                self._rebuild_index()
            if self._index is not None and self._index.ntotal > 0:
                actual_k = min(self.k, self._index.ntotal)
                _, indices = self._index.search(q_np.reshape(1, -1), actual_k)
                indices = indices[0]
            else:
                indices = self._brute_force_nn(q_np)
        else:
            indices = self._brute_force_nn(q_np)

        retrieved = self._values[indices]  # (k, D) numpy
        result = torch.from_numpy(retrieved).to(dtype=query.dtype, device=query.device)
        # Pad to exactly k if fewer entries exist
        if result.shape[0] < self.k:
            pad = torch.zeros(
                self.k - result.shape[0], self.d_model, dtype=query.dtype, device=query.device
            )
            result = torch.cat([result, pad], dim=0)
        return result

    def _brute_force_nn(self, query: np.ndarray) -> np.ndarray:
        keys = self._keys[: self._size]  # (N, D)
        dists = np.sum((keys - query) ** 2, axis=1)
        k = min(self.k, self._size)
        return np.argpartition(dists, k - 1)[:k]
