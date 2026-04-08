"""Replay buffer for Tier 3 consolidation — 7-day rolling window.

Stores (features, returns) tensors keyed by day.  FIFO eviction when
the buffer exceeds ``max_days``.  Provides a DataLoader view for the
Fisher estimator and LoRA fine-tuning in nightly consolidation.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:
    """Fixed-capacity ring buffer of daily (features, returns) tensors.

    Parameters
    ----------
    max_days : int
        Maximum number of day-keyed entries before oldest is evicted.
    """

    def __init__(self, max_days: int = 7) -> None:
        self.max_days = max_days
        # OrderedDict preserves insertion order for FIFO eviction
        self._store: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    # ── Mutation ─────────────────────────────────────────────────────────��───

    def add(self, day_key: str, features: torch.Tensor, returns: torch.Tensor) -> None:
        """Insert or replace a day's data, evicting the oldest if full.

        Parameters
        ----------
        day_key : str
            Unique identifier for the day, e.g. ``"2025-06-15"``.
        features : Tensor
            Shape ``(N_samples, ...)``.  Stored on CPU.
        returns : Tensor
            Shape ``(N_samples, ...)``.  Stored on CPU.
        """
        if day_key in self._store:
            del self._store[day_key]
        self._store[day_key] = (features.cpu(), returns.cpu())
        while len(self._store) > self.max_days:
            self._store.popitem(last=False)  # evict oldest

    def clear(self) -> None:
        self._store.clear()

    # ── Query ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def n_samples(self) -> int:
        return sum(f.shape[0] for f, _ in self._store.values())

    def memory_bytes(self) -> int:
        total = 0
        for feats, rets in self._store.values():
            total += feats.nelement() * feats.element_size()
            total += rets.nelement() * rets.element_size()
        return total

    def as_dataloader(self, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """Concatenate all stored days into a single DataLoader.

        Raises ``ValueError`` if the buffer is empty.
        """
        if not self._store:
            raise ValueError("ReplayBuffer is empty — cannot create DataLoader")

        all_feats = torch.cat([f for f, _ in self._store.values()], dim=0)
        all_rets  = torch.cat([r for _, r in self._store.values()], dim=0)
        return DataLoader(
            TensorDataset(all_feats, all_rets),
            batch_size=batch_size,
            shuffle=shuffle,
        )
