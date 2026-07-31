"""Exact k-NN over the state buffer: z-scored Euclidean distance, with an
exclusion window before the query and a minimum separation between analogs.

Standardization uses only buffer rows the query is allowed to see, so a live
query never conditions on its own future. The exclusion window keeps "last
Tuesday" out of the analog list and, kept at or above the outcome horizon,
guarantees every analog's forward window is already known history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_K = 5
EXCLUDE_RECENT = 63
MIN_SEPARATION = 21


@dataclass(frozen=True)
class Analog:
    index: int
    date: str
    distance: float


def retrieve_analogs(
    states: np.ndarray,
    dates: list[str],
    query_index: int,
    k: int = DEFAULT_K,
    exclude_recent: int = EXCLUDE_RECENT,
    min_separation: int = MIN_SEPARATION,
) -> list[Analog]:
    """Nearest historical analogs to states[query_index], nearest first."""
    query = states[query_index]
    if not bool(np.all(np.isfinite(query))):
        raise ValueError(f"state at {dates[query_index]} has missing features")
    limit = query_index - exclude_recent
    if limit <= 0:
        return []
    buffer = states[:limit]
    eligible = np.flatnonzero(np.all(np.isfinite(buffer), axis=1))
    if len(eligible) == 0:
        return []
    mean = buffer[eligible].mean(axis=0)
    std = buffer[eligible].std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z_buffer = (buffer[eligible] - mean) / std
    z_query = (query - mean) / std
    distances = np.sqrt(((z_buffer - z_query) ** 2).sum(axis=1))

    picked: list[Analog] = []
    for pos in np.argsort(distances, kind="stable"):
        index = int(eligible[pos])
        if any(abs(index - p.index) < min_separation for p in picked):
            continue
        picked.append(Analog(index=index, date=dates[index], distance=float(distances[pos])))
        if len(picked) == k:
            break
    return picked
