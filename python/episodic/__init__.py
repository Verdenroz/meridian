"""Episodic/analog memory: match today's market state against decades of
daily history via exact k-NN, and explain it ("nearest analogs: Feb 2018").

Analogs are explanation plus a *reported* nonparametric prior — they do not
touch allocation. Wiring analogs into the allocator would be a new
challenger through the walk-forward gate, pre-registered like every other.
"""

from episodic.briefing import build_briefing, format_briefing
from episodic.knn import Analog, retrieve_analogs
from episodic.outcomes import forward_outcomes
from episodic.state import STATE_FEATURES, STATE_SYMBOLS, build_state_matrix

__all__ = [
    "STATE_FEATURES",
    "STATE_SYMBOLS",
    "Analog",
    "build_briefing",
    "build_state_matrix",
    "format_briefing",
    "forward_outcomes",
    "retrieve_analogs",
]
