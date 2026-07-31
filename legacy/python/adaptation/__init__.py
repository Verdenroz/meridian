"""Continuous learning system — 3-tier adaptation (Section 8 of meridian-spec.md)."""

from adaptation.adaptation import AdaptationConfig, AdaptationSystem
from adaptation.replay_buffer import ReplayBuffer

__all__ = ["AdaptationConfig", "AdaptationSystem", "ReplayBuffer"]
