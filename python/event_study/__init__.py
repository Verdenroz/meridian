"""Event study: do filing/news signals precede elevated realized volatility?

This is the §5 gate — all LLM/text pipeline investment beyond the archive
waits on this result, and news-flow intensity as a regime feature depends on
it too. The pass/fail criteria are pre-registered in study.py as constants,
set before any result was computed.
"""

from event_study.study import GATE, run_study

__all__ = ["GATE", "run_study"]
