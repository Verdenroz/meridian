"""LLM event pipeline: 8-K filing scoring (roadmap step 7).

Unlocked by the event study (2026-07-31) and scoped by it: unscheduled 8-K
filings showed +6% 5-day forward-vol uplift beyond the EWMA forecast; news
volume showed nothing. So this pipeline scores 8-K *content* — severity,
scope, risk direction — via an LLM behind chimeric (local model default,
hosted models config swaps). Scores are versioned by (model, prompt) and
stored beside — never inside — the raw data. They enter the risk engine only
as challenger features through the walk-forward gate.
"""

from event_pipeline.scorer import PROMPT_VERSION, EventRiskScore, EventScorer

__all__ = ["PROMPT_VERSION", "EventRiskScore", "EventScorer"]
