"""EventEncoder — encodes discrete financial events as D-dimensional tokens.

Handles 18 event categories (Section 6.4 of meridian-spec.md):
  INSIDER_BUY, INSIDER_SELL,
  ANALYST_UPGRADE, ANALYST_DOWNGRADE,
  EARNINGS_BEAT, EARNINGS_MISS,
  SEC_FILING_10K, SEC_FILING_10Q, SEC_FILING_8K, SEC_FILING_OTHER,
  FOMC_DECISION, MACRO_RELEASE,
  NEWS_HIGH_IMPACT,
  DIVIDEND_ANNOUNCED, SPLIT_ANNOUNCED,
  OPTIONS_UNUSUAL_ACTIVITY,
  TRENDING_ENTRY, RATING_CHANGE

Encoding:
  event_type_embedding:   Embedding(18 → 64)
  magnitude_projection:   Linear(1 → 64)
  asset_id_embedding:     Embedding(100 → 64)
  event_token = Linear(192 → D) applied to concat(type_emb, magnitude, asset_emb)
  event_token *= exp(-λ × minutes_since_event)

~50K parameters, FP16 precision.
"""

from __future__ import annotations

# TODO: implement EventEncoder (Section 6.4 of meridian-spec.md)
