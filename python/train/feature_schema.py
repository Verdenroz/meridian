"""Feature schema for Meridian v3.0 — full data integration.

Per-asset feature vector: 224 continuous + 64 text = 288 dims
Market-wide feature vector: 96 dims
Portfolio state vector: 201 dims
Calendar feature vector: 20 dims

Per-asset blocks:
  Block 1: Price & Volume          (10 dims)   — Section 6.2 Block 1
  Block 2: Technical Indicators    (42 dims)   — Section 6.2 Block 2 (expanded from 20)
  Block 3: Options-Derived         (12 dims)   — Section 6.2 Block 3 (expanded from 5)
  Block 4: Fundamental & Ownership (30 dims)   — Section 6.2 Block 4 (expanded from 10)
  Block 5: Risk Analytics          ( 8 dims)   — Section 6.2 Block 5 (new)
  Block 6: Event State             (10 dims)   — Section 6.2 Block 6 (new)
  Block 7: Cross-Asset Context     ( 8 dims)   — Section 6.2 Block 7 (new)
  Block 8: Text Embedding          (64 dims)   — Section 6.2 Block 8 (MarketReader)
  Total continuous (blocks 1–7):   224 dims
  Total with text (block 8):       288 dims

Market-wide blocks:
  Volatility & Sentiment           (12 dims)
  Yield Curve                      (12 dims)
  FRED Macro Series                (12 dims)
  Sector & Industry Performance    (16 dims)
  World Indices & Cross-Asset      (20 dims)
  Market Breadth & Flow            (12 dims)
  RSS & News Aggregate Signals     (12 dims)
  Total:                           96 dims
"""

from __future__ import annotations

# TODO: implement v3.0 feature schema (Section 6.2 and 6.3 of meridian-spec.md)

PER_ASSET_CONTINUOUS_DIM = 224  # blocks 1–7
PER_ASSET_TEXT_DIM = 64         # block 8
PER_ASSET_FEATURE_DIM = PER_ASSET_CONTINUOUS_DIM + PER_ASSET_TEXT_DIM  # 288

MARKET_FEATURE_DIM = 96
PORTFOLIO_STATE_DIM = 201
CALENDAR_FEATURE_DIM = 20
