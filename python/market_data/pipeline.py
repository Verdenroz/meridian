"""Data pipeline orchestration.

meridian data plan output (per-asset manifest):
  {
    "asset": "AAPL",
    "asset_class": "equity",
    "per_asset_feature_dim": 288,          # 224 continuous + 64 text
    "market_feature_dim": 96,
    "portfolio_state_dim": 201,
    "calendar_feature_dim": 20,
    "blocks": {
      "price_volume": 10,
      "technical": 42,
      "options": 12,
      "fundamental": 30,
      "risk_analytics": 8,
      "event_state": 10,
      "cross_asset_context": 8,
      "text": 64
    },
    "data_sources": [...],                 # finance-query APIs required
    "update_frequencies": {...},           # per-block polling schedule
    "estimated_rows_per_year": 97500       # 250 trading days × 390 minutes
  }
"""

from __future__ import annotations

from train.feature_schema import (
    CALENDAR_FEATURE_DIM,
    MARKET_FEATURE_DIM,
    PER_ASSET_FEATURE_DIM,
    PORTFOLIO_STATE_DIM,
)

# TODO: implement pipeline plan generation and orchestration
