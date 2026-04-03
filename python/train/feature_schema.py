"""Feature dimension contracts for Meridian v3.0 (Sections 6.2 and 6.3)."""

from __future__ import annotations

from typing import NamedTuple


# ── Per-asset feature blocks ─────────────────────────────────────────────────

class FeatureBlock(NamedTuple):
    name: str
    start: int   # inclusive offset into the per-asset continuous vector
    dim: int

    @property
    def end(self) -> int:
        return self.start + self.dim

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


BLOCK_PRICE_VOLUME     = FeatureBlock("price_volume",        0,   10)
BLOCK_TECHNICAL        = FeatureBlock("technical",          10,   42)
BLOCK_OPTIONS          = FeatureBlock("options",            52,   12)
BLOCK_FUNDAMENTAL      = FeatureBlock("fundamental",        64,   30)
BLOCK_RISK_ANALYTICS   = FeatureBlock("risk_analytics",     94,    8)
BLOCK_EVENT_STATE      = FeatureBlock("event_state",       102,   10)
BLOCK_CROSS_ASSET      = FeatureBlock("cross_asset_context",112,   8)
BLOCK_TEXT             = FeatureBlock("text",              120,   64)  # offset into 184-dim vector

PER_ASSET_BLOCKS: tuple[FeatureBlock, ...] = (
    BLOCK_PRICE_VOLUME,
    BLOCK_TECHNICAL,
    BLOCK_OPTIONS,
    BLOCK_FUNDAMENTAL,
    BLOCK_RISK_ANALYTICS,
    BLOCK_EVENT_STATE,
    BLOCK_CROSS_ASSET,
)

# ── Market-wide feature blocks ───────────────────────────────────────────────

class MarketBlock(NamedTuple):
    name: str
    start: int
    dim: int

    @property
    def end(self) -> int:
        return self.start + self.dim

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


MBLOCK_VOLATILITY_SENTIMENT   = MarketBlock("volatility_sentiment",     0,  12)
MBLOCK_YIELD_CURVE             = MarketBlock("yield_curve",             12,  12)
MBLOCK_FRED_MACRO              = MarketBlock("fred_macro",              24,  12)
MBLOCK_SECTOR_INDUSTRY         = MarketBlock("sector_industry",         36,  16)
MBLOCK_WORLD_INDICES           = MarketBlock("world_indices_cross_asset",52, 20)
MBLOCK_BREADTH_FLOW            = MarketBlock("breadth_flow",            72,  12)
MBLOCK_RSS_NEWS                = MarketBlock("rss_news_aggregate",      84,  12)

MARKET_BLOCKS: tuple[MarketBlock, ...] = (
    MBLOCK_VOLATILITY_SENTIMENT,
    MBLOCK_YIELD_CURVE,
    MBLOCK_FRED_MACRO,
    MBLOCK_SECTOR_INDUSTRY,
    MBLOCK_WORLD_INDICES,
    MBLOCK_BREADTH_FLOW,
    MBLOCK_RSS_NEWS,
)

# ── Scalar dimension constants ───────────────────────────────────────────────

PER_ASSET_CONTINUOUS_DIM: int = 120   # blocks 1–7
PER_ASSET_TEXT_DIM: int       = 64    # block 8 (MarketReader embedding)
PER_ASSET_FEATURE_DIM: int    = PER_ASSET_CONTINUOUS_DIM + PER_ASSET_TEXT_DIM  # 184

MARKET_FEATURE_DIM: int    = 96
PORTFOLIO_STATE_DIM: int   = 201
CALENDAR_FEATURE_DIM: int  = 20

# ── Sanity checks (evaluated at import time) ─────────────────────────────────

_continuous_total = sum(b.dim for b in PER_ASSET_BLOCKS)
assert _continuous_total == PER_ASSET_CONTINUOUS_DIM, (
    f"per-asset continuous dims sum to {_continuous_total}, expected {PER_ASSET_CONTINUOUS_DIM}"
)

_market_total = sum(b.dim for b in MARKET_BLOCKS)
assert _market_total == MARKET_FEATURE_DIM, (
    f"market dims sum to {_market_total}, expected {MARKET_FEATURE_DIM}"
)
