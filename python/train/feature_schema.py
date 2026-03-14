from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    features: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.features)


PRICE_VOLUME_BLOCK = FeatureBlock(
    name="price_volume",
    features=(
        "open_norm",
        "high_norm",
        "low_norm",
        "close_norm",
        "volume_norm",
        "return_1m",
        "return_5m",
        "return_15m",
        "return_60m",
        "return_1d",
    ),
)

TECHNICAL_BLOCK = FeatureBlock(
    name="technical",
    features=(
        "sma20_z",
        "sma50_z",
        "sma200_z",
        "adx14",
        "rsi14",
        "macd_hist_atr_norm",
        "stochastic_k_14_3_3",
        "roc10",
        "atr14_price_norm",
        "bb_percent_b_20_2",
        "keltner_position",
        "rv_ratio_5d_20d",
        "obv_5_slope",
        "cmf20",
        "mfi14",
        "vwap_deviation",
        "supertrend_direction",
        "ichimoku_cloud_position",
        "parabolic_sar_direction",
        "choppiness14",
    ),
)

OPTIONS_BLOCK = FeatureBlock(
    name="options",
    features=(
        "iv_30d_agg",
        "put_call_volume_ratio",
        "iv_25d_skew",
        "iv_term_structure_slope",
        "open_interest_change_norm",
    ),
)

FUNDAMENTAL_BLOCK = FeatureBlock(
    name="fundamental",
    features=(
        "trailing_pe_pct_rank",
        "forward_pe_pct_rank",
        "dividend_yield",
        "market_cap_log_pct_rank",
        "beta",
        "earnings_growth_winsor",
        "revenue_growth_winsor",
        "profit_margin",
        "debt_to_equity_pct_rank",
        "sector_scalar_embedding",
    ),
)

TEXT_BLOCK = FeatureBlock(
    name="text",
    features=tuple(f"text_embedding_{index}" for index in range(64)),
)

ASSET_CALENDAR_BLOCK = FeatureBlock(
    name="asset_calendar",
    features=(
        "days_to_next_earnings_inv",
        "days_to_next_ex_dividend_inv",
        "is_earnings_week",
    ),
)

PER_ASSET_BLOCKS: tuple[FeatureBlock, ...] = (
    PRICE_VOLUME_BLOCK,
    TECHNICAL_BLOCK,
    OPTIONS_BLOCK,
    FUNDAMENTAL_BLOCK,
    TEXT_BLOCK,
    ASSET_CALENDAR_BLOCK,
)

PER_ASSET_FEATURE_DIM = sum(block.dim for block in PER_ASSET_BLOCKS)
MARKET_FEATURE_DIM = 24
PORTFOLIO_STATE_DIM = 201
CALENDAR_FEATURE_DIM = 20


def validate_feature_schema() -> None:
    expected_dim = 112
    if PER_ASSET_FEATURE_DIM != expected_dim:
        raise ValueError(
            f"Per-asset feature dimension mismatch: "
            f"{PER_ASSET_FEATURE_DIM} != {expected_dim}"
        )

