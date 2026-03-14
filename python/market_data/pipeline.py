from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from train.feature_schema import PER_ASSET_FEATURE_DIM
from train.splits import TEST_RANGE, TRAIN_RANGE, VALIDATION_RANGE


@dataclass(frozen=True)
class PipelineConfig:
    start_date: date
    end_date: date
    assets: list[str]
    output_root: Path = Path(".")
    trading_days_per_year: int = 252
    market_minutes_per_day: int = 390
    dtype: str = "float16"

    def validate(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        if not self.assets:
            raise ValueError("At least one asset is required")
        if len(self.assets) > 100:
            raise ValueError("Asset count exceeds Meridian max_assets=100")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be one of: float16, float32")


DATA_SOURCES: tuple[dict[str, str], ...] = (
    {"data_type": "minute_ohlcv_equities_etfs", "source": "Polygon.io/FirstRate", "granularity": "1-minute"},
    {"data_type": "minute_ohlcv_crypto", "source": "Binance/CoinGecko", "granularity": "1-minute"},
    {"data_type": "technical_indicators", "source": "finance-query indicators", "granularity": "1-minute"},
    {"data_type": "options_chains", "source": "CBOE/OptionMetrics", "granularity": "daily"},
    {"data_type": "fundamentals", "source": "finance-query + EDGAR XBRL", "granularity": "quarterly"},
    {"data_type": "news_headlines", "source": "finance-query RSS + GDELT", "granularity": "event-driven"},
    {"data_type": "earnings_transcripts", "source": "finance-query", "granularity": "quarterly"},
    {"data_type": "fred_macro", "source": "finance-query fred::series", "granularity": "daily-monthly"},
    {"data_type": "treasury_curve", "source": "finance-query fred::treasury_yields", "granularity": "daily"},
    {"data_type": "vix", "source": "finance-query ticker ^VIX", "granularity": "1-minute"},
    {"data_type": "fear_greed", "source": "finance-query", "granularity": "daily"},
    {"data_type": "sector_performance", "source": "finance-query finance::sector", "granularity": "daily"},
)


def _estimate_market_minutes(start_date: date, end_date: date, trading_days_per_year: int, market_minutes_per_day: int) -> int:
    years = (end_date - start_date).days / 365.25
    return int(years * trading_days_per_year * market_minutes_per_day)


def build_pipeline_plan(config: PipelineConfig) -> dict[str, object]:
    config.validate()
    estimated_minutes = _estimate_market_minutes(
        config.start_date,
        config.end_date,
        config.trading_days_per_year,
        config.market_minutes_per_day,
    )

    output_root = config.output_root.resolve()
    return {
        "date_range": {
            "start": config.start_date.isoformat(),
            "end": config.end_date.isoformat(),
        },
        "assets": config.assets,
        "asset_count": len(config.assets),
        "per_asset_feature_dim": PER_ASSET_FEATURE_DIM,
        "estimated_equity_market_minutes_per_asset": estimated_minutes,
        "storage_dtype": config.dtype,
        "data_sources": list(DATA_SOURCES),
        "split_windows": {
            "train": {
                "start": TRAIN_RANGE.start.isoformat(),
                "end": TRAIN_RANGE.end.isoformat(),
            },
            "validation": {
                "start": VALIDATION_RANGE.start.isoformat(),
                "end": VALIDATION_RANGE.end.isoformat(),
            },
            "test": {
                "start": TEST_RANGE.start.isoformat(),
                "end": TEST_RANGE.end.isoformat(),
            },
        },
        "output_paths": {
            "raw": str((output_root / "data/raw").resolve()),
            "processed": str((output_root / "data/processed").resolve()),
            "mmaps": str((output_root / "data/mmaps").resolve()),
            "reports": str((output_root / "reports").resolve()),
        },
    }


def write_pipeline_manifest(config: PipelineConfig, output_path: Path) -> Path:
    plan = build_pipeline_plan(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return output_path

