from __future__ import annotations

import math
from dataclasses import dataclass

from train.feature_schema import PER_ASSET_FEATURE_DIM


@dataclass(frozen=True)
class AssetMinuteInput:
    price_volume: list[float]
    technical: list[float]
    options: list[float] | None
    fundamental: list[float]
    text: list[float]
    asset_calendar: list[float]
    asset_class: str


class FeatureAssembler:
    def __init__(self) -> None:
        self.options_defaults: dict[str, list[float]] = {
            "equity": [0.0, 1.0, 0.0, 1.0, 0.0],
            "etf": [0.0, 1.0, 0.0, 1.0, 0.0],
            "crypto": [0.0, 1.0, 0.0, 1.0, 0.0],
        }

    @staticmethod
    def _validate_block(name: str, values: list[float], expected_dim: int) -> None:
        if len(values) != expected_dim:
            raise ValueError(f"{name} dimension mismatch: {len(values)} != {expected_dim}")
        for value in values:
            if not math.isfinite(value):
                raise ValueError(f"{name} contains non-finite values")

    def assemble_asset(self, sample: AssetMinuteInput) -> list[float]:
        self._validate_block("price_volume", sample.price_volume, 10)
        self._validate_block("technical", sample.technical, 20)
        self._validate_block("fundamental", sample.fundamental, 10)
        self._validate_block("text", sample.text, 64)
        self._validate_block("asset_calendar", sample.asset_calendar, 3)

        options = sample.options
        if options is None:
            options = self.options_defaults.get(sample.asset_class)
            if options is None:
                raise ValueError(
                    f"Unknown asset_class={sample.asset_class!r}; expected one of "
                    f"{sorted(self.options_defaults.keys())}"
                )
        self._validate_block("options", options, 5)

        vector = (
            sample.price_volume
            + sample.technical
            + options
            + sample.fundamental
            + sample.text
            + sample.asset_calendar
        )
        if len(vector) != PER_ASSET_FEATURE_DIM:
            raise ValueError(
                f"assembled feature dim mismatch: {len(vector)} != {PER_ASSET_FEATURE_DIM}"
            )
        return vector

