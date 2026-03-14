from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .feature_assembly import AssetMinuteInput, FeatureAssembler
from train.feature_schema import PER_ASSET_FEATURE_DIM
from train.splits import resolve_split

_ROW_STRUCT = struct.Struct(f"<{PER_ASSET_FEATURE_DIM}f")


def _require_list(payload: dict[str, Any], key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{key}'")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric value found in '{key}'") from error


def _optional_list(payload: dict[str, Any], key: str) -> list[float] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Expected list or null for '{key}'")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric value found in '{key}'") from error


def _parse_row(payload: dict[str, Any]) -> tuple[datetime, AssetMinuteInput]:
    timestamp_raw = payload.get("timestamp")
    if isinstance(timestamp_raw, str):
        timestamp = datetime.fromisoformat(timestamp_raw)
    elif isinstance(timestamp_raw, (int, float)):
        timestamp = datetime.fromtimestamp(float(timestamp_raw))
    else:
        raise ValueError("Missing or invalid 'timestamp'")

    asset_class = payload.get("asset_class", "equity")
    if not isinstance(asset_class, str):
        raise ValueError("asset_class must be a string")

    sample = AssetMinuteInput(
        price_volume=_require_list(payload, "price_volume"),
        technical=_require_list(payload, "technical"),
        options=_optional_list(payload, "options"),
        fundamental=_require_list(payload, "fundamental"),
        text=_require_list(payload, "text"),
        asset_calendar=_require_list(payload, "asset_calendar"),
        asset_class=asset_class,
    )
    return timestamp, sample


def _write_binary_rows(path: Path, rows: list[list[float]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(_ROW_STRUCT.pack(*row))


def ingest_feature_rows(
    rows: Iterable[dict[str, Any]],
    asset_symbol: str,
    output_root: Path,
    input_descriptor: str = "in-memory rows",
) -> dict[str, Any]:
    if not asset_symbol:
        raise ValueError("asset_symbol cannot be empty")

    assembler = FeatureAssembler()
    split_rows: dict[str, list[list[float]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    processed_rows = 0
    for index, payload in enumerate(rows, start=1):
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict row at position {index}")
        timestamp, sample = _parse_row(payload)
        vector = assembler.assemble_asset(sample)
        split_name = resolve_split(timestamp)
        split_rows[split_name].append(vector)
        processed_rows += 1

    if processed_rows == 0:
        raise ValueError("No rows were provided for ingestion")

    shard_paths: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split_name, split_values in split_rows.items():
        split_counts[split_name] = len(split_values)
        if not split_values:
            continue

        split_dir = output_root / "data" / "mmaps" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_path = split_dir / f"{asset_symbol}.f32bin"
        _write_binary_rows(shard_path, split_values)
        shard_paths[split_name] = str(shard_path.resolve())

        metadata_path = split_dir / f"{asset_symbol}.meta.json"
        metadata = {
            "asset": asset_symbol,
            "split": split_name,
            "dtype": "float32",
            "feature_dim": PER_ASSET_FEATURE_DIM,
            "row_count": len(split_values),
            "bytes_per_row": _ROW_STRUCT.size,
            "data_file": str(shard_path.resolve()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "input_path": input_descriptor,
        "asset_symbol": asset_symbol,
        "feature_dim": PER_ASSET_FEATURE_DIM,
        "dtype": "float32",
        "bytes_per_row": _ROW_STRUCT.size,
        "split_counts": split_counts,
        "shard_paths": shard_paths,
    }

    manifest_path = output_root / "reports" / "ingestion" / f"{asset_symbol}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def ingest_feature_jsonl(
    input_path: Path,
    asset_symbol: str,
    output_root: Path,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not asset_symbol:
        raise ValueError("asset_symbol cannot be empty")

    jsonl_rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_number}")
        jsonl_rows.append(payload)

    return ingest_feature_rows(
        rows=jsonl_rows,
        asset_symbol=asset_symbol,
        output_root=output_root,
        input_descriptor=str(input_path.resolve()),
    )
