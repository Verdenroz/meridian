from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegimeHMMConfig:
    n_states: int = 4
    covariance_type: str = "diag"
    n_iter: int = 200
    random_state: int = 42

    def validate(self) -> None:
        if self.n_states != 4:
            raise ValueError("Meridian spec requires exactly 4 regime states")
        if self.covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_type must be 'diag' or 'full'")


def build_hmm_training_spec(
    config: RegimeHMMConfig, feature_names: list[str]
) -> dict[str, object]:
    config.validate()
    if not feature_names:
        raise ValueError("feature_names cannot be empty")
    return {
        **asdict(config),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "regime_labels": [
            "low_vol_trending",
            "high_vol_trending",
            "mean_reverting",
            "crisis_dislocation",
        ],
    }


def load_regime_feature_rows(
    input_path: Path,
    feature_names: list[str],
    max_rows: int | None = None,
) -> list[list[float]]:
    if not input_path.exists():
        raise FileNotFoundError(f"HMM feature input not found: {input_path}")
    if not feature_names:
        raise ValueError("feature_names cannot be empty")

    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in HMM features at line {line_number}") from error

        row_values: list[float]
        if isinstance(payload, dict):
            missing = [name for name in feature_names if name not in payload]
            if missing:
                raise ValueError(
                    f"Missing HMM feature(s) {missing} at line {line_number}"
                )
            row_values = [float(payload[name]) for name in feature_names]
        elif isinstance(payload, list):
            if len(payload) != len(feature_names):
                raise ValueError(
                    f"HMM feature row length mismatch at line {line_number}: "
                    f"{len(payload)} != {len(feature_names)}"
                )
            row_values = [float(value) for value in payload]
        else:
            raise ValueError(f"Expected JSON object or list at line {line_number}")

        rows.append(row_values)
        if max_rows is not None and len(rows) >= max_rows:
            break

    if not rows:
        raise ValueError("No HMM feature rows found in input")
    return rows


def fit_regime_hmm(observations: Any, config: RegimeHMMConfig) -> Any:
    config.validate()
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "hmmlearn is required for HMM fitting. Install with: pip install '.[train]'"
        ) from error

    model = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        random_state=config.random_state,
    )
    return model.fit(observations)


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run_hmm_training_pipeline(
    config: RegimeHMMConfig,
    input_path: Path,
    feature_names: list[str],
    model_output_path: Path,
    meta_output_path: Path | None = None,
    max_rows: int | None = None,
) -> dict[str, object]:
    observations = load_regime_feature_rows(
        input_path=input_path,
        feature_names=feature_names,
        max_rows=max_rows,
    )
    model = fit_regime_hmm(observations, config)

    states = list(model.predict(observations))
    state_counts = {str(state): states.count(state) for state in range(config.n_states)}

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with model_output_path.open("wb") as handle:
        pickle.dump(model, handle)

    meta = {
        "input_path": str(input_path.resolve()),
        "feature_names": feature_names,
        "rows_used": len(observations),
        "n_states": config.n_states,
        "state_counts": state_counts,
        "transition_matrix": _to_list(getattr(model, "transmat_", [])),
        "start_probability": _to_list(getattr(model, "startprob_", [])),
        "means": _to_list(getattr(model, "means_", [])),
        "covariances": _to_list(getattr(model, "covars_", [])),
        "model_output_path": str(model_output_path.resolve()),
    }
    output_meta_path = meta_output_path or model_output_path.with_suffix(".meta.json")
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    meta["meta_output_path"] = str(output_meta_path.resolve())
    return meta

