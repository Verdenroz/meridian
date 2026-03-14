from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence


def default_rust_collector_command_template() -> tuple[str, ...]:
    return (
        "cargo",
        "run",
        "--quiet",
        "--",
        "--symbol",
        "{symbol}",
        "--start",
        "{start}",
        "--end",
        "{end}",
    )


@dataclass(frozen=True)
class FinanceQueryCollectorConfig:
    command_template: tuple[str, ...] = default_rust_collector_command_template()
    timeout_seconds: int = 300

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.command_template:
            raise ValueError(
                "command_template cannot be empty. "
                "Provide a Rust collector command that outputs schema-compatible JSONL rows."
            )


def build_collect_command(
    symbol: str,
    start_date: date,
    end_date: date,
    command_template: Sequence[str],
) -> list[str]:
    start_token = start_date.isoformat()
    end_token = end_date.isoformat()
    return [
        token.replace("{symbol}", symbol).replace("{start}", start_token).replace("{end}", end_token)
        for token in command_template
    ]


def _parse_jsonl_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"finance-query output contained invalid JSON on line {line_number}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"finance-query output line {line_number} is not a JSON object")
        rows.append(payload)
    if not rows:
        raise RuntimeError("finance-query returned no JSONL rows")
    return rows


def collect_feature_rows(
    symbol: str,
    start_date: date,
    end_date: date,
    config: FinanceQueryCollectorConfig | None = None,
) -> list[dict[str, Any]]:
    collector_config = config or FinanceQueryCollectorConfig()
    collector_config.validate()
    command = build_collect_command(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        command_template=collector_config.command_template,
    )

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=collector_config.timeout_seconds,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"collector executable not found while running: {' '.join(command)}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"finance-query command timed out after {collector_config.timeout_seconds}s"
        ) from error

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"finance-query command failed (exit {result.returncode}): {stderr or 'no stderr output'}"
        )

    return _parse_jsonl_rows(result.stdout)
