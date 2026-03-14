from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any


def _parse_major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    if len(parts) < 2:
        return (0, 0)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (0, 0)


def _detect_flash_linear_attention() -> bool:
    return (
        importlib.util.find_spec("fla") is not None
        or importlib.util.find_spec("flash_linear_attention") is not None
    )


def _resolve_torch_module(torch_module: ModuleType | Any | None) -> ModuleType | Any | None:
    if torch_module is not None:
        return torch_module
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch


def build_environment_report(
    torch_module: ModuleType | Any | None = None,
    fla_available: bool | None = None,
) -> dict[str, object]:
    resolved_torch = _resolve_torch_module(torch_module)
    report: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "torch_available": False,
        "torch_version": None,
        "torch_compatible": False,
        "cuda_available": False,
        "cuda_device": None,
        "is_h100": False,
        "fla_available": False,
        "ready_for_training": False,
    }

    if resolved_torch is not None:
        torch_version = str(getattr(resolved_torch, "__version__", "0.0"))
        major, minor = _parse_major_minor(torch_version)
        cuda_available = bool(resolved_torch.cuda.is_available())
        cuda_device = (
            str(resolved_torch.cuda.get_device_name(0)) if cuda_available else None
        )
        is_h100 = bool(cuda_device and "H100" in cuda_device.upper())

        report.update(
            {
                "torch_available": True,
                "torch_version": torch_version,
                "torch_compatible": (major, minor) >= (2, 0),
                "cuda_available": cuda_available,
                "cuda_device": cuda_device,
                "is_h100": is_h100,
            }
        )

    fla_installed = _detect_flash_linear_attention() if fla_available is None else fla_available
    report["fla_available"] = fla_installed

    report["ready_for_training"] = bool(
        report["torch_available"]
        and report["torch_compatible"]
        and report["cuda_available"]
        and report["fla_available"]
    )
    return report


def validate_training_environment(
    require_h100: bool = True,
    torch_module: ModuleType | Any | None = None,
    fla_available: bool | None = None,
) -> dict[str, object]:
    report = build_environment_report(torch_module=torch_module, fla_available=fla_available)
    if not report["ready_for_training"]:
        raise RuntimeError("Training environment is not ready: missing torch/cuda/fla prerequisites")
    if require_h100 and not report["is_h100"]:
        raise RuntimeError("Training environment does not expose an NVIDIA H100 device")
    return report
