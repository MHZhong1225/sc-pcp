from __future__ import annotations

import platform
import sys
from collections.abc import Sequence

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        # Canonicalize aliases such as ``cuda`` to ``cuda:0``.  Besides making
        # metadata unambiguous, this prevents a caller from scheduling two
        # workers on the same physical GPU via ``cuda,cuda:0``.
        return torch.device("cuda", index)
    return device


def resolve_devices(requested: str | Sequence[str]) -> tuple[str, ...]:
    names = requested.split(",") if isinstance(requested, str) else list(requested)
    devices = tuple(str(resolve_device(name.strip())) for name in names if name.strip())
    if not devices:
        raise ValueError("at least one device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must be unique")
    return devices


def runtime_metadata(device: str | torch.device) -> dict[str, object]:
    resolved = resolve_device(str(device))
    metadata: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(resolved),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else 0
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_bytes": properties.total_memory,
            }
        )
    return metadata
