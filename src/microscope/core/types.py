"""Type definitions and aliases for microscope simulation."""

from typing import Any

# Try to import torch types, fall back to generic types
try:
    import torch

    Tensor = torch.Tensor
    Device = torch.device
    DType = torch.dtype
except ImportError:
    Tensor = Any  # type: ignore
    Device = str  # type: ignore
    DType = Any  # type: ignore

# Common type aliases
GridSpec = dict[str, Any]
Field2D = dict[str, Any]  # Will be replaced with proper protocol later
Context = dict[str, Any]

# Coordinate types
Position3D = tuple[float, float, float]
Rotation3D = tuple[float, float, float]
Transform = dict[str, Any]

__all__ = [
    "Tensor",
    "Device",
    "DType",
    "GridSpec",
    "Field2D",
    "Context",
    "Position3D",
    "Rotation3D",
    "Transform",
]
