"""Unit conversion utilities for microscope simulation.

All internal calculations use micrometers. This module provides
conversion functions for common units.
"""


def nm_to_um(value: float | int) -> float:
    """Convert nanometers to micrometers."""
    return float(value) / 1000.0


def um_to_nm(value: float | int) -> float:
    """Convert micrometers to nanometers."""
    return float(value) * 1000.0


def mm_to_um(value: float | int) -> float:
    """Convert millimeters to micrometers."""
    return float(value) * 1000.0


def um_to_mm(value: float | int) -> float:
    """Convert micrometers to millimeters."""
    return float(value) / 1000.0


def deg_to_rad(value: float | int) -> float:
    """Convert degrees to radians."""
    import math

    return float(value) * math.pi / 180.0


def rad_to_deg(value: float | int) -> float:
    """Convert radians to degrees."""
    import math

    return float(value) * 180.0 / math.pi


__all__ = [
    "nm_to_um",
    "um_to_nm",
    "mm_to_um",
    "um_to_mm",
    "deg_to_rad",
    "rad_to_deg",
]
