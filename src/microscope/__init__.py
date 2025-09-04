"""Microscope CLI package scaffold.

This package provides a thin CLI wrapper for configuration validation and inspection.
Physics solvers are not invoked in this package.
"""

__all__ = []

"""Microscope optical simulation package.

Simulate 3D vector electromagnetic fields in a microscope column and produce 
intensity images at arbitrary recorder planes. Prioritizes accuracy and determinism.
Single-GPU workstation target.
"""

__version__ = "0.1.0"

__all__ = [
    "cli",
    "core",
    "gpu",
    "physics",
    "io",
    "validate",
    "presets",
]
