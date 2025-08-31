"""Lightweight Gaussian source stub for CPU CI.

Generates a normalized Gaussian amplitude using numpy; no torch dependency
for the scaffold tests. Real implementation to be provided later.
"""

from __future__ import annotations

import numpy as np


class GaussianFiniteBand:
    """Minimal numpy-based Gaussian field generator for tests."""

    def __init__(self, waist_um: float = 10.0):
        self.waist_um = waist_um

    def prepare(self, cfg: dict | None = None, device: str = "cpu") -> None:  # noqa: ARG002
        return None

    def emit(self, shape: tuple[int, int]):
        ny, nx = shape
        y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        r2 = xx * xx + yy * yy
        g = np.exp(-r2 / max(self.waist_um, 1e-6))
        return g.astype(np.complex64)
