from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CircularAperture:
    """Stub circular aperture with unit radius in normalized coordinates."""

    radius_norm: float = 0.5

    def transmission(self, shape: tuple[int, int]) -> np.ndarray:
        ny, nx = shape
        y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        mask = (xx * xx + yy * yy) <= (self.radius_norm**2)
        return mask.astype(np.complex64)
