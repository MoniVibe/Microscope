from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(slots=True)
class ThinLens:
    """Stub thin lens applying a quadratic phase (no units enforcement here)."""

    focal_length_um: float

    def transmission(self, shape: Tuple[int, int]) -> np.ndarray:
        ny, nx = shape
        y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        phase = (xx * xx + yy * yy) / max(self.focal_length_um, 1e-6)
        return np.exp(1j * phase).astype(np.complex64)


