from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PhaseGrating:
    """Stub 1D phase grating along x."""

    period_norm: float = 0.1
    phase_rad: float = 1.0

    def transmission(self, shape: tuple[int, int]) -> np.ndarray:
        ny, nx = shape
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        phase = ((x / max(self.period_norm, 1e-6)) * 2 * np.pi) % (2 * np.pi)
        tx = np.exp(1j * (phase - np.pi) * (self.phase_rad / np.pi))
        return np.tile(tx[None, :], (ny, 1)).astype(np.complex64)
