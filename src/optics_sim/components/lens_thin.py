from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ThinLens:
    """Thin lens applying a quadratic phase for focusing."""

    focal_length_um: float

    def transmission(self, shape: tuple[int, int], dx_um: float = 1.0) -> np.ndarray:
        """
        Calculate lens transmission function with proper phase.
        
        Args:
            shape: (ny, nx) grid dimensions
            dx_um: Grid spacing in microns (for physical scaling)
            
        Returns:
            Complex transmission function
        """
        ny, nx = shape
        
        # Create physical coordinate grid
        y = (np.arange(ny, dtype=np.float32) - ny//2) * dx_um
        x = (np.arange(nx, dtype=np.float32) - nx//2) * dx_um
        yy, xx = np.meshgrid(y, x, indexing="ij")
        
        # Wavelength (default 550nm for visible light)
        wavelength_um = 0.55
        k = 2 * np.pi / wavelength_um
        
        # Quadratic phase for focusing (negative for converging lens)
        r_squared = xx**2 + yy**2
        phase = -k * r_squared / (2 * self.focal_length_um)
        
        return np.exp(1j * phase).astype(np.complex64)
    
    def transmission_normalized(self, shape: tuple[int, int]) -> np.ndarray:
        """Legacy method for backward compatibility (normalized coords)."""
        ny, nx = shape
        y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        
        # Simple quadratic phase (keep original for compatibility)
        phase = (xx * xx + yy * yy) / max(self.focal_length_um, 1e-6)
        return np.exp(1j * phase).astype(np.complex64)