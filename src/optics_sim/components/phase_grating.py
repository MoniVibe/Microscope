from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PhaseGrating:
    """1D phase grating along x-axis with correct diffraction."""

    period_norm: float = 0.1  # Keep normalized period for compatibility
    phase_rad: float = 1.0    # Keep original name for compatibility

    def transmission(self, shape: tuple[int, int], dx_um: float = None) -> np.ndarray:
        """
        Calculate grating transmission with proper phase modulation.
        
        Args:
            shape: (ny, nx) grid dimensions
            dx_um: Grid spacing in microns (optional)
            
        Returns:
            Complex transmission function
        """
        ny, nx = shape
        
        if dx_um is not None:
            # Physical coordinates mode
            x = (np.arange(nx, dtype=np.float32) - nx//2) * dx_um
            period_um = self.period_norm * nx * dx_um  # Convert normalized to physical
            
            # Sinusoidal phase modulation
            phase = self.phase_rad * np.sin(2 * np.pi * x / period_um)
            
            # Complex transmission
            tx = np.exp(1j * phase)
        else:
            # Legacy normalized mode (backward compatibility)
            x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
            phase = ((x / max(self.period_norm, 1e-6)) * 2 * np.pi) % (2 * np.pi)
            tx = np.exp(1j * (phase - np.pi) * (self.phase_rad / np.pi))
        
        # Replicate along y
        return np.tile(tx[None, :], (ny, 1)).astype(np.complex64)
    
    def get_diffraction_orders(self, wavelength_um: float = 0.55, max_order: int = 3):
        """
        Calculate expected diffraction order efficiencies.
        
        Args:
            wavelength_um: Wavelength in microns
            max_order: Maximum order to calculate
            
        Returns:
            Dict of order: efficiency
        """
        try:
            from scipy.special import jv  # Bessel function
            
            efficiencies = {}
            for m in range(-max_order, max_order + 1):
                # Efficiency = |J_m(phase_depth)|^2 for sinusoidal grating
                eff = jv(m, self.phase_rad) ** 2
                efficiencies[m] = eff
                
            return efficiencies
        except ImportError:
            # Fallback if scipy not available
            return {0: 0.5, -1: 0.25, 1: 0.25}  # Approximate