"""Validation metrics for optical propagation simulations.

Provides quantitative metrics for assessing simulation accuracy
including L2 error, energy conservation, Strehl ratio, and MTF.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import numpy as np


def l2_field_error(field_computed: torch.Tensor,
                   field_reference: torch.Tensor,
                   normalize: bool = True) -> float:
    """Compute L2 error between computed and reference fields.
    
    Args:
        field_computed: Computed complex field
        field_reference: Reference complex field
        normalize: If True, normalize by reference norm
        
    Returns:
        L2 error (relative if normalized)
    """
    # Ensure same shape
    if field_computed.shape != field_reference.shape:
        raise ValueError(f"Shape mismatch: {field_computed.shape} vs {field_reference.shape}")
    
    # Compute L2 norm of difference
    diff = field_computed - field_reference
    error_norm = torch.norm(diff.flatten(), p=2)
    
    if normalize:
        ref_norm = torch.norm(field_reference.flatten(), p=2)
        if ref_norm > 0:
            return (error_norm / ref_norm).item()
        else:
            return error_norm.item()
    else:
        return error_norm.item()


def energy_conservation(field_in: torch.Tensor,
                       field_out: torch.Tensor,
                       dx: float = 1.0,
                       dy: float = 1.0) -> float:
    """Check energy conservation through propagation.
    
    Args:
        field_in: Input complex field
        field_out: Output complex field
        dx, dy: Grid spacing for integration
        
    Returns:
        Relative energy change (should be < 1% for good propagation)
    """
    # Compute total energy (intensity integrated over area)
    energy_in = torch.sum(torch.abs(field_in)**2).item() * dx * dy
    energy_out = torch.sum(torch.abs(field_out)**2).item() * dx * dy
    
    if energy_in > 0:
        rel_change = abs(energy_out - energy_in) / energy_in
    else:
        rel_change = 0.0
    
    return rel_change


def strehl_ratio(psf: torch.Tensor,
                 psf_ideal: Optional[torch.Tensor] = None,
                 normalize: bool = True) -> float:
    """Compute Strehl ratio for optical system quality.
    
    Strehl ratio is the ratio of peak intensity to ideal peak intensity.
    Values > 0.8 indicate diffraction-limited performance.
    
    Args:
        psf: Computed PSF (intensity)
        psf_ideal: Ideal/reference PSF (if None, use peak of input)
        normalize: Normalize PSFs before comparison
        
    Returns:
        Strehl ratio (0 to 1)
    """
    # Ensure positive (intensity)
    if torch.is_complex(psf):
        psf = torch.abs(psf)**2
    
    if normalize:
        psf = psf / torch.sum(psf)
    
    peak_computed = torch.max(psf).item()
    
    if psf_ideal is not None:
        if torch.is_complex(psf_ideal):
            psf_ideal = torch.abs(psf_ideal)**2
        
        if normalize:
            psf_ideal = psf_ideal / torch.sum(psf_ideal)
        
        peak_ideal = torch.max(psf_ideal).item()
    else:
        # Use theoretical maximum (all energy at one point)
        total_energy = torch.sum(psf).item()
        peak_ideal = total_energy  # If all energy were concentrated
    
    if peak_ideal > 0:
        strehl = peak_computed / peak_ideal
    else:
        strehl = 0.0
    
    return min(strehl, 1.0)  # Cap at 1.0


def mtf_cutoff(field: torch.Tensor,
               dx: float,
               dy: float,
               threshold: float = 0.1) -> Tuple[float, float]:
    """Compute MTF cutoff frequency.
    
    The cutoff frequency where MTF drops below threshold indicates
    the resolution limit of the optical system.
    
    Args:
        field: Complex field or PSF
        dx, dy: Grid spacing
        threshold: MTF threshold for cutoff (typically 0.1)
        
    Returns:
        Tuple of (fx_cutoff, fy_cutoff) in cycles/µm
    """
    # Compute MTF (magnitude of OTF)
    if torch.is_complex(field):
        intensity = torch.abs(field)**2
    else:
        intensity = field
    
    # FFT to get OTF
    otf = torch.fft.fft2(intensity)
    otf = torch.fft.fftshift(otf)
    
    # Normalize
    mtf = torch.abs(otf)
    mtf = mtf / torch.max(mtf)
    
    # Find cutoff frequencies
    ny, nx = mtf.shape
    fx = torch.fft.fftfreq(nx, d=dx)
    fy = torch.fft.fftfreq(ny, d=dy)
    fx = torch.fft.fftshift(fx)
    fy = torch.fft.fftshift(fy)
    
    # Find where MTF drops below threshold
    above_threshold = mtf > threshold
    
    # X direction cutoff
    center_y = ny // 2
    mtf_x = mtf[center_y, :]
    above_x = above_threshold[center_y, :]
    if torch.any(above_x):
        indices_x = torch.where(above_x)[0]
        fx_cutoff = torch.abs(fx[indices_x]).max().item()
    else:
        fx_cutoff = 0.0
    
    # Y direction cutoff
    center_x = nx // 2
    mtf_y = mtf[:, center_x]
    above_y = above_threshold[:, center_x]
    if torch.any(above_y):
        indices_y = torch.where(above_y)[0]
        fy_cutoff = torch.abs(fy[indices_y]).max().item()
    else:
        fy_cutoff = 0.0
    
    return fx_cutoff, fy_cutoff


def compute_fwhm(psf: torch.Tensor,
                 dx: float,
                 dy: float) -> Tuple[float, float]:
    """Compute Full Width at Half Maximum of PSF.
    
    Args:
        psf: PSF intensity distribution
        dx, dy: Grid spacing
        
    Returns:
        Tuple of (fwhm_x, fwhm_y) in micrometers
    """
    if torch.is_complex(psf):
        psf = torch.abs(psf)**2
    
    # Find peak
    peak_val = torch.max(psf)
    half_max = peak_val / 2
    
    # Find peak location
    peak_idx = torch.argmax(psf)
    ny, nx = psf.shape
    peak_y, peak_x = peak_idx // nx, peak_idx % nx
    
    # X direction FWHM
    profile_x = psf[peak_y, :]
    above_half_x = profile_x > half_max
    if torch.any(above_half_x):
        indices_x = torch.where(above_half_x)[0]
        fwhm_x = (indices_x[-1] - indices_x[0]).item() * dx
    else:
        fwhm_x = dx
    
    # Y direction FWHM
    profile_y = psf[:, peak_x]
    above_half_y = profile_y > half_max
    if torch.any(above_half_y):
        indices_y = torch.where(above_half_y)[0]
        fwhm_y = (indices_y[-1] - indices_y[0]).item() * dy
    else:
        fwhm_y = dy
    
    return fwhm_x, fwhm_y


def phase_rmse(phase_computed: torch.Tensor,
              phase_reference: torch.Tensor,
              unwrap: bool = False) -> float:
    """Compute RMS error in phase.
    
    Args:
        phase_computed: Computed phase (radians)
        phase_reference: Reference phase (radians)
        unwrap: If True, unwrap phase before comparison
        
    Returns:
        RMS phase error in radians
    """
    # Extract phase if complex
    if torch.is_complex(phase_computed):
        phase_computed = torch.angle(phase_computed)
    if torch.is_complex(phase_reference):
        phase_reference = torch.angle(phase_reference)
    
    if unwrap:
        # Simple phase unwrapping (2D unwrapping is complex)
        # For now, just ensure phase difference is in [-π, π]
        diff = phase_computed - phase_reference
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    else:
        diff = phase_computed - phase_reference
    
    rmse = torch.sqrt(torch.mean(diff**2))
    return rmse.item()


def beam_quality_m2(field: torch.Tensor,
                   z_positions: torch.Tensor,
                   dx: float,
                   dy: float,
                   wavelength: float) -> Tuple[float, float]:
    """Compute M² beam quality factor.
    
    M² = 1 for ideal Gaussian beam, >1 for real beams.
    
    Args:
        field: Complex field at multiple z positions, shape (nz, ny, nx)
        z_positions: Z positions of field slices
        dx, dy: Transverse grid spacing
        wavelength: Wavelength in micrometers
        
    Returns:
        Tuple of (M2_x, M2_y)
    """
    if field.dim() == 2:
        # Single z-plane, cannot compute M²
        return 1.0, 1.0
    
    nz, ny, nx = field.shape
    
    # Compute beam widths at each z
    widths_x = []
    widths_y = []
    
    for iz in range(nz):
        intensity = torch.abs(field[iz])**2
        
        # Compute second moments
        x = torch.arange(nx, dtype=torch.float32) * dx
        y = torch.arange(ny, dtype=torch.float32) * dy
        x = x - x.mean()
        y = y - y.mean()
        
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        total = torch.sum(intensity)
        if total > 0:
            x_mean = torch.sum(xx * intensity) / total
            y_mean = torch.sum(yy * intensity) / total
            
            w2_x = torch.sum((xx - x_mean)**2 * intensity) / total
            w2_y = torch.sum((yy - y_mean)**2 * intensity) / total
            
            widths_x.append(2 * torch.sqrt(w2_x).item())  # 2w definition
            widths_y.append(2 * torch.sqrt(w2_y).item())
    
    if len(widths_x) < 3:
        # Not enough points for fit
        return 1.0, 1.0
    
    # Fit to hyperbolic beam propagation
    # w²(z) = w₀² + (M² λ/π w₀)² (z - z₀)²
    # This is simplified - proper fitting would use least squares
    
    # Find minimum width (approximate waist)
    w0_x = min(widths_x)
    w0_y = min(widths_y)
    
    # Estimate M² from beam divergence
    # Simplified: use first and last points
    if len(z_positions) >= 2:
        dz = abs(z_positions[-1] - z_positions[0]).item()
        if dz > 0:
            dw_x = abs(widths_x[-1] - widths_x[0])
            dw_y = abs(widths_y[-1] - widths_y[0])
            
            # Theoretical divergence for Gaussian beam
            theta_0 = wavelength / (np.pi * w0_x)
            
            # Actual divergence
            theta_x = dw_x / dz if dz > 0 else 0
            theta_y = dw_y / dz if dz > 0 else 0
            
            # M² ratio
            M2_x = theta_x / theta_0 if theta_0 > 0 else 1.0
            M2_y = theta_y / theta_0 if theta_0 > 0 else 1.0
        else:
            M2_x = M2_y = 1.0
    else:
        M2_x = M2_y = 1.0
    
    # M² should be >= 1
    M2_x = max(M2_x, 1.0)
    M2_y = max(M2_y, 1.0)
    
    return M2_x, M2_y


def compare_propagation_methods(field: torch.Tensor,
                               methods: dict,
                               plan: dict) -> dict:
    """Compare results from different propagation methods.
    
    Args:
        field: Input field
        methods: Dictionary of method_name: propagation_function
        plan: Propagation plan
        
    Returns:
        Dictionary of comparison metrics
    """
    results = {}
    fields_out = {}
    
    # Run each method
    for name, method in methods.items():
        fields_out[name] = method(field, plan)
    
    # Compare all pairs
    method_names = list(methods.keys())
    for i, name1 in enumerate(method_names):
        for name2 in method_names[i+1:]:
            pair_key = f"{name1}_vs_{name2}"
            
            # Compute L2 error
            l2_err = l2_field_error(fields_out[name1], fields_out[name2])
            
            # Phase RMSE
            phase_err = phase_rmse(fields_out[name1], fields_out[name2])
            
            results[pair_key] = {
                'l2_error': l2_err,
                'phase_rmse': phase_err,
            }
    
    return results
