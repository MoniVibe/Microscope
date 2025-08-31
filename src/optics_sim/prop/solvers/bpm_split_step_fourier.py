"""Beam Propagation Method with split-step Fourier approach.

Scalar wide-angle BPM using frequency-domain propagation for
efficient computation of beam propagation in inhomogeneous media.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import numpy as np


def run(field: torch.Tensor,
        plan: Optional[Dict] = None,
        sampler: Optional[Dict] = None,
        refractive_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run scalar wide-angle BPM using split-step Fourier method.
    
    Efficient propagation using alternating spatial and frequency domain
    operations. Suitable for moderate to high NA systems.
    
    Args:
        field: Complex field tensor of shape (ny, nx) or (S, ny, nx)
        plan: Propagation plan with grid and step parameters
        sampler: Sampling parameters (optional)
        refractive_index: Refractive index distribution (optional)
        
    Returns:
        Propagated field with same shape as input
    """
    if plan is None:
        return field
    
    # Extract parameters
    dx = plan.dx_um if hasattr(plan, 'dx_um') else plan.get('dx_um', 0.5)
    dy = plan.dy_um if hasattr(plan, 'dy_um') else plan.get('dy_um', 0.5)
    dz_list = plan.dz_list_um if hasattr(plan, 'dz_list_um') else plan.get('dz_list_um', [1.0])
    wavelengths = plan.wavelengths_um if hasattr(plan, 'wavelengths_um') else plan.get('wavelengths_um', np.array([0.55]))
    na_max = plan.na_max if hasattr(plan, 'na_max') else plan.get('na_max', 0.25)
    
    # Configuration flags
    use_mixed = plan.use_mixed_precision if hasattr(plan, 'use_mixed_precision') else plan.get('use_mixed_precision', False)
    pml_thickness = plan.pml_thickness_px if hasattr(plan, 'pml_thickness_px') else plan.get('pml_thickness_px', 16)
    
    device = field.device
    
    # Handle spectral dimension
    if field.dim() == 2:
        field = field.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    S, ny, nx = field.shape
    
    # Default refractive index
    if refractive_index is None:
        n = torch.ones((ny, nx), dtype=torch.float32, device=device)
    else:
        n = refractive_index.to(device)
    
    # Create PML and band-limit mask
    pml = _create_pml_smooth(ny, nx, pml_thickness, device)
    band_mask = _create_band_limit_mask(ny, nx, dx, dy, na_max, wavelengths[0], n.mean().item(), device)
    
    # Process each spectral component
    output_fields = []
    
    for s in range(S):
        lambda_um = wavelengths[s] if s < len(wavelengths) else wavelengths[0]
        k0 = 2 * np.pi / lambda_um
        
        E = field[s].clone()
        E = E * pml  # Apply input PML
        
        # Propagate through z-steps
        for dz in dz_list:
            E = _split_step_fourier(E, n, k0, dx, dy, dz, na_max, pml, band_mask, use_mixed)
        
        output_fields.append(E)
    
    output = torch.stack(output_fields, dim=0)
    
    if squeeze_output:
        output = output.squeeze(0)
    
    return output


def _create_pml_smooth(ny: int, nx: int, thickness: int, device: str) -> torch.Tensor:
    """Create smooth PML absorber using polynomial grading.
    
    Args:
        ny, nx: Grid dimensions
        thickness: PML thickness in pixels
        device: Computation device
        
    Returns:
        PML transmission mask
    """
    if thickness <= 0:
        return torch.ones((ny, nx), device=device)
    
    pml = torch.ones((ny, nx), dtype=torch.float32, device=device)
    
    # Create coordinate grids
    y = torch.arange(ny, device=device, dtype=torch.float32)
    x = torch.arange(nx, device=device, dtype=torch.float32)
    
    # Distance from edges (vectorized)
    dist_top = y
    dist_bottom = ny - 1 - y
    dist_left = x
    dist_right = nx - 1 - x
    
    # Minimum distance to edge for each row/column
    for i in range(ny):
        y_dist = min(dist_top[i], dist_bottom[i])
        for j in range(nx):
            x_dist = min(dist_left[j], dist_right[j])
            edge_dist = min(y_dist.item(), x_dist.item())
            
            if edge_dist < thickness:
                # Polynomial grading (quartic for smoother transition)
                t = edge_dist / thickness
                pml[i, j] = t**4
    
    return pml


def _create_band_limit_mask(ny: int, nx: int, dx: float, dy: float,
                           na_max: float, lambda_um: float, n_avg: float,
                           device: str) -> torch.Tensor:
    """Create frequency domain band-limiting mask.
    
    Args:
        ny, nx: Grid dimensions
        dx, dy: Grid spacing
        na_max: Maximum NA
        lambda_um: Wavelength
        n_avg: Average refractive index
        device: Computation device
        
    Returns:
        Band limit mask in frequency domain
    """
    # Frequency grids
    fx = torch.fft.fftfreq(nx, d=dx, device=device)
    fy = torch.fft.fftfreq(ny, d=dy, device=device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # Maximum spatial frequency from NA
    f_max = na_max / lambda_um
    
    # Circular band limit with smooth edge
    f_radial = torch.sqrt(fx_grid**2 + fy_grid**2)
    
    # Hard cutoff with smooth transition
    transition_width = 0.1 * f_max
    mask = torch.where(
        f_radial < f_max - transition_width,
        torch.ones_like(f_radial),
        torch.where(
            f_radial > f_max + transition_width,
            torch.zeros_like(f_radial),
            0.5 * (1 + torch.cos(np.pi * (f_radial - f_max + transition_width) / (2 * transition_width)))
        )
    )
    
    return mask


def _split_step_fourier(E: torch.Tensor, n: torch.Tensor, k0: float,
                        dx: float, dy: float, dz: float, na_max: float,
                        pml: torch.Tensor, band_mask: torch.Tensor,
                        use_mixed: bool = False) -> torch.Tensor:
    """Perform one split-step Fourier propagation.
    
    Args:
        E: Current field
        n: Refractive index
        k0: Vacuum wave number
        dx, dy, dz: Grid spacings
        na_max: Maximum NA
        pml: PML mask
        band_mask: Frequency band limit mask
        use_mixed: Use mixed precision
        
    Returns:
        Propagated field
    """
    ny, nx = E.shape
    device = E.device
    n_avg = n.mean()
    k_avg = k0 * n_avg
    
    # Mixed precision setup
    if use_mixed and device.type == 'cuda':
        compute_dtype = torch.complex32  # FP16 complex
        accumulate_dtype = torch.complex64  # FP32 complex
    else:
        compute_dtype = torch.complex64
        accumulate_dtype = torch.complex64
    
    # Step 1: Apply half of refractive index phase
    phase_spatial = 0.5 * k0 * (n - n_avg) * dz
    E = E * torch.exp(1j * phase_spatial.to(E.dtype))
    
    # Step 2: Fourier transform
    if use_mixed:
        E_fft = torch.fft.fft2(E.to(compute_dtype))
    else:
        E_fft = torch.fft.fft2(E)
    
    # Step 3: Apply propagation in frequency domain
    fx = torch.fft.fftfreq(nx, d=dx, device=device)
    fy = torch.fft.fftfreq(ny, d=dy, device=device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # Wave vector components
    kx = 2 * np.pi * fx_grid
    ky = 2 * np.pi * fy_grid
    
    # Longitudinal wave vector with wide-angle correction
    kz2 = k_avg**2 - kx**2 - ky**2
    
    # Handle evanescent waves (kz2 < 0)
    is_propagating = kz2 > 0
    kz = torch.sqrt(torch.abs(kz2))
    
    # Wide-angle propagator using Taylor expansion
    # For better accuracy at high angles
    if na_max > 0.5:
        # Use Padé approximant for sqrt(1 - (kx²+ky²)/k²)
        kt2_norm = (kx**2 + ky**2) / k_avg**2
        kt2_norm = torch.clamp(kt2_norm, max=0.99)
        
        # Padé (2,2) for higher accuracy
        # sqrt(1-x) ≈ (1 - 5x/8 + x²/8) / (1 - x/8 - x²/8)
        numerator = 1 - 5*kt2_norm/8 + kt2_norm**2/8
        denominator = 1 - kt2_norm/8 - kt2_norm**2/8
        denominator = torch.where(torch.abs(denominator) > 1e-10,
                                 denominator,
                                 torch.ones_like(denominator))
        
        kz_norm = numerator / denominator
        kz_effective = k_avg * kz_norm
    else:
        # Simple propagator for low NA
        kz_effective = torch.where(is_propagating, kz, torch.zeros_like(kz))
    
    # Propagation phase
    prop_phase = kz_effective * dz
    
    # Apply band limiting
    prop_phase = prop_phase * band_mask
    
    # Evanescent wave decay
    if na_max > 0.8:  # Only for high-NA
        decay = torch.where(~is_propagating,
                          torch.exp(-kz * dz),
                          torch.ones_like(kz))
    else:
        decay = torch.ones_like(kz)
    
    # Apply propagator
    propagator = decay * torch.exp(1j * prop_phase.to(E_fft.dtype))
    E_fft = E_fft * propagator
    
    # Step 4: Inverse Fourier transform
    E = torch.fft.ifft2(E_fft)
    
    if use_mixed:
        E = E.to(accumulate_dtype)
    
    # Step 5: Apply second half of refractive index phase
    E = E * torch.exp(1j * phase_spatial.to(E.dtype))
    
    # Step 6: Apply PML
    E = E * pml
    
    return E


def compute_beam_parameters(field: torch.Tensor, dx: float, dy: float) -> Dict:
    """Compute beam quality parameters.
    
    Args:
        field: Complex field
        dx, dy: Grid spacing
        
    Returns:
        Dictionary with beam parameters (centroid, width, M², etc.)
    """
    intensity = torch.abs(field)**2
    
    # Grid coordinates
    ny, nx = field.shape
    x = torch.arange(nx, device=field.device, dtype=torch.float32) * dx
    y = torch.arange(ny, device=field.device, dtype=torch.float32) * dy
    x = x - x.mean()  # Center
    y = y - y.mean()
    
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Total power
    power = torch.sum(intensity) * dx * dy
    
    if power > 0:
        # Centroid
        x_cent = torch.sum(xx * intensity) / torch.sum(intensity)
        y_cent = torch.sum(yy * intensity) / torch.sum(intensity)
        
        # Second moments (beam width)
        x2 = torch.sum((xx - x_cent)**2 * intensity) / torch.sum(intensity)
        y2 = torch.sum((yy - y_cent)**2 * intensity) / torch.sum(intensity)
        
        # RMS widths
        wx = torch.sqrt(x2)
        wy = torch.sqrt(y2)
    else:
        x_cent = y_cent = 0.0
        wx = wy = 0.0
    
    return {
        'power': power.item(),
        'centroid_x': x_cent.item() if power > 0 else 0.0,
        'centroid_y': y_cent.item() if power > 0 else 0.0,
        'width_x': wx.item() if power > 0 else 0.0,
        'width_y': wy.item() if power > 0 else 0.0,
    }


def adaptive_propagate(field: torch.Tensor, plan: Dict,
                      error_threshold: float = 0.01) -> torch.Tensor:
    """Propagate with adaptive step sizing based on error estimate.
    
    Args:
        field: Input field
        plan: Propagation plan
        error_threshold: Maximum allowed phase error per step
        
    Returns:
        Propagated field
    """
    # This is a more advanced version that subdivides steps if needed
    # For now, delegate to standard propagation
    return run(field, plan)
