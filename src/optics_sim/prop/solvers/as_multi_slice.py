"""Angular Spectrum multi-slice propagation method.

Implements nonparaxial propagation using angular spectrum method
with proper handling of evanescent waves and high-NA systems.
"""

from __future__ import annotations

from typing import Dict, Optional, List

import torch
import numpy as np


def run(field: torch.Tensor,
        plan: Optional[Dict] = None,
        sampler: Optional[Dict] = None,
        refractive_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run angular spectrum multi-slice propagation.
    
    Nonparaxial propagation using exact angular spectrum transfer function.
    Properly handles evanescent waves and high-NA fields.
    
    Args:
        field: Complex field tensor of shape (ny, nx) or (S, ny, nx)
        plan: Propagation plan with grid and step parameters
        sampler: Sampling parameters (optional)
        refractive_index: Refractive index distribution (optional, for multi-slice)
        
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
    
    device = field.device
    
    # Handle spectral dimension
    if field.dim() == 2:
        field = field.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    S, ny, nx = field.shape
    
    # Default refractive index (vacuum/air)
    if refractive_index is None:
        n = torch.ones((ny, nx), dtype=torch.float32, device=device)
    else:
        n = refractive_index.to(device)
    
    # Process each spectral component
    output_fields = []
    
    for s in range(S):
        lambda_um = wavelengths[s] if s < len(wavelengths) else wavelengths[0]
        
        E = field[s].clone()
        
        # Multi-slice propagation through all z-steps
        for i, dz in enumerate(dz_list):
            # For multi-slice, we can have different n at each slice
            if isinstance(refractive_index, list):
                n_slice = refractive_index[i] if i < len(refractive_index) else n
            else:
                n_slice = n
            
            E = _angular_spectrum_step(E, n_slice, lambda_um, dx, dy, dz, na_max)
        
        output_fields.append(E)
    
    output = torch.stack(output_fields, dim=0)
    
    if squeeze_output:
        output = output.squeeze(0)
    
    return output


def _angular_spectrum_step(E: torch.Tensor, n: torch.Tensor, lambda_um: float,
                          dx: float, dy: float, dz: float, na_max: float) -> torch.Tensor:
    """Propagate field using angular spectrum method.
    
    Exact nonparaxial propagation with proper evanescent wave handling.
    
    Args:
        E: Current field
        n: Refractive index (can be inhomogeneous)
        lambda_um: Wavelength in micrometers
        dx, dy: Transverse grid spacing
        dz: Propagation distance
        na_max: Maximum NA for band limiting
        
    Returns:
        Propagated field
    """
    ny, nx = E.shape
    device = E.device
    
    # Wave number
    k0 = 2 * np.pi / lambda_um
    n_avg = n.mean()
    k = k0 * n_avg
    
    # FFT to angular spectrum
    E_spectrum = torch.fft.fft2(E)
    
    # Create frequency grids
    fx = torch.fft.fftfreq(nx, d=dx, device=device).to(torch.float32)
    fy = torch.fft.fftfreq(ny, d=dy, device=device).to(torch.float32)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # Angular spectrum transfer function
    # H(fx, fy, λ) = exp(i * k * z * sqrt(1 - (λ*fx)² - (λ*fy)²))
    
    # Normalized spatial frequencies
    fx_norm = lambda_um * fx_grid
    fy_norm = lambda_um * fy_grid
    
    # Square root argument
    arg = 1 - fx_norm**2 - fy_norm**2
    
    # Separate propagating and evanescent waves
    is_propagating = arg > 0
    
    # Compute kz for both regimes
    kz_propagating = k * torch.sqrt(torch.clamp(arg, min=0))
    kz_evanescent = k * torch.sqrt(torch.clamp(-arg, min=0))
    
    # Transfer function with evanescent wave clamping
    # Propagating waves: H = exp(i * kz * dz)
    # Evanescent waves: H = exp(-|kz| * dz) with clamping
    
    # Maximum decay factor to prevent numerical issues
    max_decay = 1e-6
    evanescent_decay = torch.exp(-kz_evanescent * dz)
    evanescent_decay = torch.clamp(evanescent_decay, min=max_decay)
    
    # Combine transfer functions
    H = torch.where(
        is_propagating,
        torch.exp(1j * kz_propagating * dz),
        evanescent_decay.to(torch.complex64)
    )
    
    # Band limiting based on NA
    # Maximum transverse k-vector from NA
    k_max = k0 * na_max
    k_transverse = k0 * torch.sqrt(fx_norm**2 + fy_norm**2)
    
    # Smooth band limit with cosine taper
    band_limit = _create_band_limit(k_transverse, k_max, k0)
    H = H * band_limit
    
    # Apply transfer function
    E_spectrum = E_spectrum * H
    
    # Inverse FFT back to spatial domain
    E_propagated = torch.fft.ifft2(E_spectrum)
    
    # Handle inhomogeneous refractive index (multi-slice)
    if not torch.allclose(n, n_avg * torch.ones_like(n)):
        # Apply phase modulation for index variations
        phase_mod = k0 * (n - n_avg) * dz
        E_propagated = E_propagated * torch.exp(1j * phase_mod.to(torch.complex64))
    
    return E_propagated


def _create_band_limit(k_transverse: torch.Tensor, k_max: float, k0: float) -> torch.Tensor:
    """Create smooth band-limiting filter.
    
    Args:
        k_transverse: Transverse wave vector magnitude
        k_max: Maximum k from NA
        k0: Vacuum wave number
        
    Returns:
        Band limit mask with smooth edges
    """
    # Use cosine taper for smooth transition
    transition_width = 0.05 * k0  # 5% of k0 for transition
    
    mask = torch.where(
        k_transverse < k_max - transition_width,
        torch.ones_like(k_transverse),
        torch.where(
            k_transverse > k_max + transition_width,
            torch.zeros_like(k_transverse),
            0.5 * (1 + torch.cos(np.pi * (k_transverse - k_max + transition_width) / (2 * transition_width)))
        )
    )
    
    return mask.to(torch.complex64)


def angular_spectrum_exact(field: torch.Tensor, lambda_um: float,
                          dx: float, dy: float, z: float,
                          n: float = 1.0) -> torch.Tensor:
    """Exact angular spectrum propagation for a single distance.
    
    Reference implementation without approximations.
    
    Args:
        field: Input field
        lambda_um: Wavelength
        dx, dy: Grid spacing
        z: Propagation distance
        n: Refractive index (scalar)
        
    Returns:
        Propagated field
    """
    ny, nx = field.shape
    device = field.device
    
    # Wave number
    k = 2 * np.pi * n / lambda_um
    
    # FFT
    field_spectrum = torch.fft.fft2(field)
    
    # Frequency grids
    fx = torch.fft.fftfreq(nx, d=dx, device=device)
    fy = torch.fft.fftfreq(ny, d=dy, device=device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # Exact transfer function
    kx = 2 * np.pi * fx_grid
    ky = 2 * np.pi * fy_grid
    kz2 = k**2 - kx**2 - ky**2
    
    # Handle both propagating and evanescent
    kz = torch.where(kz2 >= 0,
                    torch.sqrt(kz2),
                    1j * torch.sqrt(-kz2))
    
    H = torch.exp(1j * kz * z)
    
    # Apply and inverse transform
    field_spectrum = field_spectrum * H
    field_out = torch.fft.ifft2(field_spectrum)
    
    return field_out


def multi_slice_propagate(field: torch.Tensor, 
                         slices: List[Dict],
                         lambda_um: float,
                         dx: float, dy: float) -> torch.Tensor:
    """Propagate through multiple slices with varying properties.
    
    Each slice can have different refractive index and thickness.
    
    Args:
        field: Input field
        slices: List of dicts with 'n' (refractive index) and 'dz' (thickness)
        lambda_um: Wavelength
        dx, dy: Grid spacing
        
    Returns:
        Field after all slices
    """
    E = field.clone()
    
    for slice_params in slices:
        n_slice = slice_params.get('n', 1.0)
        dz_slice = slice_params.get('dz', 1.0)
        
        # Convert n to tensor if needed
        if isinstance(n_slice, (int, float)):
            n_slice = torch.full_like(field.real, n_slice)
        
        # Propagate through slice
        E = _angular_spectrum_step(E, n_slice, lambda_um, dx, dy, dz_slice, na_max=1.0)
    
    return E


def compute_psf(lambda_um: float, na: float, 
               nx: int = 256, ny: int = 256,
               dx: float = None, dy: float = None,
               z: float = 0.0) -> torch.Tensor:
    """Compute point spread function using angular spectrum.
    
    Args:
        lambda_um: Wavelength
        na: Numerical aperture
        nx, ny: Grid size
        dx, dy: Grid spacing (auto if None)
        z: Defocus distance
        
    Returns:
        PSF intensity distribution
    """
    # Auto grid spacing from NA
    if dx is None:
        dx = lambda_um / (4 * na)  # Oversample by 2x
    if dy is None:
        dy = dx
    
    # Create pupil function (circular aperture)
    x = torch.linspace(-nx//2, nx//2-1, nx) * dx
    y = torch.linspace(-ny//2, ny//2-1, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Pupil radius in spatial frequency
    pupil_radius = na / lambda_um
    
    # FFT frequencies
    fx = torch.fft.fftfreq(nx, d=dx)
    fy = torch.fft.fftfreq(ny, d=dy)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # Pupil function in frequency domain
    f_radial = torch.sqrt(fx_grid**2 + fy_grid**2)
    pupil = (f_radial <= pupil_radius).to(torch.complex64)
    
    # Phase for defocus
    if abs(z) > 0:
        k = 2 * np.pi / lambda_um
        kz = k * torch.sqrt(1 - (lambda_um * f_radial)**2)
        kz = torch.where(f_radial <= pupil_radius, kz, torch.zeros_like(kz))
        pupil = pupil * torch.exp(1j * kz * z)
    
    # Inverse FFT to get PSF
    psf_complex = torch.fft.ifft2(torch.fft.ifftshift(pupil))
    psf_intensity = torch.abs(psf_complex)**2
    
    # Normalize
    psf_intensity = psf_intensity / psf_intensity.sum()
    
    return psf_intensity


def validate_angular_spectrum(field_in: torch.Tensor,
                            field_out: torch.Tensor,
                            lambda_um: float,
                            dx: float, dy: float,
                            z: float) -> Dict:
    """Validate angular spectrum propagation.
    
    Args:
        field_in: Input field
        field_out: Output field
        lambda_um: Wavelength
        dx, dy: Grid spacing
        z: Propagation distance
        
    Returns:
        Dictionary with validation metrics
    """
    # Energy conservation
    energy_in = torch.sum(torch.abs(field_in)**2) * dx * dy
    energy_out = torch.sum(torch.abs(field_out)**2) * dx * dy
    energy_ratio = energy_out / energy_in if energy_in > 0 else 0
    
    # Check reciprocity (backward propagation should recover input)
    field_back = angular_spectrum_exact(field_out, lambda_um, dx, dy, -z)
    reciprocity_error = torch.norm(field_back - field_in) / torch.norm(field_in)
    
    # Fresnel number
    L = min(field_in.shape[-2] * dy, field_in.shape[-1] * dx)  # Field size
    fresnel_number = L**2 / (lambda_um * abs(z)) if z != 0 else float('inf')
    
    return {
        'energy_ratio': energy_ratio.item(),
        'reciprocity_error': reciprocity_error.item(),
        'fresnel_number': fresnel_number,
        'is_near_field': fresnel_number > 1,
    }
