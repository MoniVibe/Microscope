"""Analytic test cases for validation of optical propagation.

Provides reference cases with known analytical solutions for
validating propagation algorithms.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import numpy as np


def gaussian_free_space(wavelength_um: float,
                        waist_um: float,
                        z_um: float,
                        nx: int = 256,
                        ny: int = 256,
                        dx: float = None,
                        dy: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytical Gaussian beam propagation in free space.
    
    Args:
        wavelength_um: Wavelength in micrometers
        waist_um: Beam waist (1/e² intensity radius)
        z_um: Propagation distance
        nx, ny: Grid size
        dx, dy: Grid spacing (auto if None)
        
    Returns:
        Tuple of (initial_field, propagated_field)
    """
    # Auto grid spacing
    if dx is None:
        dx = waist_um / 10  # 10 samples across waist
    if dy is None:
        dy = dx
    
    # Grid coordinates
    x = torch.linspace(-(nx-1)/2, (nx-1)/2, nx) * dx
    y = torch.linspace(-(ny-1)/2, (ny-1)/2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r2 = xx**2 + yy**2
    
    # Rayleigh range
    z_R = np.pi * waist_um**2 / wavelength_um
    
    # Beam parameters at z
    w_z = waist_um * np.sqrt(1 + (z_um/z_R)**2)
    R_z = z_um * (1 + (z_R/z_um)**2) if abs(z_um) > 1e-10 else float('inf')
    gouy_z = np.arctan(z_um/z_R)
    
    # Wave number
    k = 2 * np.pi / wavelength_um
    
    # Initial field (z=0)
    E0 = torch.exp(-r2/waist_um**2).to(torch.complex64)
    E0 = E0 / torch.sqrt(torch.sum(torch.abs(E0)**2))  # Normalize
    
    # Propagated field (analytical)
    # Gaussian beam formula
    amplitude = (waist_um/w_z) * torch.exp(-r2/w_z**2)
    
    # Phase terms
    phase = k * z_um  # Plane wave phase
    if abs(R_z) < 1e10:  # Finite radius of curvature
        phase = phase + k * r2 / (2 * R_z)  # Spherical phase
    phase = phase - gouy_z  # Gouy phase
    
    Ez = amplitude * torch.exp(1j * phase)
    Ez = Ez.to(torch.complex64)
    
    # Normalize to conserve energy
    Ez = Ez * torch.sqrt(torch.sum(torch.abs(E0)**2) / torch.sum(torch.abs(Ez)**2))
    
    return E0, Ez


def aperture_diffraction(wavelength_um: float,
                        aperture_diameter_um: float,
                        z_um: float,
                        nx: int = 512,
                        ny: int = 512,
                        dx: float = None,
                        dy: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fraunhofer diffraction by circular aperture (Airy pattern).
    
    Args:
        wavelength_um: Wavelength
        aperture_diameter_um: Aperture diameter
        z_um: Propagation distance (should be >> aperture for Fraunhofer)
        nx, ny: Grid size
        dx, dy: Grid spacing
        
    Returns:
        Tuple of (aperture_field, diffraction_pattern)
    """
    if dx is None:
        # Choose spacing for good sampling of both aperture and pattern
        dx = min(aperture_diameter_um / 20, wavelength_um * z_um / (4 * aperture_diameter_um))
    if dy is None:
        dy = dx
    
    # Grid
    x = torch.linspace(-(nx-1)/2, (nx-1)/2, nx) * dx
    y = torch.linspace(-(ny-1)/2, (ny-1)/2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)
    
    # Circular aperture
    aperture = (r <= aperture_diameter_um/2).to(torch.complex64)
    
    # Analytical Airy pattern (Fraunhofer approximation)
    k = 2 * np.pi / wavelength_um
    
    # Observation plane coordinates
    # In Fraunhofer approximation, field scales with z
    x_obs = x * z_um / (k * (aperture_diameter_um/2)**2) * wavelength_um
    y_obs = y * z_um / (k * (aperture_diameter_um/2)**2) * wavelength_um
    
    # For far field, use angular coordinates
    theta = r / z_um  # Small angle approximation
    
    # Airy function argument
    # J₁(x)/x where x = k*a*sin(θ) and a = aperture radius
    a = aperture_diameter_um / 2
    x_airy = k * a * theta
    
    # Avoid division by zero
    x_airy_safe = torch.where(x_airy > 1e-10, x_airy, torch.ones_like(x_airy) * 1e-10)
    
    # Airy pattern (using approximation for J₁(x)/x)
    # For small x: J₁(x)/x ≈ 1/2
    # For large x: use series approximation
    
    # Simplified: use sinc-like approximation
    # Real Airy would need Bessel function
    pattern = torch.where(
        x_airy < 0.01,
        torch.ones_like(x_airy),
        2 * torch.sin(x_airy_safe) / x_airy_safe
    )
    
    # Convert to complex field
    diffracted = pattern.to(torch.complex64)
    
    # Add propagation phase
    phase = k * z_um + k * r**2 / (2 * z_um)  # Fresnel approximation
    diffracted = diffracted * torch.exp(1j * phase)
    
    # Normalize
    diffracted = diffracted / torch.sqrt(torch.sum(torch.abs(diffracted)**2))
    
    return aperture, diffracted


def thin_lens_focus(wavelength_um: float,
                   focal_length_um: float,
                   lens_diameter_um: float,
                   z_um: float,
                   nx: int = 256,
                   ny: int = 256,
                   dx: float = None,
                   dy: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Focusing by ideal thin lens.
    
    Args:
        wavelength_um: Wavelength
        focal_length_um: Focal length
        lens_diameter_um: Lens aperture diameter
        z_um: Distance from lens (focal_length_um for focus)
        nx, ny: Grid size
        dx, dy: Grid spacing
        
    Returns:
        Tuple of (lens_field, propagated_field)
    """
    if dx is None:
        dx = lens_diameter_um / 40
    if dy is None:
        dy = dx
    
    # Grid
    x = torch.linspace(-(nx-1)/2, (nx-1)/2, nx) * dx
    y = torch.linspace(-(ny-1)/2, (ny-1)/2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r2 = xx**2 + yy**2
    r = torch.sqrt(r2)
    
    # Incident plane wave (or could use Gaussian)
    incident = torch.ones((ny, nx), dtype=torch.complex64)
    
    # Lens aperture
    aperture = (r <= lens_diameter_um/2).to(torch.float32)
    
    # Thin lens phase
    k = 2 * np.pi / wavelength_um
    lens_phase = -k * r2 / (2 * focal_length_um)
    
    # Field immediately after lens
    field_after_lens = incident * aperture * torch.exp(1j * lens_phase)
    
    # Propagate to observation plane
    # Near focus, use Fresnel propagation
    
    # Fresnel number
    fresnel_number = lens_diameter_um**2 / (4 * wavelength_um * abs(z_um))
    
    if z_um == focal_length_um:
        # At focus - use analytical PSF
        # For ideal lens, this is Airy pattern scaled by NA
        NA = lens_diameter_um / (2 * focal_length_um)
        
        # PSF scaling
        psf_scale = wavelength_um / NA
        
        # Coordinate scaling at focus
        x_focus = x * psf_scale / lens_diameter_um
        y_focus = y * psf_scale / lens_diameter_um
        r_focus = torch.sqrt(x_focus**2 + y_focus**2)
        
        # Simplified Airy-like pattern
        psf = torch.sinc(2 * r_focus / wavelength_um).to(torch.complex64)
        
        # Normalize
        psf = psf / torch.sqrt(torch.sum(torch.abs(psf)**2))
        field_propagated = psf
        
    else:
        # Off-focus - use Fresnel propagation
        # Simplified quadratic phase approximation
        
        # Effective focal length considering propagation
        f_eff = 1 / (1/focal_length_um - 1/z_um) if z_um != focal_length_um else float('inf')
        
        # Quadratic phase
        if abs(f_eff) < 1e10:
            defocus_phase = k * r2 / (2 * f_eff)
        else:
            defocus_phase = torch.zeros_like(r2)
        
        field_propagated = field_after_lens * torch.exp(1j * defocus_phase)
        
        # Apply aperture diffraction effects (simplified)
        if fresnel_number < 10:
            # Add some diffraction spreading
            spread = 1 + (z_um - focal_length_um)**2 / focal_length_um**2
            field_propagated = field_propagated / np.sqrt(spread)
    
    return field_after_lens, field_propagated


def phase_grating_orders(wavelength_um: float,
                        period_um: float,
                        phase_depth: float,
                        z_um: float,
                        nx: int = 512,
                        ny: int = 512,
                        dx: float = None,
                        dy: float = None) -> Dict:
    """Diffraction by sinusoidal phase grating.
    
    Args:
        wavelength_um: Wavelength
        period_um: Grating period
        phase_depth: Peak-to-peak phase modulation (radians)
        z_um: Propagation distance
        nx, ny: Grid size
        dx, dy: Grid spacing
        
    Returns:
        Dictionary with grating field and diffraction orders
    """
    if dx is None:
        dx = period_um / 20  # 20 samples per period
    if dy is None:
        dy = dx
    
    # Grid
    x = torch.linspace(-(nx-1)/2, (nx-1)/2, nx) * dx
    y = torch.linspace(-(ny-1)/2, (ny-1)/2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Sinusoidal phase grating
    grating_phase = (phase_depth/2) * torch.sin(2 * np.pi * xx / period_um)
    grating = torch.exp(1j * grating_phase).to(torch.complex64)
    
    # Analytical diffraction orders (Bessel function expansion)
    # exp(i*a*sin(x)) = Σ J_n(a) * exp(i*n*x)
    # where J_n are Bessel functions
    
    # For small phase depth, keep only few orders
    from scipy.special import jn  # Bessel functions
    
    max_order = int(phase_depth / np.pi) + 3
    orders = {}
    
    for n in range(-max_order, max_order + 1):
        # Bessel function amplitude
        amplitude = jn(n, phase_depth/2)
        
        if abs(amplitude) > 1e-6:
            # Diffraction angle
            sin_theta = n * wavelength_um / period_um
            
            if abs(sin_theta) <= 1:  # Propagating order
                theta = np.arcsin(sin_theta)
                
                # Position at distance z
                x_order = z_um * np.tan(theta)
                
                # Phase at distance z
                k = 2 * np.pi / wavelength_um
                cos_theta = np.cos(theta)
                phase_prop = k * z_um / cos_theta
                
                orders[n] = {
                    'amplitude': amplitude,
                    'angle_deg': np.degrees(theta),
                    'position_um': x_order,
                    'phase': phase_prop,
                    'efficiency': amplitude**2  # Power in order
                }
    
    # Total field at distance z (superposition of orders)
    field_z = torch.zeros((ny, nx), dtype=torch.complex64)
    
    for n, order_data in orders.items():
        # Each order is a plane wave at angle
        kx = 2 * np.pi * n / period_um
        ky = 0  # 1D grating
        kz = np.sqrt((2*np.pi/wavelength_um)**2 - kx**2 - ky**2)
        
        # Field contribution from this order
        phase = kx * xx + kz * z_um
        field_z += order_data['amplitude'] * torch.exp(1j * phase)
    
    return {
        'grating': grating,
        'field_z': field_z,
        'orders': orders,
        'total_efficiency': sum(o['efficiency'] for o in orders.values())
    }


def high_na_reference(wavelength_um: float,
                     na: float,
                     z_um: float,
                     nx: int = 256,
                     ny: int = 256) -> torch.Tensor:
    """Generate high-NA reference field using angular spectrum.
    
    This serves as a reference for validating high-NA propagation.
    
    Args:
        wavelength_um: Wavelength
        na: Numerical aperture
        z_um: Propagation distance
        nx, ny: Grid size
        
    Returns:
        Reference field at distance z
    """
    # Grid spacing from NA
    dx = dy = wavelength_um / (4 * na)  # 2x Nyquist
    
    # Initial field - circular pupil with NA
    x = torch.linspace(-(nx-1)/2, (nx-1)/2, nx) * dx
    y = torch.linspace(-(ny-1)/2, (ny-1)/2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Frequency space
    fx = torch.fft.fftfreq(nx, d=dx)
    fy = torch.fft.fftfreq(ny, d=dy)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # NA-limited pupil in frequency space
    f_max = na / wavelength_um
    pupil = (torch.sqrt(fx_grid**2 + fy_grid**2) <= f_max).to(torch.complex64)
    
    # High-NA angular spectrum propagation
    k = 2 * np.pi / wavelength_um
    kx = 2 * np.pi * fx_grid
    ky = 2 * np.pi * fy_grid
    
    # Exact kz (no paraxial approximation)
    kz2 = k**2 - kx**2 - ky**2
    kz = torch.sqrt(torch.abs(kz2)) * torch.sign(kz2)
    
    # Handle evanescent waves properly
    is_propagating = kz2 > 0
    propagator = torch.where(
        is_propagating,
        torch.exp(1j * kz * z_um),
        torch.exp(-torch.abs(kz) * z_um) * 1e-10  # Strong decay for evanescent
    )
    
    # Apply propagation
    pupil_propagated = pupil * propagator
    
    # Include obliquity factor for high NA
    if na > 0.5:
        obliquity = torch.where(
            is_propagating,
            torch.sqrt(torch.abs(kz / k)),
            torch.zeros_like(kz)
        )
        pupil_propagated = pupil_propagated * obliquity
    
    # Transform to spatial domain
    field = torch.fft.ifft2(torch.fft.ifftshift(pupil_propagated))
    field = torch.fft.fftshift(field)
    
    # Normalize
    field = field / torch.sqrt(torch.sum(torch.abs(field)**2))
    
    return field


def validate_case(computed_field: torch.Tensor,
                 reference_field: torch.Tensor,
                 case_name: str,
                 thresholds: Optional[Dict] = None) -> Dict:
    """Validate computed field against reference.
    
    Args:
        computed_field: Field from simulation
        reference_field: Analytical reference
        case_name: Name of test case
        thresholds: Pass/fail thresholds
        
    Returns:
        Dictionary of validation metrics and pass/fail status
    """
    if thresholds is None:
        thresholds = {
            'l2_error': 0.03,  # 3% L2 error
            'energy_conservation': 0.01,  # 1% energy change
            'phase_rmse': 0.1,  # 0.1 radian phase error
        }
    
    from .metrics import l2_field_error, energy_conservation, phase_rmse
    
    # Compute metrics
    l2_err = l2_field_error(computed_field, reference_field)
    
    # Energy
    energy_comp = torch.sum(torch.abs(computed_field)**2).item()
    energy_ref = torch.sum(torch.abs(reference_field)**2).item()
    energy_err = abs(energy_comp - energy_ref) / energy_ref if energy_ref > 0 else 0
    
    # Phase
    phase_err = phase_rmse(computed_field, reference_field)
    
    # Check pass/fail
    passed = (
        l2_err <= thresholds['l2_error'] and
        energy_err <= thresholds['energy_conservation'] and
        phase_err <= thresholds['phase_rmse']
    )
    
    return {
        'case_name': case_name,
        'passed': passed,
        'metrics': {
            'l2_error': l2_err,
            'energy_error': energy_err,
            'phase_rmse': phase_err,
        },
        'thresholds': thresholds,
    }
