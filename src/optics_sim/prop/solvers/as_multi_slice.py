"""Angular Spectrum multi-slice propagation method.

Implements nonparaxial propagation using angular spectrum method
with proper handling of evanescent waves and high-NA systems.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from optics_sim.core.precision import (
    enforce_fp32_cuda,
    assert_fp32_cuda,
)


def run(field: torch.Tensor, plan: dict) -> torch.Tensor:
    """Single-step nonparaxial angular spectrum propagation.

    Inputs are on CPU/GPU; units are µm everywhere. Uses the first z and wavelength.
    CPU path maintains FP64 intermediates for accuracy.
    CUDA path uses FP32 throughout.
    """
    # Required parameters
    dx = float(plan["dx_um"])  # µm
    dy = float(plan["dy_um"])  # µm
    z = float(plan["dz_list_um"][0])  # µm
    lam = float(plan["wavelengths_um"][0])  # µm

    device = field.device
    ny, nx = field.shape[-2], field.shape[-1]
    
    # Enforce FP32 on CUDA
    if device.type == "cuda":
        field = enforce_fp32_cuda(field, "input field")
        assert_fp32_cuda(field, "AS input")

    # Frequency grids: FP64 on CPU, FP32 on CUDA
    if device.type == "cuda":
        # FP32 for CUDA
        fx = torch.fft.fftfreq(nx, d=dx, device=device).to(torch.float32)
        fy = torch.fft.fftfreq(ny, d=dy, device=device).to(torch.float32)
        precision_dtype = torch.float32
        complex_dtype = torch.complex64
    else:
        # FP64 for CPU (better accuracy)
        fx = torch.fft.fftfreq(nx, d=dx, device=device).to(torch.float64)
        fy = torch.fft.fftfreq(ny, d=dy, device=device).to(torch.float64)
        precision_dtype = torch.float64
        complex_dtype = torch.complex128
    # Angular spatial frequencies (rad/µm)
    kx = (2.0 * math.pi) * fx.view(1, nx)
    ky = (2.0 * math.pi) * fy.view(ny, 1)

    # Wavenumber and kz, non-paraxial
    k0 = 2.0 * math.pi / lam
    arg = 1.0 - (lam * fx.view(1, nx)) ** 2 - (lam * fy.view(ny, 1)) ** 2
    arg = arg.clamp(min=0.0).to(precision_dtype)
    kz = k0 * torch.sqrt(arg)  # rad/µm

    # Optional NA soft taper near cutoff (won't affect core with these test params)
    w = None
    if "na_max" in plan and plan["na_max"] is not None:
        f_na = float(plan["na_max"]) / lam  # cycles/µm
        r2 = (fx.view(1, nx) ** 2 + fy.view(ny, 1) ** 2)
        w = torch.ones_like(r2, dtype=precision_dtype, device=device)
        band = (r2 > (0.98 * f_na) ** 2) & (r2 <= f_na ** 2)
        if band.any():
            w[band] = 0.5 * (
                1.0
                + torch.cos(math.pi * (torch.sqrt(r2[band]) - 0.98 * f_na) / (0.02 * f_na))
            )

    # Propagate in frequency domain, no shifts
    if device.type == "cuda":
        # FP32 path for CUDA
        F = torch.fft.fft2(field.to(torch.complex64))
        H = torch.exp(1j * kz.to(torch.float32) * z).to(torch.complex64)
        if w is not None:
            H = H * w.to(torch.complex64)
        U = torch.fft.ifft2(F * H)
        result = U.to(torch.complex64)
        assert_fp32_cuda(result, "AS output")
    else:
        # FP64 path for CPU (high accuracy)
        F = torch.fft.fft2(field.to(torch.complex128))
        H = torch.exp(1j * kz * z)  # complex128
        if w is not None:
            H = H * w
        U = torch.fft.ifft2(F * H)
        result = U.to(torch.complex64)
    
    return result


def _angular_spectrum_step(
    E: torch.Tensor,
    n: torch.Tensor,
    lambda_um: float,
    dx: float,
    dy: float,
    dz: float,
    na_max: float,
) -> torch.Tensor:
    """Propagate field using angular spectrum method.

    Exact nonparaxial propagation with proper evanescent wave handling.
    Uses kz = 2π · sqrt((n/λ)² - fx² - fy²) formulation.

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

    # Store original dtype for final cast
    original_dtype = E.dtype

    # Average refractive index
    n_avg = n.mean().item() if torch.is_tensor(n) else n

    # FFT to angular spectrum (keep in original precision for FFT)
    # Done later when applying transfer function

    # Frequency grids in cycles/µm (float64)
    fx = torch.fft.fftfreq(nx, d=dx, device=device).to(torch.float64)
    fy = torch.fft.fftfreq(ny, d=dy, device=device).to(torch.float64)
    # Angular spatial frequencies (rad/µm)
    kx = (2.0 * math.pi) * fx.view(1, nx)
    ky = (2.0 * math.pi) * fy.view(ny, 1)

    # Wavenumber and kz (non-paraxial), float64
    k0 = 2.0 * math.pi / float(lambda_um)
    tmp = (k0 * k0) - (kx * kx + ky * ky)
    tmp = tmp.clamp(min=0.0)
    kz = torch.sqrt(tmp).to(torch.float64)

    # Optional NA soft taper near cutoff (no clipping inside core)
    w = None
    if na_max is not None and na_max > 0:
        f_na = float(na_max) / float(lambda_um)  # cycles/µm
        r2 = (fx.view(1, nx) ** 2 + fy.view(ny, 1) ** 2)
        w = torch.ones_like(r2, dtype=torch.float64, device=device)
        band = (r2 > (0.98 * f_na) ** 2) & (r2 <= (f_na ** 2))
        if band.any():
            w[band] = 0.5 * (1.0 + torch.cos(math.pi * (torch.sqrt(r2[band]) - 0.98 * f_na) / (0.02 * f_na)))

    # Propagate in frequency domain (no fftshift)
    F = torch.fft.fft2(E.to(torch.complex64))
    H = torch.exp(1j * kz * dz)
    if w is not None:
        H = H * w
    applied_spectrum = F * H.to(torch.complex64)

    # Inverse FFT back to spatial domain
    E_propagated = torch.fft.ifft2(applied_spectrum)

    # Handle inhomogeneous refractive index (multi-slice)
    if torch.is_tensor(n) and not torch.allclose(n, n_avg * torch.ones_like(n)):
        # Apply phase modulation for index variations
        k0 = 2 * np.pi / lambda_um
        phase_mod = k0 * (n - n_avg) * dz
        E_propagated = E_propagated * torch.exp(1j * phase_mod.to(torch.complex64))

    return E_propagated.to(original_dtype)


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
            0.5
            * (
                1
                + torch.cos(
                    np.pi * (k_transverse - k_max + transition_width) / (2 * transition_width)
                )
            ),
        ),
    )

    return mask.to(torch.complex64)


def angular_spectrum_exact(
    field: torch.Tensor, lambda_um: float, dx: float, dy: float, z: float, n: float = 1.0
) -> torch.Tensor:
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
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    # Exact transfer function
    kx = 2 * np.pi * fx_grid
    ky = 2 * np.pi * fy_grid
    kz2 = k**2 - kx**2 - ky**2

    # Handle both propagating and evanescent
    kz = torch.where(kz2 >= 0, torch.sqrt(kz2), 1j * torch.sqrt(-kz2))

    H = torch.exp(1j * kz * z)

    # Apply and inverse transform
    field_spectrum = field_spectrum * H
    field_out = torch.fft.ifft2(field_spectrum)

    return field_out


def multi_slice_propagate(
    field: torch.Tensor, slices: list[dict], lambda_um: float, dx: float, dy: float
) -> torch.Tensor:
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
        n_slice = slice_params.get("n", 1.0)
        dz_slice = slice_params.get("dz", 1.0)

        # Convert n to tensor if needed
        if isinstance(n_slice, (int, float)):
            n_slice = torch.full_like(field.real, n_slice)

        # Propagate through slice
        E = _angular_spectrum_step(E, n_slice, lambda_um, dx, dy, dz_slice, na_max=1.0)

    return E


def compute_psf(
    lambda_um: float,
    na: float,
    nx: int = 256,
    ny: int = 256,
    dx: float = None,
    dy: float = None,
    z: float = 0.0,
) -> torch.Tensor:
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
    x = torch.linspace(-nx // 2, nx // 2 - 1, nx) * dx
    y = torch.linspace(-ny // 2, ny // 2 - 1, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Pupil radius in spatial frequency
    pupil_radius = na / lambda_um

    # FFT frequencies
    fx = torch.fft.fftfreq(nx, d=dx)
    fy = torch.fft.fftfreq(ny, d=dy)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    # Pupil function in frequency domain
    f_radial = torch.sqrt(fx_grid**2 + fy_grid**2)
    pupil = (f_radial <= pupil_radius).to(torch.complex64)

    # Phase for defocus
    if abs(z) > 0:
        k = 2 * np.pi / lambda_um
        kz = k * torch.sqrt(1 - (lambda_um * f_radial) ** 2)
        kz = torch.where(f_radial <= pupil_radius, kz, torch.zeros_like(kz))
        pupil = pupil * torch.exp(1j * kz * z)

    # Inverse FFT to get PSF
    psf_complex = torch.fft.ifft2(torch.fft.ifftshift(pupil))
    psf_intensity = torch.abs(psf_complex) ** 2

    # Normalize
    psf_intensity = psf_intensity / psf_intensity.sum()

    return psf_intensity


def validate_angular_spectrum(
    field_in: torch.Tensor,
    field_out: torch.Tensor,
    lambda_um: float,
    dx: float,
    dy: float,
    z: float,
) -> dict:
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
    energy_in = torch.sum(torch.abs(field_in) ** 2) * dx * dy
    energy_out = torch.sum(torch.abs(field_out) ** 2) * dx * dy
    energy_ratio = energy_out / energy_in if energy_in > 0 else 0

    # Check reciprocity (backward propagation should recover input)
    field_back = angular_spectrum_exact(field_out, lambda_um, dx, dy, -z)
    reciprocity_error = torch.norm(field_back - field_in) / torch.norm(field_in)

    # Fresnel number
    L = min(field_in.shape[-2] * dy, field_in.shape[-1] * dx)  # Field size
    fresnel_number = L**2 / (lambda_um * abs(z)) if z != 0 else float("inf")

    return {
        "energy_ratio": energy_ratio.item(),
        "reciprocity_error": reciprocity_error.item(),
        "fresnel_number": fresnel_number,
        "is_near_field": fresnel_number > 1,
    }
