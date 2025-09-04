"""Enhanced Beam Propagation Method with vector wide-angle correction.

Implements split-step BPM with wide-angle corrections, adaptive stepping,
cosine windowing, and energy conservation auditing.
"""

from __future__ import annotations

import numpy as np
import torch


def run(
    field: torch.Tensor,
    plan: dict | None = None,
    sampler: dict | None = None,
    refractive_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run enhanced BPM vector wide-angle propagation.

    Split-step method with:
    - Wide-angle vector corrections for high-NA
    - Adaptive Δz based on field curvature
    - Cosine windowing for far-field evaluation
    - Energy conservation auditing

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

    # Extract parameters from plan
    dx = plan.get("dx_um", 0.5)
    dy = plan.get("dy_um", 0.5)
    dz_list = plan.get("dz_list_um", [1.0])
    wavelengths = plan.get("wavelengths_um", np.array([0.55]))
    na_max = plan.get("na_max", 0.25)

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

    # Create windowing functions
    cosine_window = _create_cosine_window(ny, nx, device)
    pml = _create_pml_graded(ny, nx, thickness=32, device=device)

    # Store initial energy for conservation check
    initial_energies = []
    for s in range(S):
        energy = torch.sum(torch.abs(field[s]) ** 2).item() * dx * dy
        initial_energies.append(energy)

    # Propagate each spectral component
    output_fields = []

    for s in range(S):
        lambda_um = wavelengths[s] if s < len(wavelengths) else wavelengths[0]
        k0 = 2 * np.pi / lambda_um

        # Current field with windowing
        E = field[s].clone()
        E = E * cosine_window * pml

        # Track energy through propagation
        energy_history = [initial_energies[s]]

        # Propagate through all z-steps with adaptive stepping
        for dz_nominal in dz_list:
            # Compute adaptive step size
            dz_steps = _compute_adaptive_steps(E, dx, dy, dz_nominal, k0, n, na_max)

            for dz in dz_steps:
                # Split-step propagation with wide-angle correction
                E = _propagate_step_enhanced(E, n, k0, dx, dy, dz, na_max, pml, cosine_window)

                # Monitor energy
                energy = torch.sum(torch.abs(E) ** 2).item() * dx * dy
                energy_history.append(energy)

        # Energy audit
        final_energy = energy_history[-1]
        energy_change = abs(final_energy - initial_energies[s]) / initial_energies[s]

        if energy_change > 0.01:
            print(f"Warning: Energy change {energy_change:.2%} exceeds 1% for λ={lambda_um}µm")

        output_fields.append(E)

    # Stack spectral components
    output = torch.stack(output_fields, dim=0)

    if squeeze_output:
        output = output.squeeze(0)

    return output


def _create_cosine_window(ny: int, nx: int, device: str) -> torch.Tensor:
    """Create cosine windowing function for far-field evaluation.

    Smooth transition from 1 in center to 0 at edges.

    Args:
        ny, nx: Grid dimensions
        device: Computation device

    Returns:
        Cosine window of shape (ny, nx)
    """
    # Create 1D cosine windows
    window_y = torch.hann_window(ny, periodic=False, device=device)
    window_x = torch.hann_window(nx, periodic=False, device=device)

    # Create 2D window as outer product
    window_2d = window_y.unsqueeze(1) * window_x.unsqueeze(0)

    # Modify to have wider flat center region
    # Use raised cosine with plateau
    y = torch.linspace(-1, 1, ny, device=device)
    x = torch.linspace(-1, 1, nx, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)

    # Raised cosine from r=0.7 to r=1.0
    window = torch.where(
        r < 0.7,
        torch.ones_like(r),
        torch.where(r > 1.0, torch.zeros_like(r), 0.5 * (1 + torch.cos(np.pi * (r - 0.7) / 0.3))),
    )

    return window


def _create_pml_graded(ny: int, nx: int, thickness: int, device: str) -> torch.Tensor:
    """Create graded PML with polynomial profile.

    Args:
        ny, nx: Grid dimensions
        thickness: PML thickness in pixels
        device: Computation device

    Returns:
        PML transmission mask
    """
    pml = torch.ones((ny, nx), dtype=torch.float32, device=device)

    if thickness <= 0:
        return pml

    # Create distance arrays
    y_idx = torch.arange(ny, device=device, dtype=torch.float32)
    x_idx = torch.arange(nx, device=device, dtype=torch.float32)

    # Distance from each edge
    for i in range(ny):
        for j in range(nx):
            d_top = i
            d_bottom = ny - 1 - i
            d_left = j
            d_right = nx - 1 - j

            d_min = min(d_top, d_bottom, d_left, d_right)

            if d_min < thickness:
                # Quartic polynomial grading for smooth absorption
                t = d_min / thickness
                pml[i, j] = t**4

    return pml


def _compute_adaptive_steps(
    E: torch.Tensor,
    dx: float,
    dy: float,
    dz_total: float,
    k0: float,
    n: torch.Tensor,
    na_max: float,
) -> list[float]:
    """Compute adaptive step sizes based on field curvature and CFL.

    Args:
        E: Current field
        dx, dy: Transverse grid spacing
        dz_total: Total propagation distance
        k0: Wave number in vacuum
        n: Refractive index
        na_max: Maximum NA

    Returns:
        List of adaptive step sizes
    """
    # Estimate phase curvature
    phase = torch.angle(E)

    # Phase gradients (avoiding boundary artifacts)
    pad = 5
    phase_interior = phase[pad:-pad, pad:-pad]

    if phase_interior.numel() > 0:
        grad_y = torch.gradient(phase_interior, dim=0)[0] / dy
        grad_x = torch.gradient(phase_interior, dim=1)[0] / dx

        # Maximum phase gradient
        max_grad = max(torch.abs(grad_y).max().item(), torch.abs(grad_x).max().item())
    else:
        max_grad = 0.0

    # Estimate local NA from phase gradient
    n_avg = n.mean().item()
    local_na = min(max_grad / (k0 * n_avg), na_max) if max_grad > 0 else 0.1

    # Stability criteria
    # 1. Diffraction-based limit
    dx_min = min(dx, dy)
    lambda_um = 2 * np.pi / k0

    if local_na > 0.8:
        stability_factor = 0.25  # Very conservative for high NA
    elif local_na > 0.5:
        stability_factor = 0.5  # Moderate
    else:
        stability_factor = 1.0  # Standard

    dz_diffraction = stability_factor * dx_min**2 / lambda_um

    # 2. CFL-like condition for numerical stability
    n_max = n.max().item()
    dz_cfl = 0.5 * dx_min / (n_max * np.sqrt(2))

    # 3. Curvature-based limit
    # Estimate from beam divergence
    if local_na > 0:
        rayleigh_range = np.pi * (dx_min * 10) ** 2 / lambda_um  # Estimate
        dz_curvature = rayleigh_range / 10  # Step size fraction of Rayleigh range
    else:
        dz_curvature = dz_total

    # Choose step size
    dz_max = min(dz_diffraction, dz_cfl, dz_curvature, dz_total)
    dz_max = max(dz_max, lambda_um)  # At least one wavelength

    # Generate adaptive steps
    if dz_total <= dz_max:
        return [dz_total]
    else:
        n_steps = int(np.ceil(dz_total / dz_max))
        dz_step = dz_total / n_steps
        return [dz_step] * n_steps


def _propagate_step_enhanced(
    E: torch.Tensor,
    n: torch.Tensor,
    k0: float,
    dx: float,
    dy: float,
    dz: float,
    na_max: float,
    pml: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Enhanced propagation step with wide-angle corrections.

    Args:
        E: Current field
        n: Refractive index
        k0: Wave number in vacuum
        dx, dy, dz: Grid spacings
        na_max: Maximum NA
        pml: PML mask
        window: Cosine window

    Returns:
        Propagated field
    """
    ny, nx = E.shape
    device = E.device

    # Average refractive index
    n_avg = n.mean().item()
    k_avg = k0 * n_avg

    # Step 1: Half-step propagation in Fourier domain
    E_fft = torch.fft.fft2(E)

    # Frequency grids
    fx = torch.fft.fftfreq(nx, d=dx, device=device, dtype=torch.float64)
    fy = torch.fft.fftfreq(ny, d=dy, device=device, dtype=torch.float64)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    # Wave vector components
    kx = 2 * np.pi * fx_grid
    ky = 2 * np.pi * fy_grid
    kt2 = kx**2 + ky**2

    # Band limiting
    kt_max = k_avg * na_max / n_avg
    band_limit = torch.sqrt(kt2) <= kt_max

    # Wide-angle propagator using Padé approximation
    # For (2,2) Padé of sqrt(1-x):
    kt2_norm = kt2 / k_avg**2
    kt2_norm = torch.clamp(kt2_norm, max=0.999)

    # Padé coefficients for better accuracy
    if na_max > 0.7:
        # (2,2) Padé for very high NA
        a0, a1, a2 = 1.0, -5 / 8, 1 / 8
        b0, b1, b2 = 1.0, -1 / 8, -1 / 8
    elif na_max > 0.4:
        # (1,1) Padé for moderate NA
        a0, a1, a2 = 1.0, -3 / 4, 0.0
        b0, b1, b2 = 1.0, -1 / 4, 0.0
    else:
        # Paraxial approximation for low NA
        a0, a1, a2 = 1.0, -1 / 2, 0.0
        b0, b1, b2 = 1.0, 0.0, 0.0

    numerator = a0 + a1 * kt2_norm + a2 * kt2_norm**2
    denominator = b0 + b1 * kt2_norm + b2 * kt2_norm**2

    # Avoid division by zero
    denominator = torch.where(
        torch.abs(denominator) > 1e-10, denominator, torch.ones_like(denominator)
    )

    kz_norm = numerator / denominator
    kz = k_avg * kz_norm

    # Apply band limit
    kz = torch.where(band_limit, kz, torch.zeros_like(kz))

    # Propagator for half step
    H_half = torch.exp(1j * kz * dz / 2)
    E_fft = E_fft * H_half.to(torch.complex64)

    # Back to spatial domain
    E = torch.fft.ifft2(E_fft)

    # Step 2: Apply refractive index modulation
    phase_mod = k0 * (n - n_avg) * dz
    E = E * torch.exp(1j * phase_mod.to(torch.complex64))

    # Step 3: Second half-step propagation
    E_fft = torch.fft.fft2(E)
    E_fft = E_fft * H_half.to(torch.complex64)
    E = torch.fft.ifft2(E_fft)

    # Step 4: Apply windowing and PML
    E = E * window * pml

    # Step 5: Vector correction for high NA
    if na_max > 0.5:
        E = _apply_vector_correction_enhanced(E, kx, ky, k_avg, band_limit)

    return E


def _apply_vector_correction_enhanced(
    E: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, k: float, band_limit: torch.Tensor
) -> torch.Tensor:
    """Enhanced vector correction for high-NA fields.

    Args:
        E: Scalar field
        kx, ky: Transverse wave vectors
        k: Wave number
        band_limit: Band limiting mask

    Returns:
        Vector-corrected field
    """
    # Compute kz with proper handling of evanescent waves
    kz2 = k**2 - kx**2 - ky**2
    kz2 = torch.clamp(kz2, min=0)
    kz = torch.sqrt(kz2)

    # Obliquity factor for vector fields
    # Accounts for change in field amplitude with propagation angle
    with torch.no_grad():
        kz_safe = torch.where(kz > 0.1 * k, kz, 0.1 * k)

        # Fresnel-Kirchhoff obliquity factor
        obliquity = torch.sqrt(kz_safe / k)

        # Additional polarization correction
        # For s-polarization: correction = 1
        # For p-polarization: correction = cos(theta)
        # Average assuming unpolarized light
        cos_theta = kz_safe / k
        pol_correction = (1 + cos_theta) / 2

        correction = obliquity * pol_correction
        correction = torch.where(band_limit, correction, torch.ones_like(correction))

    # Apply correction in Fourier domain
    E_fft = torch.fft.fft2(E)
    E_fft = E_fft * correction.to(torch.complex64)
    E = torch.fft.ifft2(E_fft)

    return E


def validate_energy_conservation(
    field_in: torch.Tensor, field_out: torch.Tensor, dx: float, dy: float, tolerance: float = 0.01
) -> dict:
    """Validate energy conservation with detailed diagnostics.

    Args:
        field_in: Input field
        field_out: Output field
        dx, dy: Grid spacing
        tolerance: Maximum allowed relative change

    Returns:
        Dictionary with energy metrics and pass/fail status
    """
    # Compute energies
    energy_in = torch.sum(torch.abs(field_in) ** 2).item() * dx * dy
    energy_out = torch.sum(torch.abs(field_out) ** 2).item() * dx * dy

    # Relative change
    if energy_in > 0:
        rel_change = abs(energy_out - energy_in) / energy_in
    else:
        rel_change = 0.0

    # Peak intensity change
    peak_in = torch.max(torch.abs(field_in) ** 2).item()
    peak_out = torch.max(torch.abs(field_out) ** 2).item()
    peak_change = abs(peak_out - peak_in) / peak_in if peak_in > 0 else 0.0

    return {
        "energy_in": energy_in,
        "energy_out": energy_out,
        "relative_change": rel_change,
        "peak_change": peak_change,
        "passed": rel_change <= tolerance,
        "tolerance": tolerance,
    }
