"""Enhanced Split-Step Fourier BPM with energy conservation.

Implements scalar wide-angle BPM with:
- Frequency-domain propagation
- Cosine windowing for far-field
- Energy conservation auditing
- Adaptive step sizing
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
    """Run enhanced split-step Fourier BPM.

    Features:
    - Wide-angle propagation using Padé approximants
    - Cosine windowing for far-field evaluation
    - Energy conservation monitoring
    - Adaptive z-stepping based on field curvature

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

    # Create windowing and PML
    cosine_window = _create_cosine_taper(ny, nx, device)
    pml = _create_smooth_pml(ny, nx, thickness=32, device=device)
    band_mask = _create_band_mask(ny, nx, dx, dy, na_max, wavelengths[0], n.mean().item(), device)

    # Energy tracking
    initial_energies = []
    for s in range(S):
        E_windowed = field[s] * cosine_window * pml
        energy = _compute_energy(E_windowed, dx, dy)
        initial_energies.append(energy)

    # Process each spectral component
    output_fields = []

    for s in range(S):
        lambda_um = wavelengths[s] if s < len(wavelengths) else wavelengths[0]
        k0 = 2 * np.pi / lambda_um

        # Apply initial windowing
        E = field[s].clone()
        E = E * cosine_window * pml

        # Track energy
        energy_log = [initial_energies[s]]

        # Propagate through z-steps with adaptive stepping
        for dz_nominal in dz_list:
            # Compute adaptive steps
            adaptive_steps = _compute_adaptive_dz(E, dx, dy, dz_nominal, k0, n, na_max)

            for dz in adaptive_steps:
                E = _split_step_enhanced(
                    E, n, k0, dx, dy, dz, na_max, pml, band_mask, cosine_window
                )

                # Monitor energy
                energy = _compute_energy(E, dx, dy)
                energy_log.append(energy)

        # Energy audit
        final_energy = energy_log[-1]
        energy_error = abs(final_energy - initial_energies[s]) / initial_energies[s]

        if energy_error > 0.01:
            print(f"Energy conservation warning: {energy_error:.2%} change for λ={lambda_um}µm")

        output_fields.append(E)

    output = torch.stack(output_fields, dim=0)

    if squeeze_output:
        output = output.squeeze(0)

    return output


def _create_cosine_taper(ny: int, nx: int, device: str) -> torch.Tensor:
    """Create cosine taper window for far-field evaluation.

    Smooth transition from center to edges prevents diffraction artifacts.

    Args:
        ny, nx: Grid dimensions
        device: Computation device

    Returns:
        2D cosine taper window
    """
    # Create 1D tapers
    taper_y = torch.ones(ny, device=device)
    taper_x = torch.ones(nx, device=device)

    # Taper width (10% of each dimension)
    taper_width_y = max(int(0.1 * ny), 10)
    taper_width_x = max(int(0.1 * nx), 10)

    # Apply cosine taper to edges
    for i in range(taper_width_y):
        val = 0.5 * (1 - np.cos(np.pi * i / taper_width_y))
        taper_y[i] = val
        taper_y[-(i + 1)] = val

    for j in range(taper_width_x):
        val = 0.5 * (1 - np.cos(np.pi * j / taper_width_x))
        taper_x[j] = val
        taper_x[-(j + 1)] = val

    # Create 2D window
    window = taper_y.unsqueeze(1) * taper_x.unsqueeze(0)

    return window


def _create_smooth_pml(ny: int, nx: int, thickness: int, device: str) -> torch.Tensor:
    """Create smooth PML absorber with quartic profile.

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

    # Quartic profile for smooth absorption
    for i in range(ny):
        for j in range(nx):
            # Distance from edges
            d_y = min(i, ny - 1 - i)
            d_x = min(j, nx - 1 - j)
            d_min = min(d_y, d_x)

            if d_min < thickness:
                # Quartic profile: smoother than cubic
                t = d_min / thickness
                pml[i, j] = t**4

    return pml


def _create_band_mask(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    na_max: float,
    lambda_um: float,
    n_avg: float,
    device: str,
) -> torch.Tensor:
    """Create frequency domain band-limiting mask with smooth edge.

    Args:
        ny, nx: Grid dimensions
        dx, dy: Grid spacing
        na_max: Maximum NA
        lambda_um: Wavelength
        n_avg: Average refractive index
        device: Computation device

    Returns:
        Band limit mask
    """
    # Frequency grids
    fx = torch.fft.fftfreq(nx, d=dx, device=device)
    fy = torch.fft.fftfreq(ny, d=dy, device=device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    # Maximum spatial frequency from NA
    f_max = na_max / lambda_um
    f_radial = torch.sqrt(fx_grid**2 + fy_grid**2)

    # Smooth transition using error function profile
    transition_width = 0.05 * f_max

    # Create smooth mask
    mask = torch.where(
        f_radial <= f_max - transition_width,
        torch.ones_like(f_radial),
        torch.where(
            f_radial >= f_max + transition_width,
            torch.zeros_like(f_radial),
            0.5 * (1 - torch.tanh(5 * (f_radial - f_max) / transition_width)),
        ),
    )

    return mask


def _compute_energy(field: torch.Tensor, dx: float, dy: float) -> float:
    """Compute total field energy.

    Args:
        field: Complex field
        dx, dy: Grid spacing

    Returns:
        Total energy
    """
    intensity = torch.abs(field) ** 2
    energy = torch.sum(intensity).item() * dx * dy
    return energy


def _compute_adaptive_dz(
    E: torch.Tensor,
    dx: float,
    dy: float,
    dz_total: float,
    k0: float,
    n: torch.Tensor,
    na_max: float,
) -> list[float]:
    """Compute adaptive z-steps based on field properties.

    Args:
        E: Current field
        dx, dy: Transverse grid spacing
        dz_total: Total propagation distance
        k0: Wave number
        n: Refractive index
        na_max: Maximum NA

    Returns:
        List of adaptive step sizes
    """
    # Analyze field to determine optimal stepping
    ny, nx = E.shape

    # Compute phase gradients (avoiding edges)
    phase = torch.angle(E)
    pad = 10
    if ny > 2 * pad and nx > 2 * pad:
        phase_center = phase[pad:-pad, pad:-pad]

        # Gradients
        grad_y = torch.diff(phase_center, dim=0) / dy
        grad_x = torch.diff(phase_center, dim=1) / dx

        # Maximum gradient (related to local NA)
        max_grad = max(torch.abs(grad_y).max().item(), torch.abs(grad_x).max().item())
    else:
        max_grad = 0.0

    # Estimate local NA
    n_avg = n.mean().item()
    lambda_um = 2 * np.pi / k0

    if max_grad > 0:
        local_na = min(max_grad * lambda_um / (2 * np.pi * n_avg), na_max)
    else:
        local_na = 0.1

    # Determine step size constraints
    dx_min = min(dx, dy)

    # 1. Diffraction limit (more conservative for high NA)
    if local_na > 0.8:
        factor = 0.2
    elif local_na > 0.5:
        factor = 0.4
    else:
        factor = 1.0

    dz_diffraction = factor * dx_min**2 / lambda_um

    # 2. CFL-like stability
    n_max = n.max().item()
    dz_cfl = dx_min / (2 * n_max)

    # 3. Field curvature constraint
    # Estimate from intensity distribution
    intensity = torch.abs(E) ** 2
    if intensity.max() > 0:
        # Find beam width
        threshold = 0.135 * intensity.max()  # 1/e² threshold
        above_threshold = intensity > threshold
        if above_threshold.any():
            # Estimate beam size
            y_indices, x_indices = torch.where(above_threshold)
            beam_width = min(
                (x_indices.max() - x_indices.min()).item() * dx,
                (y_indices.max() - y_indices.min()).item() * dy,
            )
            if beam_width > 0:
                # Rayleigh range estimate
                z_R = np.pi * beam_width**2 / (4 * lambda_um)
                dz_curvature = z_R / 5  # Step = 1/5 of Rayleigh range
            else:
                dz_curvature = dz_total
        else:
            dz_curvature = dz_total
    else:
        dz_curvature = dz_total

    # Choose minimum step size
    dz_max = min(dz_diffraction, dz_cfl, dz_curvature, dz_total)
    dz_max = max(dz_max, lambda_um)  # At least one wavelength

    # Generate steps
    if dz_total <= dz_max:
        return [dz_total]
    else:
        n_steps = int(np.ceil(dz_total / dz_max))
        dz_step = dz_total / n_steps
        return [dz_step] * n_steps


def _split_step_enhanced(
    E: torch.Tensor,
    n: torch.Tensor,
    k0: float,
    dx: float,
    dy: float,
    dz: float,
    na_max: float,
    pml: torch.Tensor,
    band_mask: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Enhanced split-step Fourier propagation.

    Args:
        E: Current field
        n: Refractive index
        k0: Vacuum wave number
        dx, dy, dz: Grid spacings
        na_max: Maximum NA
        pml: PML mask
        band_mask: Frequency band limit
        window: Cosine window

    Returns:
        Propagated field
    """
    ny, nx = E.shape
    device = E.device
    n_avg = n.mean()
    k_avg = k0 * n_avg

    # Step 1: Apply half of the refractive index phase
    phase_spatial = 0.5 * k0 * (n - n_avg) * dz
    E = E * torch.exp(1j * phase_spatial.to(torch.complex64))

    # Step 2: Fourier transform
    E_fft = torch.fft.fft2(E)

    # Step 3: Apply propagation in frequency domain
    fx = torch.fft.fftfreq(nx, d=dx, device=device, dtype=torch.float64)
    fy = torch.fft.fftfreq(ny, d=dy, device=device, dtype=torch.float64)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")

    # Wave vectors
    kx = 2 * np.pi * fx_grid
    ky = 2 * np.pi * fy_grid
    kt2 = kx**2 + ky**2

    # Wide-angle propagator using Padé approximation
    kt2_norm = kt2 / k_avg**2
    kt2_norm = torch.clamp(kt2_norm, max=0.999)

    # Choose Padé order based on NA
    if na_max > 0.7:
        # (2,2) Padé for high accuracy at large angles
        numerator = 1 - 5 * kt2_norm / 8 + kt2_norm**2 / 8
        denominator = 1 - kt2_norm / 8 - kt2_norm**2 / 8
    elif na_max > 0.4:
        # (1,1) Padé for moderate angles
        numerator = 1 - 3 * kt2_norm / 4
        denominator = 1 - kt2_norm / 4
    else:
        # Paraxial approximation
        numerator = 1 - kt2_norm / 2
        denominator = torch.ones_like(kt2_norm)

    # Avoid division by zero
    denominator = torch.where(
        torch.abs(denominator) > 1e-10, denominator, torch.ones_like(denominator)
    )

    # Propagation phase
    kz_norm = numerator / denominator
    kz = k_avg * torch.abs(kz_norm)  # Ensure positive for propagating waves

    # Handle evanescent waves
    is_propagating = kt2 < k_avg**2
    kz = torch.where(is_propagating, kz, torch.zeros_like(kz))

    # Apply propagator with band limiting
    prop_phase = kz * dz * band_mask
    propagator = torch.exp(1j * prop_phase.to(torch.complex64))

    # Add evanescent decay for high NA
    if na_max > 0.6:
        decay = torch.where(
            ~is_propagating,
            torch.exp(-torch.sqrt(torch.abs(kt2 - k_avg**2)) * dz),
            torch.ones_like(kz),
        )
        propagator = propagator * decay.to(torch.complex64)

    E_fft = E_fft * propagator

    # Step 4: Inverse transform
    E = torch.fft.ifft2(E_fft)

    # Step 5: Apply second half of refractive index phase
    E = E * torch.exp(1j * phase_spatial.to(torch.complex64))

    # Step 6: Apply windowing and PML
    E = E * window * pml

    return E


def energy_audit(
    field_in: torch.Tensor, field_out: torch.Tensor, dx: float, dy: float, verbose: bool = False
) -> dict:
    """Perform detailed energy conservation audit.

    Args:
        field_in: Input field
        field_out: Output field
        dx, dy: Grid spacing
        verbose: Print detailed diagnostics

    Returns:
        Dictionary with energy metrics
    """
    # Total energies
    energy_in = _compute_energy(field_in, dx, dy)
    energy_out = _compute_energy(field_out, dx, dy)

    # Relative change
    if energy_in > 0:
        rel_change = (energy_out - energy_in) / energy_in
        abs_change = abs(rel_change)
    else:
        rel_change = 0.0
        abs_change = 0.0

    # Peak intensities
    peak_in = torch.max(torch.abs(field_in) ** 2).item()
    peak_out = torch.max(torch.abs(field_out) ** 2).item()
    peak_ratio = peak_out / peak_in if peak_in > 0 else 1.0

    # Power in central region (50% of grid)
    ny, nx = field_in.shape
    cy, cx = ny // 4, nx // 4
    center_in = field_in[cy:-cy, cx:-cx]
    center_out = field_out[cy:-cy, cx:-cx]

    energy_center_in = _compute_energy(center_in, dx, dy)
    energy_center_out = _compute_energy(center_out, dx, dy)
    center_ratio = energy_center_out / energy_center_in if energy_center_in > 0 else 1.0

    result = {
        "energy_in": energy_in,
        "energy_out": energy_out,
        "relative_change": rel_change,
        "absolute_change": abs_change,
        "peak_ratio": peak_ratio,
        "center_ratio": center_ratio,
        "passed": abs_change <= 0.01,
    }

    if verbose:
        print("Energy Audit:")
        print(f"  Input energy:  {energy_in:.6e}")
        print(f"  Output energy: {energy_out:.6e}")
        print(f"  Change: {rel_change:+.2%}")
        print(f"  Peak ratio: {peak_ratio:.3f}")
        print(f"  Center ratio: {center_ratio:.3f}")
        print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")

    return result
