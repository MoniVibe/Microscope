"""Sampling heuristics for grid and propagation parameters.

Determines grid size, pitch, and z-steps based on NA, wavelength, and FOV.
"""

from typing import Any

from .errors import SamplingError


def calculate_sampling(
    na_max: float,
    lambda_min_um: float,
    lambda_max_um: float,
    fov_um: tuple[float, float] | None = None,
    target_px: int | None = None,
    vram_gb: float = 10.0,
) -> dict[str, Any]:
    """Calculate sampling parameters from optical parameters.

    Args:
        na_max: Maximum numerical aperture
        lambda_min_um: Minimum wavelength in micrometers
        lambda_max_um: Maximum wavelength in micrometers
        fov_um: Field of view (width, height) in micrometers
        target_px: Target grid size in pixels
        vram_gb: Available VRAM in GB

    Returns:
        Dictionary with sampling parameters:
            - dx, dy: Grid pitch in micrometers
            - dz: Z-step size in micrometers
            - nx, ny: Grid dimensions in pixels
            - nz: Number of z-steps
            - memory_estimate_gb: Estimated memory usage

    Raises:
        SamplingError: If parameters lead to impractical sampling
    """
    # Nyquist sampling criterion for spatial resolution
    # Need at least 2 samples per minimum resolvable feature
    dx_max = lambda_min_um / (2.0 * na_max)

    # More conservative for high NA
    if na_max > 0.7:
        dx_max = lambda_min_um / (3.0 * na_max)

    # Default FOV if not specified
    if fov_um is None:
        # Estimate from wavelength and NA
        fov_um = (100 * lambda_max_um, 100 * lambda_max_um)

    # Grid dimensions
    if target_px is None:
        # Choose based on NA
        if na_max <= 0.3:
            target_px = 1024
        elif na_max <= 0.7:
            target_px = 1536
        else:
            target_px = 2048

    # Ensure power of 2 for FFT efficiency (round up)
    import math

    target_px = 2 ** math.ceil(math.log2(target_px))

    # Calculate actual grid parameters
    nx = ny = target_px
    dx = dy = max(dx_max, fov_um[0] / nx)

    # Check if sampling is adequate
    if dx > 2 * dx_max:
        import warnings

        warnings.warn(
            f"Grid pitch {dx:.3f} µm exceeds Nyquist limit {dx_max:.3f} µm. "
            "Consider increasing grid size or reducing FOV.",
            category=UserWarning,
        )

    # Z-stepping heuristics
    # Based on Rayleigh range and depth of focus
    rayleigh_range = lambda_max_um / (na_max**2)
    dz = min(rayleigh_range / 2, 1.0)  # Conservative z-step

    # Number of z-steps (typical microscope column)
    z_range = 100 * lambda_max_um  # Default propagation range
    nz = int(z_range / dz)
    nz = min(nz, 256)  # Cap for memory

    # Memory estimate (complex field, double precision)
    bytes_per_complex = 16  # 8 bytes real + 8 bytes imag
    field_size_gb = (nx * ny * bytes_per_complex) / 1e9

    # Multiple fields for propagation (current, next, work arrays)
    num_fields = 4  # Conservative estimate
    memory_estimate_gb = field_size_gb * num_fields

    # Add overhead for FFT and other operations
    memory_estimate_gb *= 1.5

    # Check memory constraint
    if memory_estimate_gb > vram_gb:
        # Try to reduce
        if target_px > 512:
            # Recursively try smaller grid
            return calculate_sampling(
                na_max, lambda_min_um, lambda_max_um, fov_um, target_px // 2, vram_gb
            )
        else:
            raise SamplingError(
                f"Cannot fit simulation in {vram_gb:.1f} GB VRAM. "
                f"Estimated requirement: {memory_estimate_gb:.1f} GB"
            )

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "memory_estimate_gb": memory_estimate_gb,
        "fov_um": (nx * dx, ny * dy),
    }


def validate_sampling(params: dict[str, Any]) -> bool:
    """Validate sampling parameters are reasonable.

    Args:
        params: Sampling parameters dictionary

    Returns:
        True if valid, False otherwise
    """
    # Check grid size
    if not (64 <= params["nx"] <= 16384):
        return False
    if not (64 <= params["ny"] <= 16384):
        return False

    # Check pitch
    if not (0.01 <= params["dx"] <= 100):
        return False
    if not (0.01 <= params["dy"] <= 100):
        return False

    # Check z-steps
    if not (1 <= params["nz"] <= 1024):
        return False
    if not (0.01 <= params["dz"] <= 100):
        return False

    return True


def suggest_padding(na: float, lambda_um: float, grid_size: int) -> int:
    """Suggest padding size for PML or guard bands.

    Args:
        na: Numerical aperture
        lambda_um: Wavelength in micrometers
        grid_size: Current grid size

    Returns:
        Suggested padding in pixels
    """
    # More padding for higher NA (larger angular spectrum)
    base_padding = max(16, int(0.05 * grid_size))

    if na > 0.7:
        padding = int(base_padding * 1.5)
    elif na > 0.3:
        padding = base_padding
    else:
        padding = max(8, base_padding // 2)

    return min(padding, grid_size // 4)


__all__ = [
    "calculate_sampling",
    "validate_sampling",
    "suggest_padding",
]
