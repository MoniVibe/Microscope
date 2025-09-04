"""Propagation planning and grid management.

Computes optimal grid parameters, step sizes, and memory budgets
for optical propagation simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Plan:
    """Propagation plan with grid and stepping parameters.

    Attributes:
        dx_um, dy_um: Grid spacing in micrometers
        dz_list_um: List of z-steps for propagation
        nx, ny: Grid dimensions
        spectral_samples: Number of spectral samples
        angular_samples: Number of angular samples
        guard_band_px: Guard band size in pixels
        pml_thickness_px: PML absorber thickness
        wavelengths_um: Array of wavelengths
        na_max: Maximum numerical aperture
        memory_estimate_gb: Estimated memory usage
        preset: Configuration preset name
    """

    dx_um: float
    dy_um: float
    dz_list_um: list[float]
    nx: int
    ny: int
    spectral_samples: int = 1
    angular_samples: int = 1
    guard_band_px: int = 0
    pml_thickness_px: int = 16
    wavelengths_um: np.ndarray = field(default_factory=lambda: np.array([0.55]))
    na_max: float = 0.25
    memory_estimate_gb: float = 0.0
    preset: str = "Standard"

    # Additional planning parameters
    total_z_um: float = 0.0
    num_z_planes: int = 1
    band_limit_factor: float = 1.0
    use_mixed_precision: bool = False

    def get_memory_estimate(self) -> float:
        """Calculate memory estimate for this plan.

        Returns:
            Estimated memory usage in GB
        """
        # Complex field array size
        bytes_per_complex = 8 if not self.use_mixed_precision else 4

        # Main field arrays (need at least 2 for propagation)
        field_size = self.ny * self.nx * bytes_per_complex
        num_fields = 2  # Current and next

        # Spectral samples
        total_fields = num_fields * self.spectral_samples * self.angular_samples

        # FFT workspace (typically needs 2x field size)
        fft_workspace = 2 * field_size

        # PML masks and other auxiliary arrays
        aux_arrays = 4 * self.ny * self.nx * 4  # float32

        # Total in bytes
        total_bytes = total_fields * field_size + fft_workspace + aux_arrays

        # Convert to GB
        total_gb = total_bytes / (1024**3)

        return total_gb


def make_plan(cfg: dict) -> Plan:
    """Create propagation plan from configuration.

    Args:
        cfg: Validated configuration dictionary

    Returns:
        Plan object with computed parameters
    """
    # Extract key parameters
    na_max = cfg.get("NA_max", 0.25)

    # Wavelength range
    lambda_dict = cfg.get("lambda_um", {"min": 0.4, "max": 0.7})
    if isinstance(lambda_dict, dict):
        lambda_min = lambda_dict.get("min", 0.4)
        lambda_max = lambda_dict.get("max", 0.7)
    else:
        lambda_min = lambda_max = lambda_dict

    # Grid parameters
    grid_cfg = cfg.get("grid", {})
    target_px = grid_cfg.get("target_px", 1024)

    # Determine preset
    preset = cfg.get("preset", "Standard")
    if preset not in ["Standard", "High-NA", "Aggressive"]:
        preset = "Standard"

    # Runtime budget
    runtime = cfg.get("runtime", {})
    budget = runtime.get("budget", {})
    vram_gb = budget.get("vram_gb", 10.0)

    # Calculate grid spacing based on NA and wavelength
    # Nyquist criterion: Δx ≤ λ/(2·NA)
    # High-NA preset uses tighter sampling
    NA_HIGH = 0.8  # PLR2004 named threshold
    if preset == "High-NA" or na_max > NA_HIGH:
        # Tighter sampling for high NA
        dx_factor = 3.0  # Δx ≤ λ/(3·NA)
    elif preset == "Aggressive":
        # Looser sampling for speed
        dx_factor = 2.0  # Δx ≤ λ/(2·NA)
    else:
        # Standard sampling
        dx_factor = 2.5  # Between 2 and 3

    dx_max = lambda_min / (dx_factor * na_max)

    # Check if user specified pitch
    if "pitch_um" in grid_cfg:
        dx = dy = grid_cfg["pitch_um"]
        # Validate against Nyquist
        if dx > dx_max:
            print(f"Warning: Specified pitch {dx:.3f} µm exceeds Nyquist limit {dx_max:.3f} µm")
    else:
        # Use Nyquist-limited spacing
        dx = dy = dx_max

    # Round to nice values
    dx = dy = round(dx * 1000) / 1000  # Round to nm

    # Determine grid size
    # Balance between field of view and memory constraints
    nx = ny = target_px

    # Adjust for memory constraints
    max_size = _compute_max_grid_size(vram_gb, na_max > NA_HIGH)
    if nx > max_size:
        nx = ny = max_size
        print(f"Grid size reduced to {nx}x{ny} to fit in {vram_gb:.1f} GB VRAM")

    # Ensure power of 2 for FFT efficiency (optional but recommended)
    if preset != "Aggressive":
        nx = ny = _next_power_of_2(min(nx, ny))

    # Guard bands and PML
    if preset == "High-NA":
        guard_band_px = 32
        pml_thickness_px = 32
    elif preset == "Aggressive":
        guard_band_px = 8
        pml_thickness_px = 8
    else:
        guard_band_px = 16
        pml_thickness_px = 16

    # Spectral sampling
    sources_cfg = cfg.get("sources", [])
    spectral_samples = 1
    wavelengths_um = np.array([lambda_min])

    if sources_cfg:
        # Get spectral samples from first source
        src = sources_cfg[0]
        if "spectral_samples" in src:
            spectral_samples = src["spectral_samples"]
        elif "bandwidth_um" in src:
            # Estimate samples based on bandwidth
            bw = src.get("bandwidth_um", 0.01)
            center = src.get("center_um", 0.55)
            rel_bw = bw / center

            REL_BW_NARROW = 0.01  # PLR2004 named threshold
            REL_BW_MEDIUM = 0.05  # PLR2004 named threshold
            if rel_bw < REL_BW_NARROW:
                spectral_samples = 3
            elif rel_bw < REL_BW_MEDIUM:
                spectral_samples = 7
            else:
                spectral_samples = 11

        # Generate wavelength array
        if spectral_samples > 1:
            center_um = src.get("center_um", (lambda_min + lambda_max) / 2)
            bandwidth_um = src.get("bandwidth_um", 0.01)

            # Sample wavelengths
            wavelengths_um = np.linspace(
                center_um - bandwidth_um / 2, center_um + bandwidth_um / 2, spectral_samples
            )

    # Angular sampling (for future extension)
    angular_samples = 1

    # Z-stepping
    dz_list_um = _compute_z_steps(cfg, dx, lambda_min, na_max, preset)

    # Band limit factor
    band_limit_factor = 1.0
    if preset == "High-NA":
        band_limit_factor = 1.2  # Allow slightly beyond NA limit
    elif preset == "Aggressive":
        band_limit_factor = 0.9  # Strict band limiting

    # Mixed precision
    VRAM_LOW_GB = 8.0  # PLR2004 named threshold
    use_mixed_precision = preset == "Aggressive" or vram_gb < VRAM_LOW_GB

    # Create plan
    plan = Plan(
        dx_um=dx,
        dy_um=dy,
        dz_list_um=dz_list_um,
        nx=nx,
        ny=ny,
        spectral_samples=spectral_samples,
        angular_samples=angular_samples,
        guard_band_px=guard_band_px,
        pml_thickness_px=pml_thickness_px,
        wavelengths_um=wavelengths_um,
        na_max=na_max,
        preset=preset,
        band_limit_factor=band_limit_factor,
        use_mixed_precision=use_mixed_precision,
    )

    # Calculate memory estimate
    plan.memory_estimate_gb = plan.get_memory_estimate()

    # Validate memory usage
    if plan.memory_estimate_gb > vram_gb:
        warn_prefix = "Warning: Estimated memory "
        mem_str = f"{plan.memory_estimate_gb:.2f} GB"
        budget_str = f"{vram_gb:.1f} GB"
        print(f"{warn_prefix}{mem_str} exceeds budget {budget_str}")

    return plan


def _compute_max_grid_size(vram_gb: float, high_na: bool = False) -> int:
    """Compute maximum grid size that fits in memory.

    Args:
        vram_gb: Available VRAM in GB
        high_na: Whether high-NA mode (needs more memory)

    Returns:
        Maximum grid dimension (assumes square grid)
    """
    # Bytes per complex number
    bytes_per_complex = 8  # complex64

    # Memory overhead factor (FFTs, temp arrays, etc.)
    overhead_factor = 3.0 if high_na else 2.5

    # Available bytes
    available_bytes = vram_gb * 1024**3

    # Usable bytes (leave some headroom)
    usable_bytes = available_bytes * 0.8

    # Grid size
    # Memory ~ N^2 * bytes_per_complex * overhead_factor
    max_n_squared = usable_bytes / (bytes_per_complex * overhead_factor)
    max_n = int(math.sqrt(max_n_squared))

    # Round down to multiple of 32 for alignment
    max_n = (max_n // 32) * 32

    return max_n


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    return 2 ** math.ceil(math.log2(n))


def _compute_z_steps(
    cfg: dict, dx: float, lambda_min: float, na_max: float, preset: str
) -> list[float]:
    """Compute adaptive z-steps for propagation.

    Args:
        cfg: Configuration dictionary
        dx: Grid spacing in micrometers
        lambda_min: Minimum wavelength
        na_max: Maximum NA
        preset: Configuration preset

    Returns:
        List of z-steps in micrometers
    """
    # Get propagation distance
    components = cfg.get("components", [])
    recorders = cfg.get("recorders", [])

    # Find total propagation distance from recorders
    z_positions = [0.0]
    for recorder in recorders:
        if "z_um" in recorder:
            z_positions.append(recorder["z_um"])

    if len(z_positions) > 1:
        total_z = max(z_positions) - min(z_positions)
    else:
        # Default propagation distance
        total_z = 100.0  # 100 µm default

    if total_z <= 0:
        return [1.0]  # Single step

    # Maximum step size based on Fresnel number
    # Δz_max = Δx² / λ for paraxial
    # For wide-angle, use more conservative stepping
    if preset == "High-NA" or na_max > NA_HIGH:
        # Very fine stepping for high NA
        dz_max = 0.5 * dx**2 / lambda_min
    elif preset == "Aggressive":
        # Coarser stepping for speed
        dz_max = 2.0 * dx**2 / lambda_min
    else:
        # Standard stepping
        dz_max = dx**2 / lambda_min

    # Minimum step size (to avoid numerical issues)
    dz_min = lambda_min / 10

    # Cap step size
    dz_max = min(dz_max, total_z / 10)  # At least 10 steps
    dz_max = max(dz_max, dz_min)

    # Generate adaptive steps
    dz_list = []
    z_current = 0.0

    while z_current < total_z:
        # Adaptive step based on position
        # Could be refined based on field curvature
        if z_current < total_z * 0.1:
            # Fine steps near source
            dz = dz_max * 0.5
        elif z_current > total_z * 0.9:
            # Fine steps near observation
            dz = dz_max * 0.5
        else:
            # Standard steps in bulk
            dz = dz_max

        # Don't overshoot
        if z_current + dz > total_z:
            dz = total_z - z_current

        dz_list.append(dz)
        z_current += dz

    return dz_list


def validate_plan(plan: Plan, cfg: dict) -> bool:
    """Validate that plan meets configuration requirements.

    Args:
        plan: Propagation plan
        cfg: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check Nyquist sampling
    lambda_min = plan.wavelengths_um.min()
    dx_nyquist = lambda_min / (2 * plan.na_max)

    if plan.dx_um > dx_nyquist * 1.1:  # 10% tolerance
        raise ValueError(
            f"Grid spacing {plan.dx_um:.3f} µm violates Nyquist criterion "
            f"(max {dx_nyquist:.3f} µm for NA={plan.na_max:.2f})"
        )

    # Check memory
    if plan.memory_estimate_gb > cfg["runtime"]["budget"]["vram_gb"] * 1.2:  # 20% tolerance
        raise ValueError(
            f"Memory estimate {plan.memory_estimate_gb:.2f} GB exceeds "
            f"budget {cfg['runtime']['budget']['vram_gb']:.1f} GB"
        )

    # Check grid size
    if plan.nx < 64 or plan.ny < 64:
        raise ValueError(f"Grid size {plan.nx}x{plan.ny} too small (min 64x64)")

    if plan.nx > 16384 or plan.ny > 16384:
        raise ValueError(f"Grid size {plan.nx}x{plan.ny} too large (max 16384x16384)")

    return True
