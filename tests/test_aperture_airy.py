"""Tests for aperture diffraction (Airy pattern)."""

import numpy as np
import torch

from optics_sim.prop.solvers import as_multi_slice
from optics_sim.validation.cases import aperture_diffraction


def test_aperture_airy():
    """Test circular aperture diffraction pattern."""
    wavelength_um = 0.55
    aperture_diameter_um = 100.0
    z_um = 10000.0  # Far field

    nx = ny = 256
    dx = dy = aperture_diameter_um / 40

    # Get analytical Airy pattern
    aperture, airy_analytical = aperture_diffraction(
        wavelength_um, aperture_diameter_um, z_um, nx, ny, dx, dy
    )

    # Propagate with angular spectrum
    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [z_um],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": 0.5,
    }

    airy_computed = as_multi_slice.run(aperture, plan)

    # Find peak position
    peak_idx = torch.argmax(torch.abs(airy_computed) ** 2)
    peak_y, peak_x = peak_idx // nx, peak_idx % nx
    center_y, center_x = ny // 2, nx // 2

    # Peak should be near center (within 2 pixels)
    assert abs(peak_y - center_y) <= 2, "Airy peak not centered in Y"
    assert abs(peak_x - center_x) <= 2, "Airy peak not centered in X"

    # Check first zero position (approximate)
    # For circular aperture: first zero at θ ≈ 1.22 λ/D
    theta_first_zero = 1.22 * wavelength_um / aperture_diameter_um
    r_first_zero = z_um * np.tan(theta_first_zero)
    pixels_to_zero = r_first_zero / dx

    # Extract radial profile
    intensity = torch.abs(airy_computed) ** 2
    profile_x = intensity[center_y, :]

    # Find first minimum
    center_intensity = profile_x[center_x]
    for i in range(center_x + 1, nx):
        if profile_x[i] > profile_x[i - 1]:  # Start rising again
            first_min_idx = i - 1
            break
    else:
        first_min_idx = nx - 1

    measured_pixels = abs(first_min_idx - center_x)

    # Allow numerical discretization and sampling effects in angular spectrum simulation
    error = abs(measured_pixels - pixels_to_zero) / pixels_to_zero
    assert error <= 0.12, f"First zero position error {error:.1%} exceeds 12%"

    print(
        f"✓ Airy pattern: peak centered, first zero at {measured_pixels:.1f} pixels (expected {pixels_to_zero:.1f})"
    )


if __name__ == "__main__":
    test_aperture_airy()
    print("\nAperture diffraction test passed!")
