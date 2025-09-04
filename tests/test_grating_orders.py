"""Tests for phase grating diffraction orders."""

import numpy as np
import torch
from optics_sim.prop.solvers import as_multi_slice
from optics_sim.validation.cases import phase_grating_orders
from scipy.special import jn

# Named tolerances (PLR2004)
EFF_THRESHOLD = 0.01
ORDER_ERROR_TOL = 0.03


def test_grating_orders():
    """Test diffraction orders from sinusoidal phase grating."""
    wavelength_um = 0.55
    period_um = 10.0
    phase_depth = np.pi  # π phase depth
    z_um = 1000.0

    nx = 512
    ny = 128  # Smaller in Y since grating is 1D
    dx = period_um / 20
    dy = dx

    # Get analytical solution
    result = phase_grating_orders(wavelength_um, period_um, phase_depth, z_um, nx, ny, dx, dy)

    grating = result["grating"]

    # Propagate with angular spectrum
    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [z_um],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": 0.5,
    }

    field_propagated = as_multi_slice.run(grating, plan)

    # Analyze diffraction orders in Fourier space
    field_fft = torch.fft.fft(field_propagated[ny // 2, :])  # Center row
    power_spectrum = torch.abs(field_fft) ** 2

    # Expected efficiencies from Bessel functions
    # For phase depth φ, nth order efficiency = J_n(φ/2)²
    expected_efficiencies = {}
    for n in range(-3, 4):  # Check orders -3 to +3
        eff = jn(n, phase_depth / 2) ** 2
        if eff > EFF_THRESHOLD:  # Only significant orders
            expected_efficiencies[n] = eff

    # Find peaks in power spectrum
    freq_bins = torch.fft.fftfreq(nx, d=dx)

    for n, expected_eff in expected_efficiencies.items():
        # Expected frequency for nth order
        expected_freq = n / period_um

        # Find closest bin
        idx = torch.argmin(torch.abs(freq_bins - expected_freq))

        # Measure efficiency (power in this order)
        measured_eff = power_spectrum[idx].item() / power_spectrum.sum().item()

        # Check within 3% of theoretical
        error = abs(measured_eff - expected_eff) / expected_eff if expected_eff > 0 else 0

        print(
            f"Order {n:+d}: measured={measured_eff:.3f}, "
            f"expected={expected_eff:.3f}, error={error:.1%}"
        )

        assert (
            error <= ORDER_ERROR_TOL
        ), f"Order {n} efficiency error {error:.1%} exceeds {ORDER_ERROR_TOL:.0%}"

    print(f"✓ Grating orders match theory for φ={phase_depth / np.pi:.2f}π phase depth")


if __name__ == "__main__":
    test_grating_orders()
    print("\nPhase grating diffraction test passed!")
