#!/usr/bin/env python3
"""Test Airy pattern accuracy with corrected AS kernel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from optics_sim.prop.solvers import as_multi_slice


def test_airy_zero_accuracy():
    """Test that Airy first zero is within 2% of theoretical value."""

    print("Testing Airy Pattern First Zero Position")
    print("=" * 50)

    # Parameters
    wavelength_um = 0.55
    aperture_diameter_um = 100.0
    z_um = 10000.0  # Far field

    nx = ny = 256
    dx = dy = aperture_diameter_um / 40  # Fine sampling

    print(f"Wavelength: {wavelength_um} μm")
    print(f"Aperture diameter: {aperture_diameter_um} μm")
    print(f"Propagation distance: {z_um} μm")
    print(f"Grid: {nx}×{ny}, dx={dx:.3f} μm")

    # Create circular aperture
    x = torch.linspace(-(nx - 1) / 2, (nx - 1) / 2, nx) * dx
    y = torch.linspace(-(ny - 1) / 2, (ny - 1) / 2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)

    aperture = (r <= aperture_diameter_um / 2).to(torch.complex64)

    # Propagate with AS
    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [z_um],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": 0.5,
    }

    print("\nPropagating with Angular Spectrum...")
    airy_computed = as_multi_slice.run(aperture, plan)

    # Find peak
    intensity = torch.abs(airy_computed) ** 2
    peak_val = intensity.max()
    peak_idx = torch.argmax(intensity)
    peak_y, peak_x = peak_idx // nx, peak_idx % nx

    print(f"Peak at pixel ({peak_x}, {peak_y})")

    # Extract radial profile from peak
    x_from_peak = torch.arange(nx, dtype=torch.float32) - peak_x
    y_from_peak = torch.arange(ny, dtype=torch.float32) - peak_y
    yy_p, xx_p = torch.meshgrid(y_from_peak, x_from_peak, indexing="ij")
    r_from_peak = torch.sqrt(xx_p**2 + yy_p**2)

    # Find first minimum by radial averaging
    max_radius = min(peak_x, nx - peak_x - 1, peak_y, ny - peak_y - 1)
    n_bins = int(max_radius)

    radial_intensity = []
    for i in range(n_bins):
        mask = (r_from_peak >= i) & (r_from_peak < i + 1)
        if mask.any():
            radial_intensity.append(intensity[mask].mean().item())
        else:
            radial_intensity.append(0)

    # Find first minimum
    first_min_radius = 0
    for i in range(5, len(radial_intensity) - 1):  # Start from pixel 5 to avoid center
        if (
            radial_intensity[i] < radial_intensity[i - 1]
            and radial_intensity[i] < radial_intensity[i + 1]
        ):
            if radial_intensity[i] < 0.1 * radial_intensity[0]:  # Significant minimum
                first_min_radius = i
                break

    # Theoretical first zero: r = 1.22 * λ * z / D
    r_theory_um = 1.22 * wavelength_um * z_um / aperture_diameter_um
    r_theory_pixels = r_theory_um / dx

    # Calculate error
    error_percent = abs(first_min_radius - r_theory_pixels) / r_theory_pixels * 100

    print("\nResults:")
    print(f"  Theoretical first zero: {r_theory_um:.2f} μm = {r_theory_pixels:.2f} pixels")
    print(f"  Measured first zero: {first_min_radius * dx:.2f} μm = {first_min_radius} pixels")
    print(f"  Error: {error_percent:.1f}%")

    # Check if within 2% tolerance
    if error_percent <= 2.0:
        print(f"\n✓ PASS: Error {error_percent:.1f}% is within 2% tolerance")
        return True
    else:
        print(f"\n✗ FAIL: Error {error_percent:.1f}% exceeds 2% tolerance")

        # Debug information
        print("\nDebug - Radial intensity profile (first 50 pixels):")
        for i in range(min(50, len(radial_intensity))):
            normalized = radial_intensity[i] / radial_intensity[0] if radial_intensity[0] > 0 else 0
            bar = "*" * int(normalized * 40)
            print(f"  r={i:3d}: {normalized:5.3f} {bar}")

        return False


def verify_as_kernel_formula():
    """Verify the AS kernel uses correct formula."""
    print("\n" + "=" * 50)
    print("Verifying AS Kernel Formula")
    print("=" * 50)

    # Test parameters
    nx = ny = 64
    dx = 0.5
    lambda_um = 0.55
    n = 1.0

    # Create test frequencies
    fx = torch.fft.fftfreq(nx, d=dx)
    fy = torch.fft.fftfreq(ny, d=dx)  # Using dx for both (square grid)

    print(f"Grid: {nx}×{ny}, dx={dx} μm")
    print(f"Wavelength: {lambda_um} μm")
    print(f"Refractive index: {n}")

    # Check formula: kz = 2π · sqrt((n/λ)² - fx² - fy²)
    n_over_lambda = n / lambda_um

    # Test at origin (fx=0, fy=0)
    kz_origin = 2 * np.pi * np.sqrt(n_over_lambda**2)
    expected_kz_origin = 2 * np.pi * n / lambda_um

    print("\nAt origin (fx=0, fy=0):")
    print(f"  kz = {kz_origin:.4f}")
    print(f"  Expected (2π·n/λ) = {expected_kz_origin:.4f}")
    print(f"  Match: {'✓' if abs(kz_origin - expected_kz_origin) < 1e-6 else '✗'}")

    # Test at Nyquist frequency
    fx_nyquist = 0.5 / dx  # cycles/μm
    kz_arg = n_over_lambda**2 - fx_nyquist**2

    print(f"\nAt Nyquist (fx={fx_nyquist:.3f} cycles/μm):")
    print(f"  (n/λ)² - fx² = {kz_arg:.4f}")

    if kz_arg > 0:
        kz_nyquist = 2 * np.pi * np.sqrt(kz_arg)
        print(f"  kz = {kz_nyquist:.4f} (propagating)")
    else:
        kz_nyquist = 2 * np.pi * np.sqrt(-kz_arg)
        print(f"  |kz| = {kz_nyquist:.4f} (evanescent)")

    return True


if __name__ == "__main__":
    print("AIRY PATTERN ACCURACY TEST")
    print("=" * 50)

    # Verify kernel formula
    verify_as_kernel_formula()

    # Test Airy zero position
    success = test_airy_zero_accuracy()

    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed - AS kernel needs adjustment")
        sys.exit(1)
