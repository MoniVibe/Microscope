"""Tests for Gaussian free space propagation solvers."""

import numpy as np
import torch

from optics_sim.prop.plan import Plan
from optics_sim.prop.solvers import as_multi_slice, bpm_split_step_fourier, bpm_vector_wide
from optics_sim.validation.cases import gaussian_free_space
from optics_sim.validation.metrics import energy_conservation, l2_field_error


def test_solver_identity_stub():
    """Basic test that solvers can be imported and run."""
    field = np.zeros((16, 16), dtype=np.complex64)
    out = bpm_vector_wide.run(field, plan=None)
    assert np.allclose(out, field)


def test_bpm_gaussian_propagation():
    """Test BPM propagation of Gaussian beam."""
    # Parameters
    wavelength_um = 0.55
    waist_um = 10.0
    z_um = 100.0
    nx = ny = 128
    dx = dy = 0.5

    # Get analytical solution
    E0, Ez_analytical = gaussian_free_space(wavelength_um, waist_um, z_um, nx, ny, dx, dy)

    # Create plan
    plan = Plan(
        dx_um=dx,
        dy_um=dy,
        dz_list_um=[z_um],
        nx=nx,
        ny=ny,
        wavelengths_um=np.array([wavelength_um]),
        na_max=0.25,
    )

    # Propagate with BPM
    Ez_computed = bpm_vector_wide.run(E0, plan)

    # Check L2 error
    l2_err = l2_field_error(Ez_computed, Ez_analytical)
    assert l2_err <= 0.03, f"L2 error {l2_err:.3%} exceeds 3%"

    # Check energy conservation
    energy_err = energy_conservation(E0, Ez_computed, dx, dy)
    assert energy_err <= 0.01, f"Energy error {energy_err:.3%} exceeds 1%"


def test_split_step_gaussian():
    """Test split-step Fourier BPM."""
    wavelength_um = 0.55
    waist_um = 10.0
    z_um = 100.0
    nx = ny = 128
    dx = dy = 0.5

    E0, Ez_analytical = gaussian_free_space(wavelength_um, waist_um, z_um, nx, ny, dx, dy)

    plan = Plan(
        dx_um=dx,
        dy_um=dy,
        dz_list_um=[z_um],
        nx=nx,
        ny=ny,
        wavelengths_um=np.array([wavelength_um]),
        na_max=0.25,
    )

    Ez_computed = bpm_split_step_fourier.run(E0, plan)

    l2_err = l2_field_error(Ez_computed, Ez_analytical)
    assert l2_err <= 0.03, f"Split-step L2 error {l2_err:.3%} exceeds 3%"

    energy_err = energy_conservation(E0, Ez_computed, dx, dy)
    assert energy_err <= 0.01, f"Split-step energy error {energy_err:.3%} exceeds 1%"


def test_angular_spectrum_gaussian():
    """Test angular spectrum propagation."""
    wavelength_um = 0.55
    waist_um = 10.0
    z_um = 100.0
    nx = ny = 128
    dx = dy = 0.5

    E0, Ez_analytical = gaussian_free_space(wavelength_um, waist_um, z_um, nx, ny, dx, dy)

    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [z_um],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": 0.25,
    }

    Ez_computed = as_multi_slice.run(E0, plan)

    l2_err = l2_field_error(Ez_computed, Ez_analytical)
    assert l2_err <= 0.03, f"Angular spectrum L2 error {l2_err:.3%} exceeds 3%"

    energy_err = energy_conservation(E0, Ez_computed, dx, dy)
    assert energy_err <= 0.01, f"Angular spectrum energy error {energy_err:.3%} exceeds 1%"


def test_multi_step_propagation():
    """Test propagation with multiple z-steps."""
    wavelength_um = 0.55
    waist_um = 10.0
    total_z = 100.0
    n_steps = 10

    nx = ny = 64
    dx = dy = 1.0

    # Analytical result at final z
    E0, Ez_analytical = gaussian_free_space(wavelength_um, waist_um, total_z, nx, ny, dx, dy)

    # Multi-step plan
    dz_list = [total_z / n_steps] * n_steps

    plan = Plan(
        dx_um=dx,
        dy_um=dy,
        dz_list_um=dz_list,
        nx=nx,
        ny=ny,
        wavelengths_um=np.array([wavelength_um]),
        na_max=0.25,
    )

    # Test each solver
    for solver, name in [
        (bpm_vector_wide, "BPM vector"),
        (bpm_split_step_fourier, "Split-step"),
        (as_multi_slice, "Angular spectrum"),
    ]:
        Ez_computed = solver.run(E0, plan)

        l2_err = l2_field_error(Ez_computed, Ez_analytical)
        energy_err = energy_conservation(E0, Ez_computed, dx, dy)

        print(f"{name}: L2={l2_err:.3%}, Energy={energy_err:.3%}")
        assert l2_err <= 0.05, f"{name} L2 error {l2_err:.3%} exceeds 5%"
        assert energy_err <= 0.02, f"{name} energy error {energy_err:.3%} exceeds 2%"


def test_high_na_propagation():
    """Test high-NA propagation accuracy."""
    wavelength_um = 0.55
    na = 0.8  # High NA
    waist_um = 2.0  # Small beam
    z_um = 10.0  # Short distance

    nx = ny = 128
    dx = dy = wavelength_um / (3 * na)  # Fine sampling for high NA

    E0, Ez_analytical = gaussian_free_space(wavelength_um, waist_um, z_um, nx, ny, dx, dy)

    plan = Plan(
        dx_um=dx,
        dy_um=dy,
        dz_list_um=[z_um],
        nx=nx,
        ny=ny,
        wavelengths_um=np.array([wavelength_um]),
        na_max=na,
        preset="High-NA",
    )

    # Test BPM with vector corrections
    Ez_computed = bpm_vector_wide.run(E0, plan)

    l2_err = l2_field_error(Ez_computed, Ez_analytical)
    assert l2_err <= 0.05, f"High-NA L2 error {l2_err:.3%} exceeds 5%"


def test_spectral_propagation():
    """Test propagation with multiple spectral components."""
    wavelengths = np.array([0.5, 0.55, 0.6])  # RGB
    waist_um = 10.0
    z_um = 50.0

    nx = ny = 64
    dx = dy = 1.0

    # Create broadband field (same spatial, different spectral)
    fields = []
    refs = []

    for wl in wavelengths:
        E0, Ez = gaussian_free_space(wl, waist_um, z_um, nx, ny, dx, dy)
        fields.append(E0)
        refs.append(Ez)

    E0_multi = torch.stack(fields)
    Ez_ref = torch.stack(refs)

    plan = Plan(
        dx_um=dx,
        dy_um=dy,
        dz_list_um=[z_um],
        nx=nx,
        ny=ny,
        wavelengths_um=wavelengths,
        na_max=0.25,
        spectral_samples=len(wavelengths),
    )

    # Propagate all spectral components
    Ez_computed = bpm_split_step_fourier.run(E0_multi, plan)

    # Check each spectral component
    for i, wl in enumerate(wavelengths):
        l2_err = l2_field_error(Ez_computed[i], Ez_ref[i])
        assert l2_err <= 0.03, f"Spectral {wl} µm: L2 error {l2_err:.3%} exceeds 3%"


if __name__ == "__main__":
    # Run basic tests
    test_solver_identity_stub()
    print("✓ Identity test passed")

    test_bpm_gaussian_propagation()
    print("✓ BPM Gaussian propagation passed")

    test_split_step_gaussian()
    print("✓ Split-step Gaussian passed")

    test_angular_spectrum_gaussian()
    print("✓ Angular spectrum Gaussian passed")

    test_multi_step_propagation()
    print("✓ Multi-step propagation passed")

    test_high_na_propagation()
    print("✓ High-NA propagation passed")

    test_spectral_propagation()
    print("✓ Spectral propagation passed")

    print("\nAll Gaussian free space tests passed!")
