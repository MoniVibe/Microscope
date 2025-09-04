#!/usr/bin/env python3
"""Run all validation tests and report results."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

# Import energy audit functions
from optics_sim.prop.solvers.bpm_vector_wide import validate_energy_conservation

# Import test functions
from tests.test_aperture_airy import test_aperture_airy
from tests.test_frames import test_round_trip_accuracy
from tests.test_grating_orders import test_grating_orders
from tests.test_io_shapes_meta import test_io_shapes_meta
from tests.test_lens_paraxial import test_lens_paraxial
from tests.test_solvers_gaussian_free_space import (
    test_angular_spectrum_gaussian,
    test_bpm_gaussian_propagation,
    test_split_step_gaussian,
)


def run_validation_suite():
    """Run complete validation suite and report results."""
    print("=" * 60)
    print("OPTICS SIMULATION VALIDATION SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Gaussian Free Space Propagation
    print("\n1. GAUSSIAN FREE SPACE PROPAGATION")
    print("-" * 40)

    try:
        # BPM Vector Wide
        test_bpm_gaussian_propagation()
        print("✓ BPM Vector Wide: L2 ≤3%, Energy ≤1%")
        results["bpm_gaussian"] = "PASS"
    except AssertionError as e:
        print(f"✗ BPM Vector Wide: {e}")
        results["bpm_gaussian"] = "FAIL"

    try:
        # Split-step Fourier
        test_split_step_gaussian()
        print("✓ Split-step Fourier: L2 ≤3%, Energy ≤1%")
        results["split_step_gaussian"] = "PASS"
    except AssertionError as e:
        print(f"✗ Split-step Fourier: {e}")
        results["split_step_gaussian"] = "FAIL"

    try:
        # Angular Spectrum
        test_angular_spectrum_gaussian()
        print("✓ Angular Spectrum: L2 ≤3%, Energy ≤1%")
        results["as_gaussian"] = "PASS"
    except AssertionError as e:
        print(f"✗ Angular Spectrum: {e}")
        results["as_gaussian"] = "FAIL"

    # Test 2: Airy Pattern (Aperture Diffraction)
    print("\n2. AIRY PATTERN (APERTURE DIFFRACTION)")
    print("-" * 40)

    try:
        test_aperture_airy()
        print("✓ Airy pattern: Peak centered, first zero ≤2% error")
        results["airy"] = "PASS"
    except AssertionError as e:
        print(f"✗ Airy pattern: {e}")
        results["airy"] = "FAIL"

    # Test 3: Thin Lens (Paraxial)
    print("\n3. THIN LENS FOCUSING (PARAXIAL)")
    print("-" * 40)

    try:
        test_lens_paraxial()
        print("✓ Thin lens: Strehl ≥0.95, MTF cutoff ≤2% error")
        results["lens"] = "PASS"
    except AssertionError as e:
        print(f"✗ Thin lens: {e}")
        results["lens"] = "FAIL"

    # Test 4: Phase Grating Orders
    print("\n4. PHASE GRATING DIFFRACTION ORDERS")
    print("-" * 40)

    try:
        test_grating_orders()
        print("✓ Grating orders: Power ratios ≤3% error")
        results["grating"] = "PASS"
    except AssertionError as e:
        print(f"✗ Grating orders: {e}")
        results["grating"] = "FAIL"

    # Test 5: Frame Round-trip
    print("\n5. COORDINATE FRAME TRANSFORMS")
    print("-" * 40)

    try:
        test_round_trip_accuracy()
        print("✓ Frame round-trip: Error < 1e-6 µm")
        results["frames"] = "PASS"
    except AssertionError as e:
        print(f"✗ Frame round-trip: {e}")
        results["frames"] = "FAIL"

    # Test 6: TIFF I/O
    print("\n6. TIFF I/O WITH METADATA")
    print("-" * 40)

    try:
        test_io_shapes_meta()
        print("✓ TIFF I/O: Metadata complete, shapes preserved")
        results["tiff"] = "PASS"
    except AssertionError as e:
        print(f"✗ TIFF I/O: {e}")
        results["tiff"] = "FAIL"

    # Energy Conservation Audit
    print("\n7. ENERGY CONSERVATION AUDIT")
    print("-" * 40)

    # Test energy conservation for a simple propagation
    wavelength_um = 0.55
    nx = ny = 128
    dx = dy = 0.5

    # Create test field (Gaussian)
    x = torch.linspace(-(nx - 1) / 2, (nx - 1) / 2, nx) * dx
    y = torch.linspace(-(ny - 1) / 2, (ny - 1) / 2, ny) * dy
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    waist = 10.0
    field_in = torch.exp(-(xx**2 + yy**2) / waist**2).to(torch.complex64)

    # Simple propagation test
    from optics_sim.prop.solvers import (
        bpm_vector_wide,  # noqa: PLC0415 - imported here to avoid heavy import at startup
    )

    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [50.0],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": 0.25,
    }

    field_out = bpm_vector_wide.run(field_in, plan)

    energy_metrics = validate_energy_conservation(field_in, field_out, dx, dy)

    ENERGY_TOL = 0.01  # PLR2004 named threshold
    if energy_metrics <= ENERGY_TOL:
        print(f"✓ Energy conservation: {energy_metrics:.2%} change (≤1%)")
        results["energy"] = "PASS"
    else:
        print(f"✗ Energy conservation: {energy_metrics:.2%} change (>1%)")
        results["energy"] = "FAIL"

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)

    for test_name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name:20s}: {status}")

    print("-" * 60)
    print(f"OVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL VALIDATION GATES PASSED!")
        print("\nAcceptance Criteria Met:")
        print("  • Gaussian: L2 ≤3%, energy ≤1% ✓")
        print("  • Airy: peak + first zero within 2% ✓")
        print("  • Grating orders: power ratios within 3% ✓")
        print("  • Thin lens: Strehl ≥0.95, MTF cutoff within 2% ✓")
        print("  • Frame round-trip: error < 1e-6 µm ✓")
        print("  • TIFF metadata: complete ✓")
        print("  • Energy conservation: ≤1% ✓")
        return 0
    else:
        print(f"\n⚠ {total - passed} tests failed. Review needed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation_suite())
