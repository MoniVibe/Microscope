"""Tests for thin lens focusing (paraxial)."""

import numpy as np
import torch

from optics_sim.prop.solvers import bpm_split_step_fourier
from optics_sim.validation.cases import thin_lens_focus
from optics_sim.validation.metrics import compute_fwhm, strehl_ratio


def test_lens_paraxial():
    """Test paraxial lens focusing with Strehl ratio."""
    wavelength_um = 0.55
    focal_length_um = 10000.0  # 10mm focal length
    lens_diameter_um = 2000.0  # 2mm aperture

    # Low NA for paraxial
    na = lens_diameter_um / (2 * focal_length_um)  # = 0.1
    assert na < 0.2, "Not paraxial regime"

    nx = ny = 256
    dx = dy = lens_diameter_um / 80

    # Get fields at lens and focus
    field_lens, field_focus = thin_lens_focus(
        wavelength_um,
        focal_length_um,
        lens_diameter_um,
        focal_length_um,  # Propagate to focus
        nx,
        ny,
        dx,
        dy,
    )

    # Propagate with BPM
    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [focal_length_um],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": na,
    }

    field_computed = bpm_split_step_fourier.run(field_lens, plan)

    # Compute Strehl ratio
    psf_computed = torch.abs(field_computed) ** 2
    psf_ideal = torch.abs(field_focus) ** 2

    strehl = strehl_ratio(psf_computed, psf_ideal)
    assert strehl >= 0.95, f"Strehl ratio {strehl:.3f} < 0.95 for paraxial lens"

    # Check FWHM
    fwhm_x, fwhm_y = compute_fwhm(psf_computed, dx, dy)

    # Theoretical FWHM for Airy disk: ~1.02 λ f/D
    fwhm_theory = 1.02 * wavelength_um * focal_length_um / lens_diameter_um

    error_x = abs(fwhm_x - fwhm_theory) / fwhm_theory
    error_y = abs(fwhm_y - fwhm_theory) / fwhm_theory

    assert error_x <= 0.02, f"FWHM X error {error_x:.1%} exceeds 2%"
    assert error_y <= 0.02, f"FWHM Y error {error_y:.1%} exceeds 2%"

    print(f"✓ Paraxial lens: Strehl={strehl:.3f}, FWHM={fwhm_x:.2f}µm (theory={fwhm_theory:.2f}µm)")


if __name__ == "__main__":
    test_lens_paraxial()
    print("\nThin lens focusing test passed!")
