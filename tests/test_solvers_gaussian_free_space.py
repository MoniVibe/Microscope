import numpy as np

from optics_sim.prop.solvers.bpm_vector_wide import run as run_bpm


def test_solver_identity_stub():
    field = np.zeros((16, 16), dtype=np.complex64)
    out = run_bpm(field, plan=None)
    assert np.allclose(out, field)


