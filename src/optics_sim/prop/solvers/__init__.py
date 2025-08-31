"""Solver registry and selection shim.

Exposes a minimal dispatcher `run(field, plan, solver=None, sampler=None)`
that routes to the appropriate backend based on a string key. This keeps
config-driven solver selection ready for Claude's implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Optional

import numpy as np

# Local imports are done inside the dispatcher to avoid circulars and allow
# CPU-only environments to import this module without CUDA.


_SOLVER_ALIASES: dict[str, str] = {
    "bpm_vector_wide": "optics_sim.prop.solvers.bpm_vector_wide",
    "bpm_split_step_fourier": "optics_sim.prop.solvers.bpm_split_step_fourier",
    "as_multi_slice": "optics_sim.prop.solvers.as_multi_slice",
}


def _resolve_solver_module(solver: str):  # type: ignore[no-untyped-def]
    import importlib

    key = solver.strip().lower()
    if key not in _SOLVER_ALIASES:
        raise ValueError(f"Unknown solver key: {solver}")
    module_name = _SOLVER_ALIASES[key]
    return importlib.import_module(module_name)


def run(
    field: np.ndarray, plan: Any, solver: str | None = None, sampler: Any | None = None
) -> np.ndarray:  # noqa: ANN401
    """Dispatch to a concrete solver's run.

    Args:
        field: Complex input field array (ny, nx), dtype complex64
        plan: Planning object (opaque here)
        solver: Solver key. If None, defaults to 'bpm_vector_wide'.
        sampler: Optional sampler/policy object

    Returns:
        Output field array
    """
    selected = solver or "bpm_vector_wide"
    module = _resolve_solver_module(selected)
    if not hasattr(module, "run"):
        raise AttributeError(f"Solver module '{module.__name__}' lacks a run() function")
    return module.run(field, plan, sampler)  # type: ignore[misc]
