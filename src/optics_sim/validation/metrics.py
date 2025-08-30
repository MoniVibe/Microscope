from __future__ import annotations

import numpy as np


def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 error between two arrays."""

    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + 1e-12
    return float(num / den)


