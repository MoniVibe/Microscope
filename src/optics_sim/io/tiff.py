from __future__ import annotations

from typing import Dict
import numpy as np
import tifffile as tiff


def write_tiff_stack(path: str, planes: Dict[str, np.ndarray], metadata: Dict[str, str] | None = None) -> None:
    """Write a set of planes to a TIFF stack.

    Minimal stub: writes each plane as a page with a simple description tag.
    """

    with tiff.TiffWriter(path, bigtiff=False) as tw:
        for name, arr in planes.items():
            tw.write(arr, description=name, metadata=metadata or {})


