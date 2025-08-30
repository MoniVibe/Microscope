from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal
import numpy as np


WhatToRecord = Literal["intensity", "complex", "phase"]


@dataclass(slots=True)
class PlaneRecorder:
    z_um: float
    what: List[WhatToRecord] = field(default_factory=lambda: ["intensity"]) 

    def capture(self, field_xy: np.ndarray) -> dict:
        out: dict = {}
        if "intensity" in self.what:
            out["intensity"] = (field_xy.conj() * field_xy).real.astype(np.float32)
        if "complex" in self.what:
            out["real"] = field_xy.real.astype(np.float32)
            out["imag"] = field_xy.imag.astype(np.float32)
        if "phase" in self.what:
            out["phase"] = np.angle(field_xy).astype(np.float32)
        return out


