import os
import random

import numpy as np
import pytest

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional import for device fixture
    torch = None  # type: ignore


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    config.addinivalue_line("markers", "gpu: marks tests that require a CUDA GPU")


@pytest.fixture(autouse=True)
def seed_rng() -> None:
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"


@pytest.fixture()
def device() -> str:
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
