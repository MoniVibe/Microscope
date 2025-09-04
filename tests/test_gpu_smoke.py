import pytest

pytestmark = pytest.mark.gpu


def test_gpu_smoke_tiny_tensor():
    try:
        import torch  # type: ignore  # noqa: PLC0415 - local import to allow skip when torch missing
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"torch not available: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn((8, 8), device="cuda", dtype=torch.complex64)
    y = x * (1 + 0j)
    assert y.shape == (8, 8)
