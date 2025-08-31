Microscope Cursor Report — 2025-08-31

Scope
- Prepare GPU smoke configs calling each solver on a 256² grid with 1 spectral × 1 angle sample.
- Wire CI job to run `-m gpu` and minimal functional path on GPU runner.
- Keep README install instructions aligned with cpu/cuda extras; reference new examples.

Changes
- Added GPU smoke example configs:
  - examples/gpu_smoke_bpm_vector.yml
  - examples/gpu_smoke_ssf.yml
  - examples/gpu_smoke_as_multi_slice.yml
  Each sets na=0.25, 256×256 grid, 0.5 µm pitch, 1×1 samples, gaussian source, device=cuda.

- Updated GitHub Actions workflow `.github/workflows/ci.yml`:
  - GPU job now pins Python 3.11, enforces a CUDA availability gate, checks commit ancestry (>= 722f3e3), runs the scoped GPU tests: `-m gpu -k "frames or sources or solvers_gaussian_free_space"`, captures `gpu_env.txt` and `gpu_pytest.txt`, and uploads them as artifacts.

- Updated `README.md`:
  - Kept CPU and CUDA extras instructions consistent with `pyproject.toml`.
  - Listed the three new GPU smoke example configs and clarified test commands.

Notes / Next steps for Claude (Opus)
- Implement solver selection honoring the `solver` key in configs; current tests import solver directly.
- Ensure planners/samplers respect `spectral_samples` and `angular_samples` when present.
- Once solvers land, add a CLI example showing how to run the provided GPU smoke configs.

CI Impact
- CPU jobs unchanged.
- GPU job remains self-hosted; now also runs a slim functional subset beyond the gpu-marked smoke to validate frames/sources integration on CUDA.


