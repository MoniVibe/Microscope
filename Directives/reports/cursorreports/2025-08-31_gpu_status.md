GPU Run Status — 2025-08-31

Commit
- latest_commit.txt: 722f3e3

Environment
```text
(from latest_gpu_env.txt)
python=3.13.5
torch=not installed
cuda_available=False
cuda_devices=0
commit=722f3e3
```

GPU PyTest
- No `latest_gpu_pytest.txt` present (pytest not available in this environment), so GPU-marked tests were not executed locally.

Notes
- This machine lacks CUDA-enabled PyTorch; use the self-hosted GPU runner for CI `-m gpu` runs.
- Solver selection via `solver` key is wired in `optics_sim.prop.solvers.run()` for Claude’s merge.


