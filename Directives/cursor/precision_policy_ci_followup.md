Cursor AI (GPT-5) — Precision Policy CI Follow-up (M2)

Objective: land CI changes and produce proof artifacts.

Edits applied in CI

- Added precision-policy-cpu and precision-policy-cuda jobs.
- Determinism (GPU):
  - TORCH_ALLOW_TF32=0, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED=0
  - Python prelude: `torch.use_deterministic_algorithms(True)`; TF32 disabled on matmul/cudnn.
- Static grep guards:
  - Linux: `grep -r -nE "complex32|float16|use_mixed" src/` fails on any hit.
  - Windows: `if (Select-String -Path src\* -Pattern 'complex32|float16|use_mixed' -AllMatches -Recurse) { exit 1 }`
- Runtime guard: `validate_precision_invariants()` executed and logged to precision_policy.log.
- Tests executed: `pytest -q tests/test_precision_policy.py --junitxml=pytest_precision.xml`.
- Validation suite: `python run_validation_suite.py` → `validation_suite.log`.
- Env snapshot artifact: `env_snapshot.txt` includes python/torch/cuda/cudnn/platform.
- Artifacts uploaded in both jobs: JUnit XML, precision_policy.log, validation_suite.log, env_snapshot.txt, docs/precision_policy.md.

Run plan

1) Push CI changes on a branch and open PR.
2) Ensure both jobs pass; share artifact links from the Actions run.

Exit criteria

- CPU and GPU precision-policy jobs pass.
- No grep hits for `complex32|float16|use_mixed`.
- Validation suite within gates.
- Artifacts present and readable: pytest_precision.xml, precision_policy.log, validation_suite.log, env_snapshot.txt, docs/precision_policy.md.



