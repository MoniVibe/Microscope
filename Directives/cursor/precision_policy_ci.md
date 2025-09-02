Cursor AI (GPT-5) — Precision Policy CI Directive

Objective

Enforce strict precision policy in CI on CPU and GPU. Fail fast on any mixed-precision reintroduction. Publish precision and physics gate artifacts.

Jobs

- Add two CI jobs:
  - precision-policy-cuda (GPU runner)
  - precision-policy-cpu (CPU runner)

Commands to run (both jobs)

- Run unit policy gate:
  - pytest -q tests/test_precision_policy.py --junitxml=pytest_precision.xml
- Run validation suite:
  - python run_validation_suite.py > validation_suite.log 2>&1
- Upload artifacts: JUnit XML (pytest_precision.xml), precision_policy.log, validation_suite.log

Environment hardening (GPU jobs)

- Set environment variables:
  - TORCH_ALLOW_TF32=0
  - CUBLAS_WORKSPACE_CONFIG=":4096:8"
  - PYTHONHASHSEED=0
- Prepend Python snippet early in the job before tests:
  - import torch
  - torch.backends.cuda.matmul.allow_tf32=False
  - torch.backends.cudnn.allow_tf32=False
- Seed deterministically via existing test harness/fixtures.

Static guard (both jobs)

- Add a step before running tests:
  - On Linux/macOS runners:
    - grep -r -nE "complex32|float16|use_mixed" src/ && exit 1 || true
  - On Windows PowerShell runners:
    - if (Select-String -Path src -Pattern 'complex32|float16|use_mixed' -AllMatches -Recurse) { exit 1 }

Runtime guard (both jobs)

- Add a step to assert runtime invariants and write log:
  - python - << 'PY' > precision_policy.log
    from optics_sim.core.precision import validate_precision_invariants
    validate_precision_invariants()
    PY

Policy flags

- Assert MIXED_FFT=False in CI environment or the config consumed by CI. Never override in GPU jobs. tests/test_precision_policy.py already verifies disabled.

Artifacts to upload

- pytest_precision.xml (JUnit)
- precision_policy.log
- validation_suite.log
- docs/precision_policy.md (publish for surfacing)

Docs surfacing

- Link docs/precision_policy.md from README under “Development > Precision policy”.
- Also upload docs/precision_policy.md as an artifact in both jobs.

Acceptance (M2)

- precision-policy-cpu and precision-policy-cuda jobs green.
- No static grep hits for complex32|float16|use_mixed.
- tests/test_precision_policy.py and run_validation_suite.py pass with current gates.
- Artifacts present for both jobs: JUnit XML, precision_policy.log, validation_suite.log, precision_policy.md.
- Any deviation flips CI red.



