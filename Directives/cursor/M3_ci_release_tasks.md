Cursor AI (GPT-5) — M3 CI Release Tasks

Goal

Create a CI job to run example presets on a GPU runner, enforce runtime/VRAM budgets and physics gates, and package docs and wheels for release.

Job: gpu-smoke

- Matrix over configs in `examples/m3/*.yaml`.
- Determinism/precision prelude (at job start):
  - `torch.use_deterministic_algorithms(True)`
  - `TORCH_ALLOW_TF32=0`
  - `CUBLAS_WORKSPACE_CONFIG=":4096:8"`
  - Disable TF32: `torch.backends.cuda.matmul.allow_tf32=False`; `torch.backends.cudnn.allow_tf32=False`
  - Set matmul precision: `torch.set_float32_matmul_precision("highest")`
  - Assert: `torch.get_float32_matmul_precision()=="highest"`
- Assert CUDA is available and print device info (names, count).
- Execute per matrix entry:
  - `python -m optics_sim.cli.run --config ${{ matrix.cfg }} --out artifacts/${{ matrix.name }}`
  - Capture `stdout/stderr` to `artifacts/${{ matrix.name }}/run.log`.
  - Save `nvidia-smi` snapshots before/after to `nvidia_smi_before.txt`, `nvidia_smi_after.txt` alongside outputs.

Budget + gates

- After each run, parse `artifacts/${{ matrix.name }}/metrics.json` and fail if physics thresholds exceeded (current gates as in tests).
- Parse `artifacts/${{ matrix.name }}/perf.json` and fail if any of:
  - `peak_vram >= 4e9` bytes
  - `wall_time >= 90` seconds

Artifacts (30-day retention)

- Upload per-matrix directory contents with 30-day retention:
  - `output.tiff`
  - `metrics.json`
  - `perf.json`
  - `env.json` (environment snapshot)
  - `run.log`
  - `nvidia_smi_before.txt`, `nvidia_smi_after.txt`

Packaging (release smoke)

- Build distributions:
  - `python -m build` (wheel + sdist)
  - `python -c "import optics_sim; print(optics_sim.__version__)"` → save to `import_smoke.log`
- Upload artifacts (30-day retention): `dist/*`, `import_smoke.log`.

Docs

- Build docs site (e.g., MkDocs/Sphinx per repo tooling).
- Upload built site as artifact (30-day retention).
- Add README link: “GPU smoke (M3)” → CI docs artifact URL or docs site path.

Reference CI steps (to implement)

```yaml
jobs:
  gpu-smoke:
    runs-on: [self-hosted, gpu]
    strategy:
      fail-fast: false
      matrix:
        include:
          - name: cfg1
            cfg: examples/m3/cfg1.yaml
          # auto-discovery can be added in a setup step
    env:
      TORCH_ALLOW_TF32: "0"
      CUBLAS_WORKSPACE_CONFIG: ":4096:8"
      PYTHONHASHSEED: "0"
    steps:
      - uses: actions/checkout@v4
      - name: Install CUDA torch and project
        run: |
          python -m pip install --upgrade pip
          python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.3.* torchvision==0.18.* torchaudio==2.3.*
          python -m pip install -e .[cuda,dev]
      - name: Determinism/precision prelude
        run: |
          python - << 'PY'
          import torch
          torch.use_deterministic_algorithms(True)
          torch.backends.cuda.matmul.allow_tf32=False
          torch.backends.cudnn.allow_tf32=False
          torch.set_float32_matmul_precision("highest")
          assert torch.get_float32_matmul_precision()=="highest"
          assert torch.cuda.is_available()
          print("devices", torch.cuda.device_count(),
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
          PY
      - name: Env snapshot
        run: |
          python - << 'PY' > env.json
          import sys, torch, platform, json
          print(json.dumps({
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
          }, indent=2))
          PY
      - name: Run example
        run: |
          mkdir -p artifacts/${{ matrix.name }}
          nvidia-smi > artifacts/${{ matrix.name }}/nvidia_smi_before.txt
          python -m optics_sim.cli.run --config ${{ matrix.cfg }} --out artifacts/${{ matrix.name }} \
            > artifacts/${{ matrix.name }}/run.log 2>&1
          nvidia-smi > artifacts/${{ matrix.name }}/nvidia_smi_after.txt
      - name: Enforce budgets & gates
        run: |
          python - << 'PY'
          import json, sys, pathlib
          p = pathlib.Path('artifacts')/('${{ matrix.name }}')
          metrics = json.loads((p/'metrics.json').read_text())
          perf = json.loads((p/'perf.json').read_text())
          # physics gates: honor existing thresholds encoded in metrics
          assert metrics.get('within_gates', True), metrics
          assert perf.get('peak_vram_bytes', 0) < 4_000_000_000, perf
          assert perf.get('wall_time_s', 0.0) < 90.0, perf
          PY
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: gpu-smoke-${{ matrix.name }}
          path: |
            artifacts/${{ matrix.name }}/output.tiff
            artifacts/${{ matrix.name }}/metrics.json
            artifacts/${{ matrix.name }}/perf.json
            artifacts/${{ matrix.name }}/env.json
            artifacts/${{ matrix.name }}/run.log
            artifacts/${{ matrix.name }}/nvidia_smi_before.txt
            artifacts/${{ matrix.name }}/nvidia_smi_after.txt
          retention-days: 30

  package-and-docs:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Build wheels/sdist
        run: |
          python -m pip install build
          python -m build
          python - << 'PY' > import_smoke.log
          import optics_sim; print(optics_sim.__version__)
          PY
      - uses: actions/upload-artifact@v4
        with:
          name: dist-artifacts
          path: |
            dist/*
            import_smoke.log
          retention-days: 30
      - name: Build docs
        run: |
          # replace with actual docs build (mkdocs/sphinx)
          echo "Docs build placeholder" > site/index.html
      - uses: actions/upload-artifact@v4
        with:
          name: docs-site
          path: site
          retention-days: 30
```

Acceptance

- `gpu-smoke` green for all `examples/m3/*.yaml`; artifacts present for each matrix entry.
- Wheels and sdist build; `import_smoke.log` shows import and version.
- Docs artifact builds without errors; README links “GPU smoke (M3)”.


