# Objective 000: Bootstrap

- Owner: Opus
- Inputs: `pyproject.toml`, `pytest.ini`, `mypy.ini`, `src/optics_sim/**`, `tests/**`
- Outputs: scripts (`bootstrap`, `verify`), CI workflow, docs scaffold
- Test: `make verify` passes on CPU-only runner
- Gate: format → lint → type → unit → integration (stop on fail)

Checklist:
- [x] Add repo meta files
- [ ] Add docs scaffold
- [ ] Add scripts and Makefile
- [ ] Configure CI
- [ ] Baselines generated, zero-new policy enforced
