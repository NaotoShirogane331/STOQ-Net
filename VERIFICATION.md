# Verification of this source-only draft

Verified on 2026-09-16 using Python 3.12.3, PyTorch 2.13.0+cu130 and pytest 9.1.1.
CUDA visibility was disabled. Imports were checked to resolve to this extract,
not the installed main project. No model training, original data access, cloud
connection, or experimental inference was performed.

- `tests/test_model.py`: 57 passed.
- `tests/test_ramp_refinement_losses.py`: 9 passed.
- `tests/test_sparse_tail_losses.py`: 5 passed.
- Total: **71 passed in 8.70s**. These are synthetic functional checks, not
  prediction-performance evidence or a full-project test-suite claim.
- Ruff static check: original full snapshot passes; extracted source passes
  with original first-party module classification (`racepv`, `scripts`). The
  added `ruff.toml` records only that packaging context and suppresses no rules.
- **Ruff format check does not pass:** 57 files would be reformatted; 76 are
  already formatted. Historical code bytes are deliberately preserved. Do not
  describe this draft as passing every formatting gate.

An initial check was stopped before test collection because the local controller
omitted required Git-isolation environment pins. The controller was corrected,
not the guard or scientific code. The subsequent default lint check initially
reported six import-sorting issues because the omitted `scripts` directory was
no longer recognized as first-party. The original-context comparison established
the cause. Failure logs are retained privately outside this directory.

Reproduce the supported checks after installing dependencies in an isolated
environment:

```bash
python -m pytest -q tests/test_model.py tests/test_ramp_refinement_losses.py tests/test_sparse_tail_losses.py
python -m ruff check src tests
# Expected to report the preserved historical formatting differences:
python -m ruff format --check src tests
```

No clean-environment installation, GPU/Mamba execution, scientific validity of
historical results, or final permission to publish is established by these checks.
