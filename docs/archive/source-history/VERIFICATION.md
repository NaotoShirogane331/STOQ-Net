# Verification — four development variants, 2026-10-05

Fresh isolated Windows CPU verification of the actual publication staging tree:

```text
python -m pytest -q tests experiments/c_pilot/test_pilot_policy.py -p no:cacheprovider
277 passed, 5 skipped, 25 subtests passed in 5.60s
exit code 0
```

The original seven library test modules and pilot policy tests remain. Added
tests exercise all four development variant model/trainer entry points using
small synthetic tensors, initialization/gradient/physics/missingness contracts,
state-dict roundtrips, raw-loss optimizer recovery and pinned TimeXer encoder
parity. These are synthetic checks, not new real-data training or evaluation.

Five skips: one C2 CUDA/AMP raw-loss witness requires a GPU; four original
production-experts equivalence checks require `RACEPV_ORIGINAL_EXPERTS`, which
was unset. Neither skip is counted as passing or as new GPU qualification.
Windows `tests/exploratory/conftest.py` permits importing Linux-oriented
ResearchTrainer using a test-only `fcntl` stub. Every stub lock call raises;
native Linux recovery locks/process supervision are not tested or modified.

Environment reused from the previous isolated source release: Python 3.12.14,
PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.5.3, pandas 3.0.6, PyYAML 6.0.3,
SciPy 1.18.1 and scikit-learn 1.9.1. `PYTHONPATH` points explicitly to this
publication tree, not the editable previous release or experiment workspace.
These are observed verification versions, not a dependency lockfile.

A no-dependency/no-build-isolation wheel build exited 0, and
independent temporary extraction verified the four variant model imports,
C3 feature signature and retained MIT license. C2/C3 parameter counts were
180874/455050; the synthetic GRU C-DH configuration was 192622. This build
check does not qualify a missing production Mamba/CUDA dependency. The wheel
is a local verification artifact and is not uploaded in this source-only update.

All original 160 Python library files are byte-identical to the previous
2026-10-03 source release. Copied numerical methods are retained; relative
import wiring and portable test bootstraps are documented in the variant
source provenance. Campaign policies include only historical candidate rows
and their exact validator, no cloud execution functions or expired deadlines.

All publication Python files parse. Bounded key/token/private-host/user-path
scans found no matching patterns; this is not an exhaustive security audit.
`SHA256SUMS` records all publication payloads except itself. Data, weights,
prediction arrays, raw cloud execution logs and machine configuration are not
included. This verifies distributed source contracts, not full campaign/paper
reproduction, benchmark rankings, source-data consent or native FLOPs counts.
