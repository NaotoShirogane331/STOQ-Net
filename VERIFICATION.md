# Verification — source-only update 2026-10-03

Fresh isolated local Windows CPU verification of the files in this update:

```text
python -m pytest -q tests experiments/c_pilot/test_pilot_policy.py -p no:cacheprovider
226 passed, 4 skipped in 3.81s
exit code 0
```

Seven library test modules and the pilot policy module were included. The four
skips are the parameterized production-experts source-equivalence checks in
`test_research_variants.py`; `RACEPV_ORIGINAL_EXPERTS` was deliberately unset.
No production reference file, real dataset, pretrained checkpoint or GPU was
used. Small temporary synthetic checkpoints are created by the identity tests.

Environment: Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1, NumPy 2.5.3,
pandas 3.0.6, PyYAML 6.0.3, SciPy 1.18.1, scikit-learn 1.9.1.
The package was installed editable in a new isolated environment; the existing
cloud experiment environment was not changed. These versions describe this
verification, not a lockfile or a claim that every declared dependency range
has been tested.

All 160 library source files matched both the runtime-v4 archive and recovered
source reference byte-for-byte. All included Python files parsed successfully.
Bounded source-only scans found no matching private-key/token, private SSH-host
or user-specific credential-path patterns. This is not an exhaustive secret
audit. Published file identities are recorded in `SHA256SUMS`.

This validates the distributed synthetic suite, not the complete historical
workspace, native Mamba/CUDA kernels, full cloud orchestration, real-data model
performance or paper reproduction. No new training or evaluation is claimed.
Linux parent-death supervision is not independently exercised by this Windows
run. Source was not reformatted or changed to improve test/benchmark outcomes.
