# STOQ-Net companion update — 2026-10-13

## Scope and identity

This update adds a STOQ-Net-oriented README, complete model-only profile,
synthetic example, usage guide and example tests to the source library based
on commit `b5b658b8a3640c311bffe918bdedc411acdd4b81`.
STOQ-Net maps to `SingleHeadRaCEPV`, internal `M1`, revision
`racepv.research-single-head.v1`.

All 185 pre-existing files under `src` (the 160-file original runtime library
and retained later explorations) remain byte-identical to that base. The
license, packaging metadata, four older YAML configurations, historical
snapshot and verification records are preserved. Repository visibility stays
private. No manuscript is changed by this update.

The complete 51-field JSON model profile was compared with the saved resolved
configurations for historical M1 seeds 13/29/47. All three were identical. The
source identity record SHA-256 is
`1cbcf32250f31639f1c0671f25df736290e4f038186248d34d9003f3f343d240`.
Only constructor settings are extracted; the private record and its machine
paths are not distributed. The five tuple-valued fields are JSON arrays and
are restored by the example loader. Inherited configuration fields remain
unchanged; actual single-head metadata comes from the existing model.

Input ordering is cross-checked against `src/racepv/io.py`. Descriptions use
the existing preprocessing, causal-history, solar-feature, dataset and
prepare source definitions; the model's own missingness and output handling
remain in `src/racepv/models`.

## Actual verification

The existing isolated WSL environment is reused: Python 3.12.3, PyTorch
2.13.0+cu130 and pytest 9.1.1. No dependencies are installed. CUDA is hidden,
and synthetic model checks run on CPU with explicit GRU. The package source
path points to this update's source staging tree, not the experiment workspace.

Baseline before additions: **276 passed, 6 skipped, 25 subtests passed**.
The five new example tests were first observed failing on the missing example,
then passed after implementation. They check the restored configuration,
`[2,24,7]` finite/ordered/bounded output and night zeroing, the original
clear-sky fallback, explicit backend selection and missing-Mamba errors.

Full test command:

```bash
python -m pytest -q tests experiments/c_pilot/test_pilot_policy.py -p no:cacheprovider
```

Final result: **281 passed, 6 skipped, 25 subtests passed**; exit code 0.
The six skips are the CUDA-only raw-loss witness, Windows-only import shim,
and four original-experts-equivalence checks whose separately supplied
reference is absent. Skips are not counted as passes. The earlier Windows
verification record is not overwritten with this Linux result.

The CPU minimal command reports `SingleHeadRaCEPV`, revision
`racepv.research-single-head.v1`, actual backend `causal_conv_gru` and output
shape `[2,24,7]`. Native Mamba/CUDA execution is **not run or qualified** in
this update. Missing-dependency behavior is checked separately on the real
factory with that optional dependency marked unavailable.

These are interface and compatibility checks with random synthetic inputs,
not new scientific experiments, trained-checkpoint validation, calibration,
scoring or paper-result reproduction. No data, weights, prediction arrays,
private logs, credentials or machine configuration are added.

## Provenance records

[`STOQ_COMPANION.json`](STOQ_COMPANION.json) records this update's scope and
configuration origin. [`SHA256SUMS`](SHA256SUMS) is refreshed for current
payloads and excludes itself. The existing `SNAPSHOT.json`,
`RELEASE_STATUS.json`, `VERIFICATION.md` and upstream notices describe earlier
source releases and retain their original bytes.
