# Known limitations of the legacy extract

1. **Historical, non-final artifact.** The source was frozen before candidate A.
   Its auxiliary specialist loss remains present. Removing that term or setting
   its weight to zero must not be described as an independently proven gain.
2. **Data provenance is not established by code tests.** Timestamp anchoring,
   measurement intervals, time zones, observation masks, interpolation causality,
   split boundaries, and permitted source use must be validated for each actual
   export. This extract does not certify historical datasets or results.
3. **No experimental claims.** No result table, predictions, statistical
   significance, best-model claim, or paper-ready comparison is supplied.
   Previously inspected heldout outcomes cannot become a new blind evaluation.
4. **Partial distribution.** Cloud/recovery operations, historical orchestration
   scripts, many test fixtures, source-audit capsules and third-party checkouts
   are not included. Original packaging/CLI entry points are preserved for source
   fidelity, but are not all supported end-to-end by this library-only extract.
   Do not run the historical data-preparation/training commands as a shortcut to
   bypass a corrected protocol.
5. **Tests are scoped.** Only the listed synthetic model/loss tests are verified
   for this draft. No full-suite, clean-environment installation, official Mamba
   execution, real-data inference, or performance reproduction is claimed.
6. **License and secrets review.** Existing notices are retained; ownership and
   third-party notice completeness remain release checks. Pattern scanning is a
   useful screen, not proof that every possible secret has been detected. No
   original Git history is included or claimed to have been sanitized.

Negative or invalidated results must not be reclassified as confirmed positive
evidence. Future final artifacts should identify their own data/code/configuration
and evaluation protocol rather than inheriting this snapshot's status.
