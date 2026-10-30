# Paper companion scope

This companion pairs the existing STOQ-Net implementation with the evaluation
records supplied with the current manuscript. Use the
[result guide](../paper_results/README.md) for table and figure navigation.

## Source implementation and experiment identity

The implementation is based on source snapshot
[`dfff896185d7a1b613b9c3b5e6b3af6876c17694`](https://github.com/NaotoShirogane331/STOQ-Net/tree/dfff896185d7a1b613b9c3b5e6b3af6876c17694).
This companion preserves its model, preprocessing, training, evaluation,
configuration and test files. It adds recorded results and reader guidance.
That source snapshot is an implementation reference; it does not establish
which source commit produced each evaluation record.

The current per-seed result records contain seeds 13, 29, 47, 61, 79, 101,
127, 151, 181 and 211. The main comparison, single-array structural
comparison and twelve-array six-block TCN comparison retain separate
complete-model run families. Join records by experiment, configuration
and seed, rather than by configuration name alone.

The preserved [model-only profile](../configs/stoq-paper-model.json) was
previously checked against historical seeds 13, 29 and 47. That provenance
does not verify the execution settings of all ten seeds in the current
result archive. The [model guide](stoq-net.md) distinguishes constructor
fields, historical training facts and the available demonstration commands.
The generic [base configuration](../configs/base.yaml) must not be treated
as the resolved configuration of the current paper experiments.

## What can be checked

- Aggregate the supplied per-seed and stratified evaluation records with
  equal seed weights and sample standard deviations.
- Inspect interval treatments, lead-specific calibration corrections,
  confusion counts and the records used for plotted summaries.
- Recompute support-cell counts, distinct target-local array-days and
  daily counts from the exported NPZ indices using the supplied script.
- Inspect the model implementation and run its untrained interface example
  with the backend stated by the example.

These checks do not independently recompute the recorded performance
metrics from targets and predictions. The archive contains no raw power
series, per-cell targets or forecasts, trained checkpoints, private
execution bindings or complete run-to-source/configuration records.
The available training and evaluation modules need their own actual inputs
and execution configuration for a complete rerun.

## Version and citation

Use a fixed repository tag or release when referring to this companion,
rather than assuming that a changing default branch is the submitted
version. A release identifies the code and data tree being shared. Any
later repository archive should point to that same version.

No DOI or public-access claim is assigned by these files. A DOI, if an
archive is subsequently published, should be added only after its record
exists. The repository's software license and third-party notices remain
unchanged; reuse terms for original provider data must be checked with
that provider.
