# Scope and limitations

1. This is a current source-library synchronization, not a complete experimental
   release. Raw data, prepared arrays, predictions, trained weights, cloud
   controllers and private execution bindings are excluded.
2. Synthetic CPU checks validate only their stated code paths. Native
   Mamba/CUDA execution, every optional dependency, all CLI routes and the
   complete original workspace test suite are not validated by these checks.
3. Runtime-v4 source is preserved exactly. Historical configuration examples
   must not be substituted for a specific experiment's resolved contract.
4. The new C pilot uses training/validation development data only. Its variants
   and policy are included without execution results. Short development trials
   cannot replace the previously reported final-test results or establish a
   new blind improvement after heldout outcomes have been inspected.
5. This code release independently establishes neither provider consent,
   field timestamp/arrival guarantees nor historical heldout independence.
   Previously recorded unknowns and non-blind retrospective limitations remain.
6. Full native FLOPs/MACs/coverage remain unavailable where the actual FFT
   dispatch and fused Mamba implementation evidence is insufficient. Synthetic
   CPU tests are not a substitute count or GPU-speed/energy benchmark.
7. External backbones and pretrained weights require their own dependencies,
   licenses and identity checks. They are not redistributed here.
8. Secret/path screening is bounded, not proof of exhaustive detection; the
   existing project and third-party licenses are retained, not re-certified.
