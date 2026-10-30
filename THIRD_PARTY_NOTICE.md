# Third-party notices and source boundaries

Historical attribution review: 2026-09-16. Retained for the private source-only
update on 2026-10-03; this update is not a new exhaustive licensing audit.

The runtime-v4 library additionally includes `research_official.py` (external
NeuralForecast backbones), `research_foundation.py` (existing external foundation
adapters), and `research_tcn.py` (local decoder). These files do not bundle the
external packages, weights or upstream source trees. The C-pilot adapter calls
the existing local library. The historical rights confirmation below applies
to the historical extract and must not be read as a new independent certification.

## Attribution and retained license texts

The project MIT notice does not replace third-party notices. The entries below
identify the inspected fixed sources and the local files that refer to or adapt
them. Original source/configuration bytes are preserved; the additional
attribution is supplied here and in `licenses/`. No upstream source checkout,
pretrained weight or dataset is included.

### FACT

- Official source: [FACT](https://github.com/wanghq21/FACT/tree/aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4)
- Fixed commit: `aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4`
- Upstream license: MIT; [full retained text](licenses/FACT-LICENSE.txt).
- License SHA-256: `8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2`.
- src/racepv/baselines/fact_qr.py: backbone_features follows the upstream FACT forecast branch; time/frequency features are retained, the point projection is replaced by a seven-quantile head, and protocol-specific input/output handling is added. The MIT notice is retained for adapted portions.

### iTransformer

- Official source: [iTransformer](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124)
- Fixed commit: `c2426e68ca13f74aaec08045c5c724d8ad328124`
- Upstream license: MIT; [full retained text](licenses/iTransformer-LICENSE.txt).
- License SHA-256: `29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849`.
- src/racepv/baselines/itransformer_qr.py and extension_registry.py: fixed upstream class loading and an encoder/quantile-head adaptation. The upstream inverted embedding/encoder are obtained externally, not vendored here. Retain MIT attribution for adapted call sequences.

### SOFTS

- Official source: [SOFTS](https://github.com/Secilia-Cxy/SOFTS/tree/f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44)
- Fixed commit: `f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44`
- Upstream license: MIT; [full retained text](licenses/SOFTS-LICENSE.txt).
- License SHA-256: `02ad41e7363e960b1fa91658d0c21bdebf9ccb9881323606412dd39e79a3a178`.
- src/racepv/baselines/softs_qr.py and extension_registry.py: fixed upstream class loading and a STAR-encoder/quantile-head adaptation. The external backbone is not bundled. This historic file remains for snapshot fidelity, not as a claim that SOFTS is in the current comparison.

### OpenLTM

- Official source: [OpenLTM](https://github.com/thuml/OpenLTM/tree/0b3005099d380ecc00d512f72553c7fe8fccca99)
- Fixed commit: `0b3005099d380ecc00d512f72553c7fe8fccca99`
- Upstream license: MIT; [full retained text](licenses/OpenLTM-LICENSE.txt).
- License SHA-256: `29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849`.
- src/racepv/baselines/modern.py: Timer-XL adapter loads an identity-audited external OpenLTM model, adds covariate projection and ordered quantile readout. The upstream model/layers are not bundled.

### CycleNet

- Official source: [CycleNet](https://github.com/ACAT-SCUT/CycleNet/tree/d807e51fc2dcd143885ee639d97965a7ab0926f4)
- Fixed commit: `d807e51fc2dcd143885ee639d97965a7ab0926f4`
- Upstream license: Apache-2.0; [full retained text](licenses/CycleNet-LICENSE.txt).
- License SHA-256: `50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a`.
- src/racepv/baselines/cyclenet_qr.py: wrapper loads an external CycleNet class, maps masked history and cycle phase, and adds a seven-quantile readout. RecurrentCycle and the upstream Model are not bundled or modified in this extract.

Each retained license was compared byte-for-byte with LICENSE fetched from the
official repository at the recorded commit. The CycleNet fixed repository tree
contains no NOTICE/COPYING-named file; its model implementation (including its
in-file contributor acknowledgement) is not redistributed in this extract.
The Apache license appendix placeholders are retained verbatim, not filled in
with invented copyright holders. Including this external dependency's license
does not relicense RaCE-PV wholesale under Apache-2.0.

## Paper-derived implementations

`src/racepv/baselines/modern.py` identifies TimeMixer++ and RQR-TFT as local,
paper-derived mechanism/protocol reproductions, not official upstream code or
certified exact reproductions. `s_mamba.py` is a local quantile adaptation of
the published S-Mamba idea that imports `mamba_ssm.Mamba`. Such imports and
architectural references do not transfer ownership of upstream implementations.
This review does not independently prove that every line is original.

## Runtime dependencies (not redistributed)

The C3 exploration additionally redistributes an extracted Time-Series-Library
TimeXer numerical core and three reference modules at commit
`4e938a1767106324dd753b2a44832bf870a0252e`, copyright (c) 2021 THUML @
Tsinghua University, MIT. The full notice is retained at
`src/racepv/exploratory/c3/vendor/timexer_upstream/LICENSE` and packaged with C3.
The task adaptation and import wiring are local; this is not the official full
TimeXer experiment pipeline. The extracted implicit-causal-mask guard fails
closed; C3 supplies explicit noncausal padding masks.

`pyproject.toml` declares NumPy, pandas, PyYAML, SciPy, scikit-learn and PyTorch.
Optional groups declare pvlib, LightGBM, NeuralForecast, pytorch-forecasting,
xLSTM, mamba-ssm, pytest, pytest-cov and Ruff; setuptools and wheel are build
dependencies. Only dependency declarations and local adapters are included,
not the dependency packages. Users must consult the notices shipped with the
specific versions they install; the historical version ranges are not a
license lock or a completed transitive dependency audit.

`external.py` uses external NeuralForecast APIs. The foundation-extension
modules refer to external TimesFM, Sundial and uni2ts/Moirai code/weights.
These external packages, checkpoints, model cards and datasets are not included.
Code licenses and weight licenses can differ: in particular, the historical
TimesFM3 weight-license identity in the adapter is non-commercial. The project
MIT license grants no rights to external weights or data. These adapters are
not an endorsement of external terms or a claim that omitted workflows run.

## Scope and rights confirmation

The owner confirmed the right to publish this extract on 2026-09-16, including
necessary collaborator/institution permissions and absence of relevant
confidentiality/patent restrictions. See the historical `docs/archive/source-history/RELEASE_STATUS.json`.
The review screened the release's source/configuration references and inspected
the named adapters against available upstream source. It is a bounded attribution
review, not an exhaustive plagiarism scan, an independent verification of
ownership or legal advice. If additional copied material is identified later,
preserve its actual notices and re-evaluate release rights; do not silently
attribute it to RaCE-PV.

General reference: [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
