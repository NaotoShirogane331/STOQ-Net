# STOQ-Net

STOQ-Net integrates stable and transient representations for intraday
photovoltaic forecasting with ordered quantiles. The paper setting uses
48 hours of history to predict the next six hours at 15-minute resolution.
A shared readout and one ordered head produce seven forecast quantiles;
a separate, gradient-isolated classifier provides power-state diagnostics.

This repository provides the model implementation and the paper's recorded
evaluation results. The result archive supports summary aggregation and
support-count verification. It does not contain the per-cell observations,
forecasts or trained checkpoints needed for independent re-scoring of every
reported metric or a complete experimental rerun.

## Start here

| Resource | Contents |
|---|---|
| [Paper results](paper_results/README.md) | Experiments, manuscript table/figure map, CSV records and evaluation supports |
| [Result workbook](paper_results/Results.xlsx) | English summaries of forecasting, diagnostic strata, classification, intervals and supports |
| [Model and input guide](docs/stoq-net.md) | Architecture, tensor channels, masks and configuration semantics |
| [Paper companion scope](docs/paper-companion.md) | Relationship between the source snapshot, historical model profile and current evaluation records |

## Verify evaluation supports

With Python, NumPy and pandas available, run from the repository root:

```bash
python paper_results/Support_Counts/reproduce_counts.py
```

This recomputes the six validation/test support counts used for Tables 7, 8
and 10 from exported cell IDs. It checks cells, uniqueness, target-local
array-days and daily counts through 36 comparisons. It does not train a
model or calculate forecasting scores from predictions.

The [result guide](paper_results/README.md) explains how to aggregate the
recorded per-seed scores and how their scoring populations differ. Original
observation data are identified through the source cited by the manuscript;
they are not redistributed in this repository.

## Run the model interface example

Use Python 3.10–3.12 and a suitable PyTorch build in an isolated environment.
From a checkout of this repository:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -e ".[dev]"
python examples/stoq_minimal.py
```

The default example uses synthetic inputs, random initialization and the
explicit CPU GRU backend. It checks the `[2, 24, 7]` tensor interface, ordered
quantiles and night zeroing; it produces no paper-performance scores.
These interface checks are separate from the evaluation records in
`paper_results/`.

The paper model profile uses **Mamba**. In a compatible Linux/CUDA environment
with a matching PyTorch build:

```bash
python -m pip install -e ".[dev,mamba]"
python examples/stoq_minimal.py --backbone mamba --device cuda
```

This is also an untrained interface example. Mamba requires its optional
dependency and compatible CUDA kernels; selecting Mamba never silently
falls back to GRU.

## Code and configuration

STOQ-Net maps to `SingleHeadRaCEPV`, registered as `M1`. The existing Python
distribution `race-pv` and import namespace `racepv` are retained.

| Entry | Purpose |
|---|---|
| [Model factory](src/racepv/models/research_variants.py) | `M1` registration and the single-head implementation |
| [Model configuration](configs/stoq-paper-model.json) | Constructor profile preserved from three historical runs |
| [Input ordering](src/racepv/io.py) | Registered history, future-known and static channels |
| [Training engine](src/racepv/training/engine.py) | Training and checkpoint-selection implementation |
| [Evaluation entry point](src/racepv/cli/evaluate.py) | Evaluation and calibration implementation requiring its input artifacts |

The model-only JSON and generic `base.yaml` are not a resolved execution
configuration for all ten seeds in the current result archive. See the
[scope guide](docs/paper-companion.md) before attempting a paper rerun.

## License and provenance

The existing [MIT license](LICENSE), [third-party notices](THIRD_PARTY_NOTICE.md)
and [upstream license files](licenses/) are retained. The source provider's
terms govern its original observation data. No original observations or
external pretrained weights are included here.

[Historical source records](docs/archive/README.md) document earlier code
synchronizations and their verification scope. They do not validate the
current paper's forecasting results. This companion is prepared in the
existing repository; its access follows the repository's visibility settings.
