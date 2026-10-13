# STOQ-Net — stable and transient representations for PV quantile forecasting

STOQ-Net combines a stable pathway (clear-sky anchoring and harmonic residuals)
with a transient sequence pathway. Their horizon representations feed one shared
readout and one ordered seven-quantile head. The historical paper configuration
uses a 48-hour history to forecast the next 6 hours on a 15-minute grid.

The existing implementation is **`SingleHeadRaCEPV`**, registered as **`M1`**.
The repository name and Python package remain `RaCE-PV-v1` and `race-pv`;
no model class, checkpoint identifier or original source file is renamed.

## Quick start: synthetic CPU interface demo

Use Python 3.10–3.12 in an isolated environment with an appropriate PyTorch
build. From this repository's root:

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python examples/stoq_minimal.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/stoq_minimal.py
```

The demo constructs a randomly initialized model with the **explicit GRU**
backend and synthetic inputs. It reports the model revision, actual backend
(`causal_conv_gru`) and output shape `[2, 24, 7]`. It checks finite, ordered,
bounded quantiles and night zeroing. It reads no data or checkpoints and reports
no forecast-performance scores.

## Paper model configuration

[configs/stoq-paper-model.json](configs/stoq-paper-model.json) contains the
complete registered model configuration shared by historical seeds 13, 29 and
47: `192` history steps, `24` forecast steps, `14/9/6` input dimensions and
quantiles `0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95`.
Its backend is explicitly **Mamba**, not the CPU demo's GRU.

For a qualified Linux/CUDA environment with a matching PyTorch build:

```bash
python -m pip install -e ".[dev,mamba]"
python examples/stoq_minimal.py --backbone mamba --device cuda
```

Native Mamba requires its optional dependency and compatible CUDA kernels.
Missing or unusable Mamba dependencies raise an error; this option does not
silently select GRU. The `mamba` installation extra is Linux-only.

This JSON is a **model-only** profile. Preprocessing, training, checkpoint
selection and interval calibration have separate contracts. The preserved
[configs/base.yaml](configs/base.yaml) is a generic historical example, not
the resolved STOQ-Net paper profile. See [the usage guide](docs/stoq-net.md)
for ordered input fields, masks, output semantics and historical training facts.

## Repository guide

| Entry | Purpose |
|---|---|
| [src/racepv/models/research_variants.py](src/racepv/models/research_variants.py) | Existing `M1` factory and `SingleHeadRaCEPV` implementation |
| [src/racepv/models/config.py](src/racepv/models/config.py) | Original configuration dataclass and validation |
| [src/racepv/io.py](src/racepv/io.py) | Registered input-channel order |
| [configs/stoq-paper-model.json](configs/stoq-paper-model.json) | Complete historical STOQ-Net model profile |
| [examples/stoq_minimal.py](examples/stoq_minimal.py) | Small random-initialization synthetic demo |
| [docs/stoq-net.md](docs/stoq-net.md) | Identity mapping and usage contract |
| [tests/test_stoq_minimal.py](tests/test_stoq_minimal.py) | New example/configuration interface tests |
| [STOQ_COMPANION.md](STOQ_COMPANION.md) | Scope and verification of this companion update |

## Verification

After installing the development dependencies:

```bash
python -m pytest -q tests experiments/c_pilot/test_pilot_policy.py -p no:cacheprovider
```

These are synthetic engineering checks. Native Mamba/CUDA execution and paper
performance reproduction are separate from the CPU checks. The actual checks
for this update are recorded in [STOQ_COMPANION.md](STOQ_COMPANION.md).
The earlier [VERIFICATION.md](VERIFICATION.md), [SNAPSHOT.json](SNAPSHOT.json)
and [RELEASE_STATUS.json](RELEASE_STATUS.json) remain historical records;
[SHA256SUMS](SHA256SUMS) covers the current repository payloads except itself.

## Retained research library

The original library and older C-family explorations remain available:
[C pilot](experiments/c_pilot/README.md) and
[architecture explorations](experiments/c_variants/README.md).
These historical development entries are distinct from the STOQ-Net `M1`
entrypoint and do not replace it. Existing CLI routes may require data,
bindings and scripts not supplied in this source-only repository.

## 中文说明

本仓库以论文模型 STOQ-Net 的使用入口为主线。它对应现有
`SingleHeadRaCEPV`（内部 ID：`M1`），由稳定与瞬态双表征、共享读出和
单个有序分位头构成。论文模型配置采用 Mamba；默认最小示例使用显式 GRU
和合成输入，仅检查接口。原源码、旧探索与历史记录保留，不修改实验成绩。

## License and distributed scope

The existing [MIT license](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICE.md) remain unchanged.
This is a source-only companion: no raw data, prepared observations,
predictions, weights, private logs, credentials or machine configuration are
included. Repository visibility remains private; this update does not publish
a public release or modify the manuscript. See also
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
