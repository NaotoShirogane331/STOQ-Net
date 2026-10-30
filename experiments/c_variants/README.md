# Four historical C-family development variants

This 2026-10-05 source-only update publishes the numerical implementations
used in earlier development screens. Original runtime-v4 library files remain
byte-identical. Numerical forward/training methods are retained; local imports
are package-relative. Only the historical `candidates()` / `validate_candidate()`
functions are extracted from campaign policies. SSH, process supervision,
expired campaign clocks, datasets, predictions and checkpoints are excluded.
This repository does not launch a historical cloud campaign.

| Variant | Architecture change | Historical development status |
|---|---|---|
| C2 | Solar-reference-conditioned latent branches at scales 1/2/4; covariate/future attention; one ordered head. Capacity-matched SS control uses 1/1/1. | Raw-loss MS improved over SS at both seeds 29/47. This is not a comparison against all final-test baselines. |
| C3-TimeXer-PV | Power patch/global token, covariate cross-attention, future/site-conditioned ordered head. | Both paired one-epoch scores were worse than C; not advanced. |
| C-LG | Independent center/lower-gap/upper-gap role weights; repeated original router rows; +390 parameters. | Mixed seed-level outcomes; advancement gate failed. |
| C-DH-linear | Zero-initialized 96-to-8 head from last 24 power/residual/mask slots; bounded ±0.08 location shift for first eight leads; +776 parameters. | Both paired full-horizon scores were worse than C; not advanced. |

Historical screens used fresh seeds 29/47, one complete epoch, batch 64,
LR 0.001, weight decay 0.0001, seven fixed quantiles and train/validation only.
The repeatedly used validation split is development evidence, not a blind test.
Publishing these sources does not change any previously delivered main ranking.

## Imports and synthetic use

```python
from racepv.exploratory.c2.c2_model import C2, C2Config
from racepv.exploratory.c3.c3_model import C3, C3Config
from racepv.exploratory.location_gap.lg_adapter import LocationGapScientificAdapter
from racepv.exploratory.direct_history.model_adapter import DirectHistoryC
from racepv.models import RaCEPVConfig
from racepv.reliability_experiment import ReliabilityControlRaCEPV

c2 = C2(C2Config(14, 9, 6, variant="multiscale"))
c2_single_scale = C2(C2Config(14, 9, 6, variant="single_scale"))
c3 = C3(C3Config(14, 9, 6, 0, 1, 0))

# Explicit CPU-GRU is synthetic qualification only, not production Mamba.
cfg = RaCEPVConfig(history_features=14, future_features=9, site_features=6,
    future_hour_sin_index=4, future_hour_cos_index=5, backbone="gru",
    clear_sky_history_feature_indices=(2, 7))
lg = LocationGapScientificAdapter(ReliabilityControlRaCEPV(cfg), backend="cpu-gru")
dh = DirectHistoryC(cfg)
```

Each model consumes history `[B,192,14]`, binary history masks, deterministic
future features `[B,24,9]` and site metadata `[B,6]`. Future observed weather is
rejected. `quantiles` and `state_logits` are returned; the state path is detached
from forecasting. C-LG role weights are `[B,24,role,expert]`, not the old single
gate schema. Preserve `state_dict` tensor checkpoints, not pickled whole models.

`c2_adapter.C2ResearchTrainer` and `c3_adapter.C3ResearchTrainer` train using
raw ordered quantile qCRPS, retaining projected validation scoring and detached
state CE. C-LG and C-DH retain the original projected loss. Raw C2/C3 model
outputs are not physically projected: use their trainer `predict_batch` for
clamp `[0,1.05]` and daylight-mask zeroing. This distinction is intentional.
C3 versus C also changes loss/capacity, so it is not a pure architecture ablation.

`lg_adapter` / `dh_adapter` and the C2/C3 adapters expose `build(signature,row)`
and `trainer(...)`; rows come from their respective minimal policy module.
C-LG/DH require an explicit `backend="cpu-gru"` for synthetic CPU builds or
`"cuda-mamba"` for the original qualified production backend. C2/C3 do not
silently substitute a GRU. Missing CUDA/Mamba dependencies are not a fallback.
ResearchTrainer imports original Linux recovery code; training-adapter use is
Linux-oriented. Windows tests use an import-only `fcntl` stub whose lock calls
always fail. Native locks/CUDA/AMP are not validated by this source update.
C-DH dispatch additionally requires the exact retained ResearchTrainer source
SHA; changing the base source fails closed rather than silently adapting it.

## Verification and scope

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests experiments/c_pilot/test_pilot_policy.py
```

Tests use small synthetic tensors; they cover shape/order, missingness, raw-loss
gradient recovery, physical projection, independent state gradients, initialization
and RNG preservation, role weights, direct-history locality, state-dict roundtrip,
actual small synthetic optimizer steps, and copied TimeXer encoder parity.
No real experiment is rerun. See the [historical verification record](../../docs/archive/source-history/VERIFICATION.md) for those results/skips.

The MIT Time-Series-Library core is pinned to
`4e938a1767106324dd753b2a44832bf870a0252e`.
Its license and three reference modules are retained under the C3 vendor folder.
C3 is our photovoltaic task adaptation, not the official full TimeXer pipeline.
External data rights, provider timing guarantees and blind-heldout independence
are not established by publishing code; previous limitations remain.
