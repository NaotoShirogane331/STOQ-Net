# STOQ-Net model and input guide

## Identity and architecture

The paper name **STOQ-Net** maps to the existing **M1** entry in
[`build_research_candidate`](../src/racepv/models/research_variants.py), which
constructs `SingleHeadRaCEPV`. The model revision is
`racepv.research-single-head.v1`. Use the original factory:

```python
from racepv.models.research_variants import build_research_candidate

model, objective = build_research_candidate("M1", config)
```

The stable pathway supplies a persistence/clear-sky location and harmonic
residual, alongside learned horizon features. The transient pathway supplies
shared horizon features. Concatenation is followed by one shared readout and
one ordered quantile head. Detached state diagnostics are separate from
forecast fusion.

[`stoq-paper-model.json`](../configs/stoq-paper-model.json) preserves every
registered constructor field for historical seeds 13/29/47. Some inherited
fields (`active_experts`, `fusion_mode`, `router_mode`) retain the base-class
schema. Do not replace them with the output metadata's `single_head`/`absent`:
those strings describe the instantiated M1 model, not valid constructor
values. The readout width is derived by the model's parameter-matching rule,
not a JSON configuration field.

JSON arrays must be restored to tuples for `quantiles`,
`harmonic_periods_hours`, `active_experts`,
`clear_sky_history_feature_indices` and `clear_sky_future_feature_indices`.
The example's `load_model_config()` implements that conversion without
rewriting the original dataclass or validation.

## Tensor interface

```python
output = model(history_values, history_mask, future_known, site_metadata)
quantiles = output["quantiles"]
```

| Tensor | Historical shape | Meaning |
|---|---|---|
| `history_values` | `[B, 192, 14]` | Historical power-derived and weather features |
| `history_mask` | `[B, 192, 14]` | Per-feature observation-validity mask; boolean in the demo |
| `future_known` | `[B, 24, 9]` | Deterministically known clear-sky, geometry, calendar and lead fields |
| `site_metadata` | `[B, 6]` | Fixed array descriptors |
| `output["quantiles"]` | `[B, 24, 7]` | Capacity-normalized ordered forecast quantiles |

All channels are listed in their original zero-based order below, matching
[`racepv/io.py`](../src/racepv/io.py). Future measured power and weather are
not inputs. The historical profile has `forecast_weather_mode="off"`.

### Historical channels

| Index | Name | Description |
|---:|---|---|
| 0 | `power_norm` | Capacity-normalized power, bounded to `[0,1.05]` in preprocessing |
| 1 | `clear_sky_power_norm` | Capacity-normalized clear-sky reference |
| 2 | `clear_sky_residual` | Power minus clear-sky reference |
| 3 | `power_diff_1` | Adjacent-step first power difference |
| 4 | `power_diff_2` | Adjacent-step difference of first differences |
| 5 | `rolling_volatility_1h` | Sample standard deviation of the last four first differences |
| 6 | `rolling_volatility_3h` | Sample standard deviation of the last twelve first differences |
| 7 | `clear_sky_index` | Power divided by `max(clear_sky_power_norm,0.01)` |
| 8 | `power_observed` | Original observation provenance flag |
| 9 | `power_imputed` | Qualified preprocessing gap-fill provenance flag |
| 10 | `ghi_z` | Standardized historical GHI |
| 11 | `temperature_z` | Standardized historical temperature |
| 12 | `humidity_z` | Standardized historical humidity |
| 13 | `wind_speed_z` | Standardized historical wind speed |

Weather scaling uses each array's training mean and population standard
deviation, reused for later partitions. A difference is a numerical step
difference, not a derivative per minute. Volatility dependencies include the
power positions needed to form those differences.

### Future-known channels

| Index | Name | Description |
|---:|---|---|
| 0 | `clear_sky_power_norm` | Clear-sky reference |
| 1 | `solar_elevation` | Apparent solar elevation in degrees |
| 2 | `solar_azimuth_sin` | Sine of solar azimuth |
| 3 | `solar_azimuth_cos` | Cosine of solar azimuth |
| 4 | `hour_sin` | Sine of local hour, period 24 hours |
| 5 | `hour_cos` | Cosine of local hour, period 24 hours |
| 6 | `day_of_year_sin` | Sine of `(day_of_year - 1) / 365.2425` times `2π` |
| 7 | `day_of_year_cos` | Corresponding cosine |
| 8 | `lead_time_hours` | Target interval-end lead: `0.25, 0.50, …, 6.00` hours |

Clear-sky, solar and calendar features use target interval starts; lead labels
refer to interval ends. The M1 forward computes its lead vector from the
configured cadence and uses the hour sine/cosine channels as its physical
clock. Prepared channel order remains fixed.

### Static channels

| Index | Name | Historical scaling |
|---:|---|---|
| 0 | `latitude_scaled` | Latitude / 90 |
| 1 | `longitude_scaled` | Longitude / 180 |
| 2 | `log_capacity_kw_scaled` | `ln(1 + capacity_kW) / 10` |
| 3 | `surface_tilt_scaled` | Tilt / 90 |
| 4 | `surface_azimuth_scaled` | Azimuth / 360 |
| 5 | `has_geometry` | Both angle fields are available |

Missing angles encode zero, with availability indicated separately.

## Masks and output handling

Historical validity is channel-specific. The original model validates input
shapes and finite future/static values, combines the historical mask with
finiteness, and fills unobserved history through its registered imputation
path. Do not turn a missing target into a zero training label. Target masks
belong to the training and scoring contract, not this forward interface.

The two power-provenance channels are features, not substitutes for the full
observation mask. `power_imputed` describes preprocessing gaps of at most two
steps with qualified observed endpoints before issue time and in the same
partition; it does not mark every value filled inside the model.

The raw forecast levels are `0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95`.
The model clamps outputs to `[0,1.05]` and zeroes all quantiles when the
future clear-sky reference is below `0.01`. With no observed historical power,
the existing forward falls back to the clear-sky reference, followed by the
same output constraints. CQR interval adjustment is a separate postprocessing
stage; the minimal example neither estimates nor applies it.

## Model profile versus execution protocol

The JSON file is consumed by the example's loader and `RaCEPVConfig`, not by
the generic training CLI. It specifies architecture and feature indices; it
does not specify a data binding, optimizer state, trained weights or
calibration parameters.

The historical M1 training records describe two AdamW optimizers with learning
rate `0.001`, weight decay `0.0001`, batch size `64`, gradient clipping norm
`1.0`, AMP FP16 and zero data-loader workers. The maximum/minimum epoch
budgets were `60/10`, with patience `8`. Checkpoints were selected by strictly
lower validation daylight qCRPS; ties retained the earlier checkpoint. The
window stride was four 15-minute steps (one hour). These facts describe the
historical protocol, not a new runnable training campaign provided here.

| Configuration | Use |
|---|---|
| `configs/stoq-paper-model.json` | Complete historical M1 model constructor fields; explicit Mamba |
| `configs/base.yaml` | Preserved generic historical example with different stride/epoch settings |
| Other existing YAML files | Historical baseline/extension examples |

The CPU demo overrides only the backbone to explicit `gru`, switches to
evaluation mode, and runs under `torch.inference_mode()` with random weights.
It uses synthetic feature values to demonstrate tensor wiring, not to emulate
a real site's physical record. Its seed 13 initializes the demo; it is not a
historical trained seed13 checkpoint. The actual backend and revision are
printed so that the two configurations are distinguishable.

## Dependencies and verification scope

Package requirements and extras remain in [`pyproject.toml`](../pyproject.toml).
Use an appropriate PyTorch build before installing the source. Native Mamba
requires a compatible Linux/CUDA installation; the Linux-only optional extra
does not provide a Windows GRU-to-Mamba replacement. Requesting `mamba` leaves
dependency/construction errors intact, without a silent fallback.

Use `python examples/stoq_minimal.py --help` for supported demo options.
No data, weights, prediction arrays or private execution bindings are supplied.
See [`STOQ_COMPANION.md`](../STOQ_COMPANION.md) for the checks actually performed.
