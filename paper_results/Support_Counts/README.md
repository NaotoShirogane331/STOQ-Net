# Supplementary data and evaluation supports

This directory contains evaluation-support indices, count records and the script used to check the supports for the structural comparison, the six-block TCN comparison and the main forecasting experiment. The accompanying result CSV files and `Results.xlsx` are in the parent directory. CSV files retain the recorded numerical precision and individual evaluation records. The workbook provides formatted result tables.

## Experiments and support counts

Each experiment uses its own scoring support. Configuration names alone do not identify a run. Use the experiment, configuration and seed together when joining records. In `common_support_metrics.csv`, `family=main`, `structural` and `matched_tcn` identify the main comparison, the single-array structural comparison and the twelve-array six-block TCN comparison, respectively. Their complete-model records remain separate. Diagnostic complete-model forecasting reuses the main complete-model forecasts and support.

| Experiment | Paper table | Split | Valid daylight cells (N) | Target-local array-days (D) |
|---|---|---|---:|---:|
| Single-array structural comparison | Table 7 | validation | 6,396 | 22 |
| Single-array structural comparison | Table 7 | test | 18,867 | 63 |
| Twelve-array six-block TCN comparison | Table 8 | validation | 73,353 | 262 |
| Twelve-array six-block TCN comparison | Table 8 | test | 223,859 | 762 |
| Main forecasting support | Table 10 | validation | 2,092,630 | 7,849 |
| Main and diagnostic forecasting support | Table 10 | test | 1,265,706 | 4,661 |

The validation row associated with Table 10 is supplied for support verification. The formal Table 10 result comparison reports test performance. Supports are shared by all configurations and seeds within each experiment and are counted once, without multiplying by configurations, seeds or quantiles.

### Cells, target intervals and local days

A forecast cell is one valid `(array, issue_time_UTC, lead)` combination. Forecasts are issued hourly, with 24 leads on a 15-minute grid. For the one-based lead k, the target interval is `[issue + 15*(k-1) minutes, issue + 15*k minutes)`. Seven quantile predictions belong to this one cell.

N is the number of distinct valid daylight forecast cells. D is the number of distinct `(array, target_local_date)` pairs represented by at least one retained cell. Convert the **target interval start time** to `Australia/Darwin` (UTC+09:30) before extracting its local date. Days with no retained cell are not counted. Calendar duration multiplied by the number of arrays is not a substitute for this deduplication.

Overlapping forecasts can refer to the same physical target interval. `distinct_array_target_intervals` counts distinct `(array, target interval start)` pairs, while N counts the contributing issue-lead forecast cells. The fields `issue_utc_array_days`, `target_utc_array_days` and `target_local_array_days` use three different date conventions and must not be interchanged.

### Selection and date boundaries

The structural comparison uses array 1A. The six-block TCN comparison uses all twelve arrays. Both use UTC issue windows `[2020-11-03 00:00, 2020-11-24 00:00)` for validation and `[2023-11-06 00:00, 2024-01-08 00:00)` for test. Historical-input eligibility is checked first, followed by the finite, observed daylight target mask applied to individual cells. Target intervals may extend beyond the restricted issue-date range.

The main support requires eligible history and valid observations for every daylight target in the full forecast window. Retained daylight cells are then extracted from those eligible windows. Thus the restricted comparisons and the main comparison retain their own N and D.

For the structural test support, 62 UTC issue array-days correspond to 63 target-local array-days. For the six-block TCN test support, the first six arrays each contribute 63 target-local days and the remaining six each contribute 64, totaling 762. Its validation support has ten arrays with 22 target-local days and two with 21, totaling 262.

### Forecasting, classification and unavailable groups

The 1,265,706-cell forecast support and the 1,210,082-cell classification support differ. Classification additionally requires a valid observed power-state label. The 55,624 forecast cells with an unavailable state label are retained in the forecasting records as `stratum_type=true_power_state, stratum=unavailable`; they are not classification cells.

The other stratification uses daily output conditions derived from observed daylight power relative to the clear-sky reference on the same unique grid intervals. Its low, middle and high groups use frozen training-day quantile thresholds. An unavailable group retains days without a valid daily output-condition classification. The CSV value `daily_output_condition` names this power-based grouping, not an observed weather class. The unavailable output-condition group contains 460 forecast cells and 400 classification cells, representing six array-days.

The low, middle, high and unavailable output-condition groups have 1,491, 1,600, 1,564 and 6 forecast array-days, respectively, totaling 4,661. Valid power-state groups and the unavailable-state group can share the same array-day, so their array-day counts need not add. Cell counts remain specific to each classification or forecast partition.

## Result files

Paths in this table are relative to the parent `Supplementary_Data` directory. A row in a per-seed table is an individual evaluation record, not a training example or a raw prediction.

| File | Record content |
|---|---|
| `Results.xlsx` | Formatted English summaries; complete evaluation records remain in CSV files. |
| `main-per-seed.csv` | 130 main-comparison records: 13 configurations and ten seeds, with validation and test scores. |
| `main-strata-per-seed.csv` | 10,400 main-comparison records by validation/test split and lead, array or season, including overall and daylight qCRPS and exact cell counts. |
| `structural-ablation-per-seed.csv` | 60 structural-comparison records with validation/test qCRPS, test nRMSE and support counts. |
| `six-block-tcn-comparison-per-seed.csv` | 20 complete-model and six-block TCN records, including paired qCRPS differences at matching seeds. |
| `common_support_metrics.csv` | 30 records for the three complete-model families on the same 18,475 test cells. |
| `diagnostic-overall-per-seed.csv` | 30 overall diagnostic-forecast records and paired qCRPS differences from the complete model. |
| `diagnostic-strata-per-seed.csv` | 180 diagnostic-forecast records for valid power states and low, middle and high daily output conditions. |
| `diagnostic-unavailable-groups-per-seed.csv` | 60 diagnostic-forecast records for unavailable power-state and unavailable daily output-condition groups. |
| `power-state-classification-per-seed.csv` | 609 classification records covering overall, lead-specific and daily output-condition evaluations. |
| `power-state-confusion-counts-per-seed.csv` | 21 overall confusion matrices with nine integer counts each and corresponding classification metrics. |
| `prediction-intervals-per-seed.csv` | 780 records: 13 configurations, ten seeds, two nominal levels and three interval treatments. |
| `calibration_corrections.csv` | 12,480 lead-specific calibration records for standard and expansion-only CQR. |
| `array-window-counts.csv` | 48 retained-window counts: twelve arrays in each of four temporal partitions. |
| `partition-support-counts.csv` | Four temporal partitions with grid, candidate-issue and retained-support counts. |
| `test-availability-cutoffs.csv` | Twelve array-specific observation cutoffs and retained daylight cell counts. |
| `figure_data/figure_statistics.csv` | 645 plotted summary records with source files, selection filters, means and sample SDs. |
| `figure_data/power-state-confusion-proportions.csv` | The three-by-three complete-model confusion proportions with explicit true-state row labels. |
| `Support_Counts/support_summary.csv` | Six support-count summaries, including local/UTC date conventions and target-time boundaries. |
| `Support_Counts/daily_valid_counts.csv` | 13,619 support-array-local-date records with valid cell and distinct-target counts. |
| `Support_Counts/data/valid_support_validation.npz` | Validation support-cell IDs, array labels and time mappings. |
| `Support_Counts/data/valid_support_test.npz` | Test support-cell IDs, array labels and time mappings. |
| `Support_Counts/reproduce_counts.py` | Support-count verification using the two count CSV files and the two NPZ files. |
| `Support_Counts/README.md` | Data definitions, file contents and support-decoding instructions. |

### Common field definitions

- `model`, `config` and `configuration` identify the configuration within the file's experiment. The recorded seed identifies the evaluation run. A blank seed for State persistence is intentional because that deterministic baseline is evaluated once; it is not an additional random seed.
- `split` identifies training, validation, calibration or test. Prefixes `val_` and `test_` identify the scoring split in metric columns. `daylight` and `overall` are distinct scoring populations. Overall scores include the rule-zeroed non-daylight cells.
- `qcrps`, `nmae`, `nrmse`, `nmbe` and `r2` are the manuscript's dimensionless forecasting metrics. The normalization is by photovoltaic capacity. Positive nMBE denotes net overprediction. Lower qCRPS, nMAE and nRMSE are better; higher R-squared is better.
- In `main-strata-per-seed.csv`, `qcrps` is the overall score and `qcrps_daylight` is the daylight-only score. `n_total`, `n_daylight` and `n_night` give their cell counts. The retained technical `scope=overall` does not remove the separately supplied daylight columns. `stratum=lead_h` uses hours, `array` uses the zero-based array index, and `season` uses the UTC target-start month.
- In classification and diagnostic-stratum records, `n` is the number of scored cells, and `array_days` is the represented target-local array-day count. `support_stable`, `support_ramp_up` and `support_ramp_down` count the true classes. The nine `true_*_pred_*` fields are integer confusion counts. Accuracy and F1 are fractions, not percentages. Macro-F1 averages the fixed three classes.
- Classification `stratum_type=lead` uses `stratum` in **hours** from 0.25 to 6.0. By contrast, `lead` in `calibration_corrections.csv` is the **one-based lead index** from 1 to 24. Its lead time in hours is `lead/4`.
- In interval files, `level` is nominal coverage in percent (80 or 90), `coverage_percent` is the observed percentage, `width` is normalized interval width and `interval_score` is the interval score. `n_test` is the number of scored cells, `hit_count` is the number inside the interval and `empty_interval_count` counts empty intervals. `nmae_same_support` records the median error on that same support.
- In calibration records, `n_calibration` is the calibration cell count for that lead, `rank` is the one-based order-statistic rank, `correction` is the signed or expansion-only adjustment and `available` states whether the adjustment is available. Standard CQR allows a negative signed correction; expansion-only CQR replaces it with zero. The selected calibration count is 2,400 per lead. The ranks are 1,921 and 2,161 for the 80% and 90% levels.
- In `figure_statistics.csv`, `n` is the number of evaluation records used in the summary, normally the number of seeds; it is not the number of forecast cells. `sd` is sample SD when n is greater than one. The 28 records for the single deterministic evaluation retain `sd=0` as a plotting placeholder; an across-seed sample SD cannot be estimated from one evaluation, and the workbook displays `n.a.`. `source` and the JSON `filters` identify the selected source records.
- `array_index` and `cutoff_grid_index` are zero-based indices. `array_id` is the source array label. The `natural_daylight_cells` field in the cutoff table is the retained daylight support count for that array. In `partition-support-counts.csv`, `table7_n` and `table8_n` retain the structural and six-block TCN support-count names. Zero entries outside their evaluated splits do not describe a model score.
- `run_file` in `common_support_metrics.csv` preserves the original run-file identifier for traceability. The referenced prediction JSON files are **not included** in this archive. These strings do not provide prediction or target values.

### Configuration and treatment labels

Technical identifiers retained in previous analyses are mapped to descriptive labels in the result CSV files. The mapping changes text only and does not combine complete-model records across experiments.

| Technical identifier | Result label |
|---|---|
| `Full` | STOQ-Net (complete model) |
| `no-stop` | Gradient-coupled diagnostic |
| `no-diagnostic` | Diagnostic branch removed |
| `minus-StableLearned` | Stable learned outputs zeroed |
| `minus-Transient` | Transient output zeroed |
| `remove-StableLearned` | Stable branch removed |
| `remove-Transient` | Transient branch removed |
| `Mamba-to-TCN` | Two-block TCN replacement |
| `TCN-parameter-matched` | Six-block TCN control |
| `state-persistence` | State persistence |
| `Raw` | Uncalibrated |
| `Expansion-only` | Expansion-only CQR |
| `weather` | daily_output_condition |

`STOQ-Net` remains the main-comparison label. `Standard CQR` retains its original label.

### Figure identifiers

`figure_statistics.csv` retains `figure` as the asset identifier and adds `paper_figure` for the manuscript numbering. Panel identifiers remain within their original assets.

| Asset identifier | Paper figure |
|---|---|
| Figure05 | Figure 3 |
| Figure06 | Figure 4 |
| Figure07 | Figure 5 |
| Figure08 | Figure 6 |
| Figure09 | Figure 7 |
| Figure10 | Figure S2 |
| Figure11 | Figure S3 |

In `power-state-confusion-proportions.csv`, `true_state` labels the rows and `pred_stable`, `pred_ramp_up` and `pred_ramp_down` label the predicted-state columns. Each seed's counts are normalized within each true class before the proportions are averaged across the ten complete-model seeds.

## Support files and NPZ decoding

`support_summary.csv` preserves both `source_reported_cells` and `recomputed_cells`. Its target-time boundaries are the first and last represented target interval starts in UTC. In `daily_valid_counts.csv`, `support_id` identifies the experiment and split, `array_index` is zero-based, `array_id` gives the array label, `date` is the target-local date, `cells` counts forecast cells and `unique_targets` counts distinct physical target starts for that array and date. Each support-array-date key occurs once.

The support IDs `Table7_validation`, `Table7_test`, `Table8_validation`, `Table8_test`, `Table10_validation` and `Table10_test` correspond to the experiment rows in the support-count table above. These stable technical IDs are retained for the verification script.

The two NPZ files contain only scoring-cell IDs, array labels and time mappings. They contain **no observed power values, forecast quantiles or model predictions**. All NPZ keys are retained:

| NPZ key | Definition |
|---|---|
| `array_id` | Source array labels in zero-based array-index order. |
| `grid_start_time_utc_minutes` | UTC grid timestamps stored as integer minutes since 1970-01-01 00:00 UTC; consecutive positions are 15 minutes apart. |
| `candidate_issue_time_utc_minutes` | Candidate issue timestamps in the same integer-minute UTC convention, one per issue index. |
| `origin_grid_index` | Zero-based target-start grid index for the first lead of each candidate issue. |
| `table7_indices` | Valid daylight cell IDs for the single-array structural support. |
| `table8_indices` | Valid daylight cell IDs for the twelve-array six-block TCN support. |
| `main_daylight_indices` | Valid daylight cell IDs for the main and diagnostic forecast support. |
| `source_note` | Reference to this `README.md`. |

For a stored nonnegative cell ID x, let `n_issues = len(origin_grid_index)`. Decode it using integer quotient and remainder:

```python
array_index, within_array = divmod(x, n_issues * 24)
issue_index, lead0 = divmod(within_array, 24)
lead = lead0 + 1
target_grid_index = origin_grid_index[issue_index] + lead0
array_label = array_id[array_index]
issue_utc_minutes = candidate_issue_time_utc_minutes[issue_index]
target_start_utc_minutes = grid_start_time_utc_minutes[target_grid_index]
```

`array_index`, `issue_index`, `lead0` and `target_grid_index` are zero-based. The timestamp is the start of the target interval, whose end is 15 minutes later. Convert `target_start_utc_minutes` to Australia/Darwin before extracting `target_local_date` for D. For the same array and physical target start, several issue indices can contribute distinct forecast cells.

To verify the counts, run the bundled script from this directory with Python, NumPy and pandas:

```bash
python reproduce_counts.py
```

It reads `support_summary.csv`, `daily_valid_counts.csv` and the two NPZ files. It checks reported/recomputed cell totals, unique cell IDs, target-local array-day totals and every daily cell count for the six supports, producing 36 comparisons. It does not train a model, regenerate forecasts or recompute forecasting/classification scores.
