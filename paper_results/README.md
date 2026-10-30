# Paper results and evaluation supports

This directory accompanies *STOQ-Net: Integrating stable and transient representations for intraday photovoltaic forecasting with ordered quantiles*. It contains the 24 files from the submission's `Supplementary_Data` directory, with their contents and subdirectory structure preserved, plus this guide. Start with [Results.xlsx](Results.xlsx) for formatted tables, the CSV files for individual evaluation records, and the [detailed support guide](Support_Counts/README.md) for definitions and NPZ decoding.

The manuscript identifies the upstream observation source as the [Desert Knowledge Australia Solar Centre, Alice Springs](https://dkasolarcentre.com.au/download?location=alice-springs). Raw observation series are not included here. This archive also lacks per-cell observed targets and forecasts, trained checkpoints, and the prediction JSON files named by `run_file`. It supports reproduction of support counts and aggregation of recorded results, rather than full independent re-scoring or retraining.

## Experiment identities

Join records by **experiment, configuration and seed**. A repeated complete-model label does not identify the same run across experiments.

| Experiment | Scope and complete-model identity | Paper location |
|---|---|---|
| Main forecasting and intervals | 13 configurations, ten seeds, 12 arrays. Main STOQ-Net forecasts are reused for interval evaluation. Test daylight support: 1,265,706 cells and 4,661 target-local array-days. | Tables 5, 11–12; Figures 3, 6–7; Tables S4, S11 |
| Structural comparison | Six configurations on array 1A, including output-zeroing, branch-removal and two-block TCN variants. Uses its own complete-model runs. Validation/test daylight cells: 6,396/18,867; array-days: 22/63. | Table 7; Figure 4; Table S5 |
| Six-block TCN comparison | Complete model versus six-block TCN control over 12 arrays. Uses a third complete-model run family. Validation/test daylight cells: 73,353/223,859; array-days: 262/762. | Table 8; Table S6 |
| Diagnostic evaluation | Complete model, gradient-coupled diagnostic (`no-stop`) and diagnostic branch removed (`no-diagnostic`) provide forecasts. Diagnostic complete-model rows reuse the main STOQ-Net forecasts. Complete model and gradient-coupled diagnostic provide classifications; State persistence is evaluated once. | Tables 9–10; Figure 5; Tables S7–S10 |

Main validation daylight support contains 2,092,630 cells and 7,849 array-days. Diagnostic forecasting uses the main test daylight support; classification additionally requires a valid power-state label, leaving 1,210,082 cells and excluding 55,624 forecast cells with unavailable labels. Low/middle/high/unavailable daily output conditions are retrospective power-based groups, not observed weather classes.

The three complete-model families are also compared on their common **18,475 test cells** in Table 6. In `common_support_metrics.csv`, `family=main`, `structural` and `matched_tcn` distinguish those families.

## File map

A per-seed row is an evaluation record, not a training example or a raw prediction. All links below are relative to this directory.

| Files | What they support |
|---|---|
| [Results.xlsx](Results.xlsx) | Formatted summaries of the CSV records. |
| [main-per-seed.csv](main-per-seed.csv) · [main-strata-per-seed.csv](main-strata-per-seed.csv) | 130 main run summaries and 10,400 validation/test records by lead, array and season, including separate overall/daylight qCRPS and counts; Figures S2–S3. |
| [structural-ablation-per-seed.csv](structural-ablation-per-seed.csv) · [six-block-tcn-comparison-per-seed.csv](six-block-tcn-comparison-per-seed.csv) · [common_support_metrics.csv](common_support_metrics.csv) | Structural, six-block TCN and common-support complete-model comparisons. |
| [diagnostic-overall-per-seed.csv](diagnostic-overall-per-seed.csv) · [diagnostic-strata-per-seed.csv](diagnostic-strata-per-seed.csv) · [diagnostic-unavailable-groups-per-seed.csv](diagnostic-unavailable-groups-per-seed.csv) | Diagnostic forecasting overall, by valid power state/output condition, and for unavailable groups. |
| [power-state-classification-per-seed.csv](power-state-classification-per-seed.csv) · [power-state-confusion-counts-per-seed.csv](power-state-confusion-counts-per-seed.csv) | 609 overall/lead/condition classification records and 21 overall confusion matrices. |
| [prediction-intervals-per-seed.csv](prediction-intervals-per-seed.csv) · [calibration_corrections.csv](calibration_corrections.csv) | 80%/90% interval coverage, width and interval score for uncalibrated, standard CQR and expansion-only CQR treatments; lead-specific calibration adjustments. |
| [array-window-counts.csv](array-window-counts.csv) · [partition-support-counts.csv](partition-support-counts.csv) · [test-availability-cutoffs.csv](test-availability-cutoffs.csv) | Retained windows, temporal-partition support and array-specific observation cutoffs. |
| [figure_statistics.csv](figure_data/figure_statistics.csv) · [power-state-confusion-proportions.csv](figure_data/power-state-confusion-proportions.csv) | Plotted summary values with source filters and row-labelled confusion proportions. `paper_figure` gives current manuscript numbering; `figure` retains asset IDs. |
| [support_summary.csv](Support_Counts/support_summary.csv) · [daily_valid_counts.csv](Support_Counts/daily_valid_counts.csv) | Six experiment/split support summaries and counts by array and target-local date. |
| [validation support export](Support_Counts/data/valid_support_validation.npz) · [test support export](Support_Counts/data/valid_support_test.npz) | Experiment exports containing scoring-cell IDs, array labels and time mappings; no observed power or predictions. |
| [reproduce_counts.py](Support_Counts/reproduce_counts.py) · [Support_Counts/README.md](Support_Counts/README.md) | Count verification, field definitions, configuration mappings and support-cell decoding. |

## Reading and reproducing summaries

Metrics first pool valid forecast cells **within each seed**; summary means weight the ten seeds equally. Sample SD uses denominator nine (`ddof=1`). Paired differences subtract the complete-model metric at the matching seed, and their SD is calculated from those differences. The seed IDs are 13, 29, 47, 61, 79, 101, 127, 151, 181 and 211.

State persistence is deterministic: it extends the last pre-issue power state across leads, using the training-majority class when that historical state is unavailable. Its blank seed is intentional, and it has no across-seed sample SD. The figure records' `sd=0` for this single evaluation is a plotting placeholder; the workbook displays `n.a.`.

The confusion-count file retains all **nine integer `true_*_pred_*` counts**: true and predicted classes each span stable, ramp-up and ramp-down. Accuracy and F1 are computed from each evaluation's pooled counts before seed aggregation. Macro-F1 always averages all three classes. Confusion proportions normalize each seed's counts within each true-class row before averaging seeds; they are not pooled integer counts.

Keep scoring populations distinct. `overall` includes rule-zeroed non-daylight cells; `daylight` uses its own denominator. In classification records, lead strata are in hours (0.25–6.0); in calibration records, `lead` is the one-based index (1–24). Coverage is reported in percent; accuracy and F1 are fractions. In the figure statistics, `n` counts evaluation records, usually seeds, rather than forecast cells.

## Reproduce support counts

From the repository root, with Python, NumPy and pandas available:

```bash
python paper_results/Support_Counts/reproduce_counts.py
```

The script reads the two count CSVs and both NPZ exports and checks cell totals, unique cell IDs, target-local array-days and every daily cell count for six supports (36 comparisons). `Table7_*`, `Table8_*` and `Table10_*` identify structural, six-block TCN and main/diagnostic forecasting supports. The Table 10 validation support is supplied for count verification; Table 10 reports test results.

One cell is a distinct `(array, UTC issue time, lead)` combination; its seven quantiles count once. Array-days deduplicate `(array, target-local date)`, converting the **target interval start** to `Australia/Darwin`. Overlapping issues can contribute separate cells for the same physical target interval. Counts are shared across configurations and seeds, so do not multiply them by seeds, configurations or quantiles. The count script does not train models, regenerate forecasts or recompute forecast/classification scores.
