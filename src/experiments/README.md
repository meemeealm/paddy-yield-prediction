# Experiment Details

* **Target variable**: `yield_ton_ha` (Paddy yield in tons per hectare)
* **Input features**: Sentinel-1 and Sentinel-2 metrics, climate metrics, merged static (climate and soil) datasets via province lookup, global target-encoded `soil_encoded`, and one-hot encoded `season`
* **Prediction task**: Paddy yield prediction
* **Evaluation metrics**: R2, Mean Fold R2, RMSE, MAE, and Bias

Test: held-out last year (2025), models trained on earlier years.

**Result:** a per-province-season **linear trend** beats every ML model for forecasting known provinces. Sentinel-2 adds almost nothing to model.

| Model (2025 test) | RMSE | R² |
|---|---|---|
| Linear trend (per province+season) | **0.385** | **0.896** |
| Previous year | 0.408 | 0.883 |
| Trend + XGBoost residual | 0.419 | 0.877 |
| Province+season mean | 0.428 | 0.872 |
| XGBoost benchmark | 0.455 | 0.855 |
| Ridge (province+season+year) | 0.645 | 0.708 |

---

## Setup

```bash
pip install pandas numpy scikit-learn>=1.3 xgboost pyyaml pyarrow matplotlib
```

All scripts read `config.yaml` from the current folder. Run them from the folder that contains it.

## Scripts

| Script | Purpose |
|---|---|
| `run_experiments.py` | Tunes and compares models in `config.yaml` with grouped CV on years < `test_year`, then scores the held-out year. Also prints null checks and a per-province error report. |
| `baseline_rules.py` | Scores 3 simple rules (province+season mean, previous year, linear trend) on the test year. |
| `baseline_linear.py` | Ridge models with a year term, alpha chosen by time-based CV. |
| `final_residual_model.py` | Trend baseline plus XGBoost on the residual. |
| `save_trend_model.py` | Fits the linear trend on all years and saves it to `linear_trend_model.json`. |
| `plots.py` | The 4 charts for the linear trend (time-based folds). |

## Key config options (`config.yaml`)

| Key | Meaning |
|---|---|
| `data_path`, `additional_data` | Parquet dataset and soil/climate CSV (merged on `province_col`). |
| `target_col`, `province_col` | Yield column and province column. |
| `group_col` | CV folds keep each group together. `province_name` = unseen-province CV. |
| `keep_group_in_features` | `true` keeps `province_name` as a feature. |
| `test_year` | Held-out year; tuning uses earlier years only. |
| `exclude_cols`, `output_dir` | Drop features (e.g. Sentinel-2 columns) for ablations; change `output_dir` per run. |
| `models` | `params` = fixed settings, `param_grid` = grid searched. |

Outputs go to `output_dir` (leaderboard CSV, per-model JSON, per-province errors, `charts/`).

## Notes

- Preprocessing (target encoding, imputation) is fitted inside each fold, so there is no leakage.
- CV by province measures unseen-province performance (R² about 0.4). The test year measures forecasting for known provinces (R² about 0.85). They answer different questions.
- Scores come from one test year (about 136 rows). Differences under about 0.02 R² are noise.
- Features use the full season, so results are end-of-season estimates, not early forecasts.

