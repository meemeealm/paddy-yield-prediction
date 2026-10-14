# Rice Yield Prediction for Vietnamese provinces

This repo demonstrates data processing and model validation pipeline for predicting seasonal rice yield across Vietnam's provinces.

- **Data:** 2015–2025, about 63 provinces × 3 seasons (Spring, Autumn, Winter), roughly 1,600 province-season-year records. [[more]](https://github.com/meemeealm/paddy-yield-prediction/tree/main/data)
- **Features (~30):** Sentinel-1 SAR statistics, Sentinel-2 Vegetation Statistics, ERA5 climate (temperature, precipitation) and static data (soil, elevation, slope, coordinates). [[more]](https://github.com/meemeealm/paddy-yield-prediction/tree/main/data)
- **Primary target:** `Yield (t/ha)`

---

## Main Finding

For forecasting **known provinces in a future year**, a **linear trend per province and season** beats every model tested, including [tuned XGBoost](https://github.com/meemeealm/paddy-yield-prediction/blob/main/notebooks/06_analyze_optimize_xgb.ipynb).

| Model (held-out 2025) | RMSE (t/ha) | R² |
| :--- | :---: | :---: |
| **Linear trend (per province + season)**</a><sup>+</sup> | **0.385** | **0.896** |
| Previous year's yield | 0.408 | 0.883 |
| Province + season mean | 0.428 | 0.872 |
| XGBoost (tuned benchmark v1)* | 0.466 | 0.848 |
| XGBoost (tuned benchmark v2)* | 0.455 | 0.855 |
| Ridge (province + season + year) | 0.645 | 0.708 |

</a><sup>+</sup>*The linear-trend rule uses only four features - province, season, year and past yield.*  
**Province_name was excluded from features in v1, but included in v2.*  


---

## Model Selection & Validation Strategy

### Models tested
Gradient-boosted trees (XGBoost, CatBoost), random forest, Ridge/linear regression, SVR, plus rule-based baselines (province+season mean, previous year, linear trend) were tested. XGBoost was the best ML model, but it did not beat the trend baseline.

### Validation
- **Held-out test year:** train on 2015–2024, test on 2025. All tuning uses earlier years only. 
- **Grouped CV:** `GroupKFold` keeps all rows of a group together to prevent leakage. The choice of group changes what is measured:
  - Grouping by **province** holds out whole provinces, so it measures performance on *unseen* provinces (R² ≈ 0.4).
  - Grouping by a coarser key such as `macro_region` + `year` scored about twice as high (R² ≈ 0.8), because the model still sees the same provinces in other folds.
- **Preprocessing** (target encoding, imputation) is fitted inside each fold, so there is no leakage.

Please see the scripts [here](https://github.com/meemeealm/paddy-yield-prediction/tree/main/src/experiments).

---

## Model Performance & Diagnostics



| Linear Trend: Predicted vs Actual (2025) | XGBoost: Predicted vs Actual (2025) |
| :---: | :---: |
| ![Trend predicted vs actual](https://github.com/meemeealm/paddy-yield-prediction/blob/main/charts/linear_trend/1_train_vs_val_r2.png) | ![XGBoost predicted vs actual](https://github.com/meemeealm/paddy-yield-prediction/blob/main/charts/xgboost/1_train_vs_val_r2.png) |
| *R² 0.896, RMSE 0.385. Points follow the ideal y = x line.* | *R² 0.855, RMSE 0.455. Predictions shrink toward the middle, with low yields over-predicted.* |

| Linear Trend: Predicted vs Actual (2025) | XGBoost: Predicted vs Actual (2025) |
| :---: | :---: |
| ![Trend residuals](https://github.com/meemeealm/paddy-yield-prediction/blob/main/charts/linear_trend/4_predicted_vs_actual.png) | ![XGBoost residuals](https://github.com/meemeealm/paddy-yield-prediction/blob/main/charts/xgboost/4_predicted_vs_actual.png) |
| *Mean residual about −0.12 t/ha: slight over-prediction.* | *Mean residual about +0.17 t/ha: under-prediction of 2025.* |

---

## Key Takeaways

- **Simple beats complex model:** the trend rule has no tuned parameters and the lowest error for known ovinces.
- **XGBoost overfits and cannot extrapolate:** train R² is about 0.97 against a test R² of 0.85. Tree models cannot continue a rising yield trend, which explains its under-prediction of 2025.
- **Satellite and climate features add little** beyond province, season and trend. Adding Sentinel-2 vegetation indices (NDVI, EVI, LSWI) changed the test result by less than the noise level.
- **Where ML may still help:** predicting unseen provinces (CV R² ≈ 0.4) or forecasting mid-season, neither of which the trend rule can do.

## Limitations

- The test set is a single year (about 136 rows). Differences under about 0.02 R² are noise, and yearly RMSE of the same model ranged from about 0.3 to 0.8.
- Features cover the whole season, so predictions are end-of-season estimates, not early forecasts.
- The trend rule needs at least three years of history per province and season, and extrapolates poorly several years ahead.
