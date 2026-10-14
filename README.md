# Rice Yield Prediction for Vietnam

End-to-end data processing and model validation pipeline for predicting seasonal rice yield (t/ha) across Vietnam's provinces.

- **Data:** 2015–2025, about 63 provinces × 3 seasons (Spring, Autumn, Winter), roughly 1,600 province-season-year records.
- **Features (~30):** Sentinel-1 SAR backscatter statistics, ERA5 climate (temperature, precipitation) and static data (soil, elevation, slope, coordinates).
- **Targets in the source data:** Planted Area (ha), Production (tons) and **Yield (t/ha), the primary target**.

Details: see the data ingestion and processing README.   

Scripts and usage: see the experiments README.

---

## Main Finding

For forecasting **known provinces in a future year**, a simple **linear trend per province and season** beats every machine-learning model tested, including tuned XGBoost.

| Model (held-out 2025) | RMSE (t/ha) | R² |
| :--- | :---: | :---: |
| **Linear trend (per province + season)** | **0.385** | **0.896** |
| Previous year's yield | 0.408 | 0.883 |
| Trend + XGBoost on residuals | 0.419 | 0.877 |
| Province + season mean | 0.428 | 0.872 |
| XGBoost (tuned benchmark) | 0.455 | 0.855 |
| Ridge (province + season + year) | 0.645 | 0.708 |

---

## Model Selection & Validation Strategy

### Models tested
Gradient-boosted trees (XGBoost, CatBoost), random forest, Ridge/linear regression, SVR, plus rule-based baselines (province+season mean, previous year, linear trend). XGBoost was the best ML model, but it did not beat the trend baseline.

### Validation
- **Held-out test year:** train on 2015–2024, test on 2025. All tuning uses earlier years only. This matches the real task of forecasting a future year.
- **Grouped CV:** `GroupKFold` keeps all rows of a group together to prevent leakage. The choice of group changes what is measured:
  - Grouping by **province** holds out whole provinces, so it measures performance on *unseen* provinces (R² ≈ 0.4).
  - Grouping by a coarser key such as **region + year** scored about twice as high (R² ≈ 0.8), because the model still sees the same provinces in other folds. That is an easier task, so the jump does not mean a better model.
- **Preprocessing** (target encoding, imputation) is fitted inside each fold, so there is no leakage.

---

## Model Performance & Diagnostics

| Linear Trend: Predicted vs Actual (2025) | XGBoost: Predicted vs Actual (2025) |
| :---: | :---: |
| ![Trend predicted vs actual](charts/trend_predicted_vs_actual.png) | ![XGBoost predicted vs actual](assets/xgb_predicted_vs_actual.png) |
| *R² 0.896, RMSE 0.385. Points follow the ideal y = x line.* | *R² 0.855, RMSE 0.455. Predictions shrink toward the middle, with low yields over-predicted.* |

| Linear Trend: Residuals | XGBoost: Residuals |
| :---: | :---: |
| ![Trend residuals](assets/trend_residuals.png) | ![XGBoost residuals](assets/xgb_residuals.png) |
| *Mean residual about −0.12 t/ha: slight over-prediction.* | *Mean residual about +0.17 t/ha: under-prediction of 2025.* |

| XGBoost Feature Attribution of XGBoost (SHAP) |
| :---: |
| ![SHAP beeswarm](assets/beeswarm_features.jpg) |
| *Province and season dominate; weather and radar features contribute smaller adjustments.* |

---

## Key Takeaways

- **Simple beats complex model:** the trend rule has no tuned parameters and the lowest error, so it is the recommended model for known provinces.
- **XGBoost overfits and cannot extrapolate:** train R² is about 0.97 against a test R² of 0.85. Tree models cannot continue a rising yield trend, which explains its under-prediction of 2025.
- **Satellite and climate features add little** beyond province, season and trend. Adding Sentinel-2 vegetation indices (NDVI, EVI, LSWI) changed the test result by less than the noise level.
- **Where ML may still help:** predicting unseen provinces (CV R² ≈ 0.4) or forecasting mid-season, neither of which the trend rule can do.

## Limitations

- The test set is a single year (about 136 rows). Differences under about 0.02 R² are noise, and yearly RMSE of the same model ranged from about 0.3 to 0.8.
- Features cover the whole season, so predictions are end-of-season estimates, not early forecasts.
- The trend rule needs at least three years of history per province and season, and extrapolates poorly several years ahead.