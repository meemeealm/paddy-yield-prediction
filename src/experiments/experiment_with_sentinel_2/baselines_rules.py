import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reference scores of XGBoost benchmark on the same 2025 test rows 
# note: these scores are without sentninel-2 data 
XGB_R2, XGB_RMSE = 0.8548, 0.4550

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_col = config["target_col"]
province_col = config.get("province_col", "province_name")
test_year = config.get("test_year", 2025)

# Same row cleaning as run_experiments.py, so the test rows are identical
df = pd.read_parquet(config["data_path"])
df[province_col] = df[province_col].astype(str).str.strip()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df[target_col].between(1.0, 9.5)].copy()
df["year"] = df["year"].astype(int)

train = df[df["year"] < test_year]
test = df[df["year"] == test_year].copy()
keys = [province_col, "season"]
print(f"Train rows: {len(train)} | Test rows ({test_year}): {len(test)}")

# ---- Rule 1: province + season historical mean (fallback: province mean, then global mean)
ps_mean = train.groupby(keys)[target_col].mean().rename("ps_mean")
p_mean = train.groupby(province_col)[target_col].mean().rename("p_mean")
test = test.join(ps_mean, on=keys).join(p_mean, on=province_col)
test["rule_mean"] = test["ps_mean"].fillna(test["p_mean"]).fillna(train[target_col].mean())

# ---- Rule 2: previous year's yield, same province + season (fallback: Rule 1)
prev = (
    train[train["year"] == test_year - 1]
    .set_index(keys)[target_col]
    .rename("prev_year")
)
prev = prev[~prev.index.duplicated(keep="last")]
test = test.join(prev, on=keys)
test["rule_prev"] = test["prev_year"].fillna(test["rule_mean"])

# ---- Rule 3: linear trend per province + season (needs >= 3 years, fallback: Rule 1)
trend_pred = {}
for key, g in train.groupby(keys):
    if g["year"].nunique() >= 3:
        slope, intercept = np.polyfit(g["year"], g[target_col], 1)
        trend_pred[key] = slope * test_year + intercept

test["rule_trend"] = [
    trend_pred.get((p, s), np.nan) for p, s in zip(test[province_col], test["season"])
]
test["rule_trend"] = test["rule_trend"].fillna(test["rule_mean"]).clip(1.0, 9.5)

print("Fallbacks used (no history):",
      f"mean={int(test['ps_mean'].isna().sum())}, prev_year={int(test['prev_year'].isna().sum())}, "
      f"trend={sum((p, s) not in trend_pred for p, s in zip(test[province_col], test['season']))}")

# ---- Compare
rows = {}
for name, col in [("Province+season mean", "rule_mean"),
                  ("Previous year", "rule_prev"),
                  ("Linear trend (per province+season)", "rule_trend")]:
    pred = test[col]
    rows[name] = {
        "R2": r2_score(test[target_col], pred),
        "RMSE": float(np.sqrt(mean_squared_error(test[target_col], pred))),
        "MAE": mean_absolute_error(test[target_col], pred),
        "Bias": float((pred - test[target_col]).mean()),
    }
rows["XGBoost benchmark (reference)"] = {"R2": XGB_R2, "RMSE": XGB_RMSE, "MAE": np.nan, "Bias": np.nan}

result = pd.DataFrame(rows).T
print("\n=== Rule-based baselines vs XGBoost on the held-out test year ===")
print(result.round(4).to_string())

best_rule_rmse = result.iloc[:3]["RMSE"].min()
print(f"\nBest rule RMSE: {best_rule_rmse:.4f} | XGBoost RMSE: {XGB_RMSE:.4f} "
      f"| XGBoost improvement: {(1 - XGB_RMSE / best_rule_rmse) * 100:.1f}%")