import importlib
import itertools
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. CONFIG
# ============================================================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config.get("output_dir", "experiments_output")
os.makedirs(output_dir, exist_ok=True)

target_col = config["target_col"]
group_config = config["group_col"]
province_col = config.get("province_col", "province_name")
test_year = config.get("test_year", 2025)
n_folds = config.get("cv_folds", 5)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def report_nulls(frame, label, top=25):
    """Print null / inf counts, null share per year and all-null columns."""
    n_rows = len(frame)
    print(f"\n--- Null check: {label} ({n_rows} rows, {frame.shape[1]} columns) ---")

    null_counts = frame.isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    if null_counts.empty:
        print("No nulls found.")
    else:
        report = pd.DataFrame({
            "null_count": null_counts,
            "null_pct": (null_counts / n_rows * 100).round(2),
        })
        print(report.head(top).to_string())
        if len(report) > top:
            print(f"... and {len(report) - top} more columns with nulls")

        rows_with_null = int(frame.isna().any(axis=1).sum())
        print(f"Rows with at least one null: {rows_with_null} ({rows_with_null / n_rows * 100:.1f}%)")

        all_null = null_counts[null_counts == n_rows].index.tolist()
        if all_null:
            print("Entirely null columns:", all_null)

        heavy = null_counts[null_counts / n_rows > 0.5].index.tolist()
        if heavy:
            print("WARNING - more than 50% null:", heavy)

        if "year" in frame.columns:
            by_year = frame.isna().any(axis=1).groupby(frame["year"]).mean().mul(100).round(1)
            print("Share of rows with any null, by year (%):")
            print(by_year.to_string())

    # Infinite values are not caught by isna()
    num = frame.select_dtypes(include=[np.number])
    inf_counts = np.isinf(num).sum()
    inf_counts = inf_counts[inf_counts > 0]
    if not inf_counts.empty:
        print("Columns with inf values (treated as missing later):")
        print(inf_counts.to_string())


# ============================================================
# 1. LOAD DATA + NULL CHECK
# ============================================================
df = pd.read_parquet(config["data_path"])
report_nulls(df, "parquet file")

dup_keys = [c for c in [province_col, "year", "season"] if c in df.columns]
if len(dup_keys) == 3:
    n_dup = int(df.duplicated(subset=dup_keys).sum())
    print(f"\nDuplicate {dup_keys} rows: {n_dup}")

# ============================================================
# 2. MERGE ADDITIONAL STATIC DATA (soil / climate)
# ============================================================
if config.get("additional_data"):
    print(f"\nMerging additional data from: {config['additional_data']}")
    additional_df = pd.read_csv(config["additional_data"])
    report_nulls(additional_df, "additional data csv")

    # Same dtype / no stray spaces on the merge key on both sides
    df[province_col] = df[province_col].astype(str).str.strip()
    additional_df[province_col] = additional_df[province_col].astype(str).str.strip()

    overlap_cols = [c for c in additional_df.columns if c in df.columns and c != province_col]
    additional_df = additional_df.drop(columns=overlap_cols)

    df = pd.merge(df, additional_df, on=province_col, how="left", validate="many_to_one")

    added_cols = [c for c in additional_df.columns if c != province_col]
    unmatched = df.loc[df[added_cols].isna().all(axis=1), province_col].unique()
    if len(unmatched) > 0:
        print(f"WARNING - {len(unmatched)} provinces have no match in additional data: {list(unmatched)}")
    else:
        print("All provinces matched in additional data.")

    report_nulls(df, "after merge")

# ============================================================
# 3. CLEAN TARGET AND YEAR (force numeric dtypes)
# ============================================================
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

n_before = len(df)
# between() is False for NaN, so unparseable / null targets are dropped here too
df = df[df[target_col].between(1.0, 9.5)].copy()
df["year"] = df["year"].astype(int)
df = df.reset_index(drop=True)
print(f"\nTarget filter (1.0-9.5): kept {len(df)} of {n_before} rows")

# ============================================================
# 4. FEATURES, TARGET AND GROUPS
# ============================================================
cols_to_exclude = [target_col, "production_tons", "planted_area_ha", "year", "composite_key"]

# Optional extra columns to leave out (e.g. Sentinel-2 columns for an ablation run)
extra_exclude = config.get("exclude_cols") or []
missing_excl = [c for c in extra_exclude if c not in df.columns]
if missing_excl:
    print("WARNING - exclude_cols not found in data:", missing_excl)
if extra_exclude:
    print("Excluding feature columns:", extra_exclude)
cols_to_exclude = cols_to_exclude + extra_exclude

group_cols = [group_config] if isinstance(group_config, str) else list(group_config)
groups = df[group_cols].astype(str).agg("_".join, axis=1)

X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns]).copy()
y = df[target_col].astype("float64")

# Group columns are removed from the features unless the config says to keep them
# (e.g. keep province_name as a target-encoded feature while still grouping folds by province)
keep_group_in_features = config.get("keep_group_in_features", False)
if keep_group_in_features:
    print("Group column(s) used for CV and KEPT as features:", group_cols)
else:
    X = X.drop(columns=[c for c in group_cols if c in X.columns])
    print("Group column(s) used for CV and removed from features:", group_cols)

all_missing = X.columns[X.isna().all()].tolist()
if all_missing:
    print("Dropping entirely missing features:", all_missing)
    X = X.drop(columns=all_missing)

# ============================================================
# 5. FEATURE GROUPS + DTYPE CLEANING (same as the notebook)
# ============================================================
target_encoded_cols = [c for c in config.get("target_encoded_cols", ["soil", "province_name"]) if c in X.columns]
onehot_cols = [c for c in config.get("onehot_cols", ["season", "macro_region"]) if c in X.columns]
numeric_cols = [c for c in X.columns if c not in target_encoded_cols + onehot_cols]

# Categorical: clean strings, real NaN stays NaN (imputer fills "Missing")
for col in target_encoded_cols + onehot_cols:
    is_valid = X[col].notna()
    X[col] = X[col].astype(str).where(is_valid, np.nan).astype(object)

# Numeric: force float64, inf -> NaN
for col in numeric_cols:
    before_nan = X[col].isna().sum()
    X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    new_nan = X[col].isna().sum() - before_nan
    if new_nan > 0:
        print(f"Warning: '{col}' had {new_nan} non-numeric values converted to NaN")

bad_numeric = [c for c in numeric_cols if X[c].isna().all()]
if bad_numeric:
    print("Dropping numeric features that are entirely NaN:", bad_numeric)
    X = X.drop(columns=bad_numeric)
    numeric_cols = [c for c in numeric_cols if c not in bad_numeric]

print("\nFeature dtypes after cleaning:")
print(X.dtypes.value_counts().to_string())
print(f"Target-encoded: {target_encoded_cols} | One-hot: {onehot_cols} | Numeric: {len(numeric_cols)}")

# ============================================================
# 6. PREPROCESSOR (fit inside every fold -> no leakage)
# ============================================================
transformers = []

if target_encoded_cols:
    transformers.append((
        "target_encoded",
        Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42)),
        ]),
        target_encoded_cols,
    ))

if onehot_cols:
    transformers.append((
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        onehot_cols,
    ))

if numeric_cols:
    transformers.append((
        "numeric",
        Pipeline([("imputer", SimpleImputer(strategy="median"))]),
        numeric_cols,
    ))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def get_model_instance(class_path, params):
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**params)


def build_pipeline(class_path, params):
    steps = [("preprocessor", clone(preprocessor))]
    if "linear_model" in class_path:          # scaling only for linear models (e.g. Ridge)
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", get_model_instance(class_path, params)))
    return Pipeline(steps)


# ============================================================
# 7. HELD-OUT TEST SPLIT (same idea as the notebook)
# ============================================================
dev_mask = df["year"] < test_year
test_mask = df["year"] == test_year

if not dev_mask.any() or not test_mask.any():
    raise ValueError(f"Empty split. Years available: {sorted(df['year'].unique())}, test_year={test_year}")

X_dev, y_dev, groups_dev = X.loc[dev_mask], y.loc[dev_mask], groups.loc[dev_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

print(f"\nDev rows (years < {test_year}): {len(X_dev)} | Test rows ({test_year}): {len(X_test)}")

if groups_dev.nunique() < n_folds:
    raise ValueError(f"Only {groups_dev.nunique()} groups in dev data, need at least {n_folds} for GroupKFold")

# ============================================================
# 8. TUNING WITH GROUPED CV ON DEV YEARS, THEN ONE TEST EVALUATION
# ============================================================
gkf = GroupKFold(n_splits=n_folds)
leaderboard = []
model_oof_predictions = {}

for model_name, model_config in config["models"].items():
    print(f"\n--- Tuning & Evaluating Model: {model_name} ---")

    class_path = model_config["class"]

    if "param_grid" in model_config:
        param_grid = {k: (v if isinstance(v, list) else [v]) for k, v in model_config["param_grid"].items()}
    elif "params" in model_config:                      # single fixed parameter set
        param_grid = {k: [v] for k, v in model_config["params"].items()}
    else:
        raise KeyError(f"Model '{model_name}' needs either 'params' or 'param_grid' in config.yaml")

    keys = list(param_grid.keys())
    param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]
    print(f"  {len(param_combinations)} parameter combination(s) x {n_folds} folds")

    best_rmse = float("inf")
    best_params = None
    best_oof_preds = None
    best_fold_r2s = None

    for i, params in enumerate(param_combinations, start=1):
        oof_preds = np.zeros(len(X_dev))
        fold_r2s, fold_rmses = [], []

        for train_idx, val_idx in gkf.split(X_dev, y_dev, groups=groups_dev):
            pipe = build_pipeline(class_path, params)
            pipe.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])

            preds = pipe.predict(X_dev.iloc[val_idx])
            oof_preds[val_idx] = preds
            fold_r2s.append(r2_score(y_dev.iloc[val_idx], preds))
            fold_rmses.append(rmse(y_dev.iloc[val_idx], preds))

        mean_rmse = float(np.mean(fold_rmses))
        if i % 10 == 0 or i == len(param_combinations):
            print(f"  [{i}/{len(param_combinations)}] last CV RMSE={mean_rmse:.4f} | best so far={min(best_rmse, mean_rmse):.4f}")

        # Selection by mean fold RMSE (same criterion as the notebook's GridSearchCV)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            best_oof_preds = oof_preds
            best_fold_r2s = fold_r2s

    print(f"  -> Best Params for {model_name}: {best_params}")
    print(f"  -> Best CV RMSE: {best_rmse:.4f} | Mean Fold R2: {np.mean(best_fold_r2s):.4f}")

    model_oof_predictions[model_name] = best_oof_preds

    # CV metrics (out-of-fold predictions on dev years)
    cv_r2 = r2_score(y_dev, best_oof_preds)
    cv_mae = mean_absolute_error(y_dev, best_oof_preds)
    cv_rmse = rmse(y_dev, best_oof_preds)
    cv_bias = float(np.mean(best_oof_preds - y_dev))

    # Refit on all dev years and score the held-out test year (used once, not for selection)
    final_pipe = build_pipeline(class_path, best_params)
    final_pipe.fit(X_dev, y_dev)
    dev_pred = final_pipe.predict(X_dev)
    test_pred = final_pipe.predict(X_test)

    test_r2 = r2_score(y_test, test_pred)
    test_rmse = rmse(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_bias = float(np.mean(test_pred - y_test))
    train_r2 = r2_score(y_dev, dev_pred)

    print(f"  -> Held-out {test_year}: R2={test_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | bias={test_bias:.4f}")

    run_data = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_parameters": best_params,
        "metrics": {
            "r2": float(cv_r2),
            "mean_fold_r2": float(np.mean(best_fold_r2s)),
            "std_fold_r2": float(np.std(best_fold_r2s)),
            "rmse": float(cv_rmse),
            "mae": float(cv_mae),
            "bias": cv_bias,
        },
        "test_metrics": {
            "test_year": int(test_year),
            "r2": float(test_r2),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "bias": test_bias,
            "train_r2": float(train_r2),
        },
    }

    with open(f"{output_dir}/{model_name}_tuned_run.json", "w") as f:
        json.dump(run_data, f, indent=4)

    leaderboard.append({
        "model": model_name,
        "cv_r2": cv_r2,
        "mean_fold_r2": np.mean(best_fold_r2s),
        "std_fold_r2": np.std(best_fold_r2s),
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "cv_bias": cv_bias,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_bias": test_bias,
    })

# ============================================================
# 9. LEADERBOARD (ranked by CV only; test columns are for reporting)
# ============================================================
lb_df = pd.DataFrame(leaderboard).sort_values(by="cv_rmse", ascending=True)
lb_df.to_csv(f"{output_dir}/leaderboard_summary_tuned.csv", index=False)

print("\n=== Spatial Hyperparameter Tuning Complete! ===")
print(lb_df.round(4).to_string(index=False))
print(f"\nResults saved to '{output_dir}/leaderboard_summary_tuned.csv'")

# ============================================================
# 10. PER-PROVINCE ERROR BREAKDOWN (best model by CV, out-of-fold dev predictions)
# ============================================================
best_model_name = lb_df.iloc[0]["model"]
print(f"\nGenerating Per-Province Error Breakdown for Best Model: {best_model_name}")

err = pd.DataFrame({
    "province": df.loc[dev_mask, province_col].values,
    "actual": y_dev.values,
    "pred": model_oof_predictions[best_model_name],
})
err["error"] = err["pred"] - err["actual"]
err["abs_error"] = err["error"].abs()
err["sq_error"] = err["error"] ** 2

province_error_summary = (
    err.groupby("province")
    .agg(n=("error", "size"), MAE=("abs_error", "mean"), RMSE=("sq_error", lambda s: np.sqrt(s.mean())), bias=("error", "mean"))
    .reset_index()
    .sort_values("MAE", ascending=False)
)
province_error_summary.to_csv(f"{output_dir}/province_errors_{best_model_name}.csv", index=False)

print("\n--- Top 5 Most Challenging Provinces (Highest MAE) ---")
print(province_error_summary.head(5).round(4).to_string(index=False))