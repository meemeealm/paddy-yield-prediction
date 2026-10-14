import yaml
import importlib
import json
import os
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create tracking directory
os.makedirs("experiments_output", exist_ok=True)

# 1. Load Datasets
df = pd.read_parquet(config["data_path"])

# Load and merge additional static features data (soil/climate metrics) using province_name as key
if "additional_data" in config and config["additional_data"]:
    print(f"Merging additional data from: {config['additional_data']}")
    additional_df = pd.read_csv(config["additional_data"])
    
    merge_col = "province_name"
    
    # Drop any overlapping columns from additional_df to prevent duplicate suffix issues (_x, _y)
    overlap_cols = [col for col in additional_df.columns if col in df.columns and col != merge_col]
    additional_df = additional_df.drop(columns=overlap_cols)
    
    df = pd.merge(df, additional_df, on=merge_col, how="left")

# Clean Dataset Target Range
df = df[(df[config["target_col"]] >= 1.0) & (df[config["target_col"]] <= 9.5)].copy()

target_col = config["target_col"]
group_config = config["group_col"]

# --- DYNAMIC MULTI-COLUMN GROUPING logic ---
if isinstance(group_config, list):
    print(f"Grouping cross-validation folds by composite key: {group_config}")
    groups = df[group_config].astype(str).agg('_'.join, axis=1)
else:
    print(f"Grouping cross-validation folds by single key: {group_config}")
    groups = df[group_config]
# ------------------------------------------

# Exclude target, non-feature columns, and text identifiers like province_name from features X
cols_to_exclude = [target_col, 'production_tons', 'planted_area_ha', 'year', 'province_name']

X_raw = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])
y = df[target_col]

# One-hot encode low-cardinality categorical features globally (macro_region, season)
# Keep 'soil' raw so it can be target-encoded safely inside each CV fold
categorical_cols = ['macro_region', 'season']
X = pd.get_dummies(X_raw, columns=[col for col in categorical_cols if col in X_raw.columns], drop_first=True)

# Remove group columns if they exist in feature matrix X
if isinstance(group_config, list):
    X = X.drop(columns=[col for col in group_config if col in X.columns])
elif group_config in X.columns:
    X = X.drop(columns=[group_config])

# Fill missing values safely on numeric columns only
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Helper function to dynamically load model classes
def get_model_instance(class_path, params):
    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**params)

# 2. Run Tuning Matrix with GroupKFold
gkf = GroupKFold(n_splits=5)
leaderboard = []
model_oof_predictions = {}  # Store OOF predictions per model

for model_name, model_config in config["models"].items():
    print(f"\n--- Tuning & Evaluating Model: {model_name} ---")
    
    class_path = model_config["class"]
    param_grid = model_config["param_grid"]
    
    # Generate all hyperparameter combinations from grid
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    best_score = -float("inf")
    best_params = None
    best_oof_preds = None
    best_fold_r2s = None
    
    # Grid Search across strict spatial/temporal folds
    for params in param_combinations:
        oof_preds = np.zeros(len(df))
        fold_r2s = []
        
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # --- FOLD-LEVEL TARGET ENCODING FOR SOIL ---
            if 'soil' in X_train.columns:
                soil_means = y_train.groupby(X_train['soil']).mean()
                global_mean = y_train.mean()
                
                # Map means to train and test folds securely
                X_train['soil_encoded'] = X_train['soil'].map(soil_means).fillna(global_mean)
                X_test['soil_encoded'] = X_test['soil'].map(soil_means).fillna(global_mean)
                
                X_train = X_train.drop(columns=['soil'])
                X_test = X_test.drop(columns=['soil'])
            # -------------------------------------------

            # Apply feature scaling specifically for linear models (like Ridge) per fold
            if "linear_model" in class_path:
                scaler = StandardScaler()
                X_train_processed = scaler.fit_transform(X_train)
                X_test_processed = scaler.transform(X_test)
            else:
                X_train_processed = X_train
                X_test_processed = X_test
            
            model = get_model_instance(class_path, params)
            model.fit(X_train_processed, y_train)
            
            preds = model.predict(X_test_processed)
            oof_preds[test_idx] = preds
            fold_r2s.append(r2_score(y_test, preds))
            
        mean_f_r2 = np.mean(fold_r2s)
        if mean_f_r2 > best_score:
            best_score = mean_f_r2
            best_params = params
            best_oof_preds = oof_preds
            best_fold_r2s = fold_r2s

    print(f"  -> Best Params for {model_name}: {best_params}")
    print(f"  -> Best Mean Fold R²: {np.mean(best_fold_r2s):.4f}")
    
    # Save this model's best OOF predictions
    model_oof_predictions[model_name] = best_oof_preds

    # Compute final metrics for winning parameter set
    overall_r2 = r2_score(y, best_oof_preds)
    overall_mae = mean_absolute_error(y, best_oof_preds)
    overall_rmse = np.sqrt(mean_squared_error(y, best_oof_preds))
    bias = np.mean(best_oof_preds - y)
    
    # Save individual run artifact JSON
    run_data = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_parameters": best_params,
        "metrics": {
            "r2": float(overall_r2),
            "mean_fold_r2": float(np.mean(best_fold_r2s)),
            "std_fold_r2": float(np.std(best_fold_r2s)),
            "rmse": float(overall_rmse),
            "mae": float(overall_mae),
            "bias": float(bias)
        }
    }
    
    with open(f"experiments_output/{model_name}_tuned_run.json", "w") as f:
        json.dump(run_data, f, indent=4)
        
    leaderboard.append({
        "model": model_name,
        "r2": overall_r2,
        "mean_fold_r2": np.mean(best_fold_r2s),
        "std_fold_r2": np.std(best_fold_r2s),
        "rmse": overall_rmse,
        "mae": overall_mae,
        "bias": bias
    })

# Save Final Leaderboard Summary CSV
lb_df = pd.DataFrame(leaderboard).sort_values(by="r2", ascending=False)
lb_df.to_csv("experiments_output/leaderboard_summary_tuned.csv", index=False)

print("\n=== Spatial-Temporal Hyperparameter Tuning Complete! ===")
print(lb_df.to_string(index=False))
print("\nResults saved to 'experiments_output/leaderboard_summary_tuned.csv'")

# --- Per-Province Error Breakdown Report (Using Overall Best Model) ---
best_model_name = lb_df.iloc[0]['model']
print(f"\nGenerating Per-Province Error Breakdown for Best Model: {best_model_name}")

print("\n--- Top 5 Most Challenging Provinces (Highest MAE) ---")
print(province_error_summary.head(5).to_string(index=False))