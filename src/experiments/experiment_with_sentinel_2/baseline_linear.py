import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reference RMSE/R2 on the same test rows (from your previous runs; edit if they change)
REFERENCE = {
    "Rule: linear trend": (0.8962, 0.3847),
    "Rule: previous year": (0.8830, 0.4084),
    "Rule: province+season mean": (0.8717, 0.4278),
    "XGBoost benchmark": (0.8548, 0.4550),
}
ALPHAS = [0.1, 1, 10, 100, 1000]

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_col = config["target_col"]
province_col = config.get("province_col", "province_name")
test_year = config.get("test_year", 2025)
extra_exclude = config.get("exclude_cols") or []


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ============================================================
# 1. LOAD + CLEAN (same steps as run_experiments.py)
# ============================================================
df = pd.read_parquet(config["data_path"])
df[province_col] = df[province_col].astype(str).str.strip()

if config.get("additional_data"):
    add = pd.read_csv(config["additional_data"])
    add[province_col] = add[province_col].astype(str).str.strip()
    add = add.drop(columns=[c for c in add.columns if c in df.columns and c != province_col])
    df = pd.merge(df, add, on=province_col, how="left", validate="many_to_one")

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df[target_col].between(1.0, 9.5)].copy()
df["year"] = df["year"].astype(int)
df = df.reset_index(drop=True)

# ============================================================
# 2. FEATURE GROUPS
# ============================================================
cat_cols = [c for c in [province_col, "season", "macro_region", "soil"] if c in df.columns]
skip = [target_col, "production_tons", "planted_area_ha", "year", "composite_key"] + extra_exclude + cat_cols
numeric_feats = [c for c in df.columns if c not in skip]

for c in cat_cols:
    ok = df[c].notna()
    df[c] = df[c].astype(str).where(ok, np.nan).astype(object)

for c in numeric_feats:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
numeric_feats = [c for c in numeric_feats if not df[c].isna().all()]

dev = df[df["year"] < test_year]
test = df[df["year"] == test_year]
y_dev, y_test = dev[target_col], test[target_col]
print(f"Dev rows: {len(dev)} | Test rows ({test_year}): {len(test)} | extra numeric features: {len(numeric_feats)}")

# Validation years for choosing alpha: last 3 dev years, each predicted from earlier years only
val_years = [int(y) for y in sorted(dev["year"].unique())[-3:]]


def make_pipe(num_cols, alpha):
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
    ], sparse_threshold=0)
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


variants = {
    "Ridge: province+season+year": ["year"],
    "Ridge: + all numeric features": ["year"] + numeric_feats,
}

# ============================================================
# 3. TIME-BASED ALPHA SELECTION, THEN ONE TEST EVALUATION
# ============================================================
rows = {}
for name, num_cols in variants.items():
    cols = cat_cols + num_cols
    best_alpha, best_cv = None, np.inf

    for alpha in ALPHAS:
        errs = []
        for vy in val_years:
            tr, va = dev[dev["year"] < vy], dev[dev["year"] == vy]
            pipe = make_pipe(num_cols, alpha).fit(tr[cols], tr[target_col])
            errs.append(rmse(va[target_col], pipe.predict(va[cols])))
        if np.mean(errs) < best_cv:
            best_cv, best_alpha = float(np.mean(errs)), alpha

    final = make_pipe(num_cols, best_alpha).fit(dev[cols], y_dev)
    pred = final.predict(test[cols])
    rows[name] = {
        "alpha": best_alpha,
        "time_CV_RMSE": best_cv,
        "R2": r2_score(y_test, pred),
        "RMSE": rmse(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "Bias": float(np.mean(pred - y_test)),
    }

    if num_cols == ["year"]:       # yearly trend in t/ha per year for the simplest model
        pre = final.named_steps["pre"]
        n_cat = len(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out())
        slope = final.named_steps["ridge"].coef_[n_cat] / pre.named_transformers_["num"].named_steps["scale"].scale_[0]
        print(f"Estimated yearly trend ({name}): {slope:+.3f} t/ha per year")

result = pd.DataFrame(rows).T
for name, (r2v, rmsev) in REFERENCE.items():
    result.loc[name, ["R2", "RMSE"]] = [r2v, rmsev]

print(f"\n=== Linear models vs rules vs XGBoost on {test_year} (alpha chosen by time-based CV on {val_years}) ===")
print(result.round(4).to_string())