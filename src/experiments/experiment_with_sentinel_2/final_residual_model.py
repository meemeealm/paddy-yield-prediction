import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

# ------------------------------------------------------------
# FINAL SETUP: baseline = linear trend per province+season.
# XGBoost predicts only the residual (actual - baseline); prediction = baseline + residual.
# ------------------------------------------------------------
ADD_BASELINE_FEATURE = True      # also give XGBoost the baseline value as an input feature
XGB_REFERENCE = (0.8548, 0.4550)  # plain XGBoost benchmark (R2, RMSE) on the same test rows

CANDIDATES = {
    "benchmark params": dict(learning_rate=0.05, max_depth=7, n_estimators=200, min_child_weight=5,
                             colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1),
    "moderate": dict(learning_rate=0.03, max_depth=4, n_estimators=150, min_child_weight=10,
                     colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1),
    "strong regularization": dict(learning_rate=0.02, max_depth=3, n_estimators=100, min_child_weight=20,
                                  colsample_bytree=0.8, subsample=0.8, reg_lambda=5, random_state=42, n_jobs=-1),
}

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_col = config["target_col"]
province_col = config.get("province_col", "province_name")
test_year = config.get("test_year", 2025)
extra_exclude = config.get("exclude_cols") or []
keys = [province_col, "season"]


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def scores(y_true, pred):
    return {"R2": r2_score(y_true, pred), "RMSE": rmse(y_true, pred),
            "MAE": mean_absolute_error(y_true, pred), "Bias": float(np.mean(pred - y_true))}


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

te_cols = [c for c in config.get("target_encoded_cols", ["soil", "province_name"]) if c in df.columns]
oh_cols = [c for c in config.get("onehot_cols", ["season", "macro_region"]) if c in df.columns]
skip = [target_col, "production_tons", "planted_area_ha", "year", "composite_key"] + extra_exclude + te_cols + oh_cols
num_cols = [c for c in df.columns if c not in skip]

for c in te_cols + oh_cols:
    ok = df[c].notna()
    df[c] = df[c].astype(str).where(ok, np.nan).astype(object)
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
num_cols = [c for c in num_cols if not df[c].isna().all()]

feature_cols = te_cols + oh_cols + num_cols
if ADD_BASELINE_FEATURE:
    num_cols_model = num_cols + ["baseline"]
else:
    num_cols_model = num_cols

dev = df[df["year"] < test_year]
test = df[df["year"] == test_year]
print(f"Dev rows: {len(dev)} | Test rows ({test_year}): {len(test)} | features: {len(feature_cols)}")


# ============================================================
# 2. BASELINE (linear trend per province+season)
# ============================================================
def trend_fit_predict(years, yields, target_year):
    """Linear trend if >= 3 distinct years, otherwise the mean, otherwise NaN."""
    if len(yields) == 0:
        return np.nan
    if len(np.unique(years)) >= 3:
        slope, intercept = np.polyfit(years, yields, 1)
        return slope * target_year + intercept
    return float(np.mean(yields))


def loyo_baseline(train_df):
    """Baseline for each TRAINING row, fitted WITHOUT that row's year.
    Gives honest residuals (an in-sample trend would make residuals too small)."""
    base = pd.Series(np.nan, index=train_df.index)
    for _, g in train_df.groupby(keys):
        for idx, yr in zip(g.index, g["year"]):
            others = g[g["year"] != yr]
            base.loc[idx] = trend_fit_predict(others["year"].values, others[target_col].values, yr)
    return base.clip(1.0, 9.5)


def forward_baseline(train_df, target_df):
    """Baseline for new rows: trend fitted on all training years of the same province+season."""
    hist = {k: (g["year"].values, g[target_col].values) for k, g in train_df.groupby(keys)}
    prov_mean = train_df.groupby(province_col)[target_col].mean()
    gmean = train_df[target_col].mean()
    out = []
    for p, s, yr in zip(target_df[province_col], target_df["season"], target_df["year"]):
        v = trend_fit_predict(*hist[(p, s)], yr) if (p, s) in hist else np.nan
        out.append(prov_mean.get(p, gmean) if np.isnan(v) else v)
    return pd.Series(np.clip(out, 1.0, 9.5), index=target_df.index)


# ============================================================
# 3. RESIDUAL MODEL
# ============================================================
def make_pipe(params):
    steps = []
    if te_cols:
        steps.append(("te", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                                      ("enc", TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42))]),
                      te_cols))
    if oh_cols:
        steps.append(("oh", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                                      ("enc", OneHotEncoder(handle_unknown="ignore"))]), oh_cols))
    steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols_model))
    pre = ColumnTransformer(steps, remainder="drop", sparse_threshold=0)
    return Pipeline([("pre", pre), ("model", xgb.XGBRegressor(**params))])


def build_X(frame, base):
    X = frame[feature_cols].copy()
    X["baseline"] = base
    return X if ADD_BASELINE_FEATURE else X.drop(columns="baseline")


def fit_residual(train_df, base_train, params):
    ok = base_train.notna()
    pipe = make_pipe(params)
    pipe.fit(build_X(train_df[ok], base_train[ok]), train_df.loc[ok, target_col] - base_train[ok])
    return pipe


# ============================================================
# 4. TIME-BASED SELECTION (last 3 dev years, each predicted from earlier years only)
# ============================================================
val_years = [int(y) for y in sorted(dev["year"].unique())[-3:]]
cv = {name: [] for name in CANDIDATES}
cv["Baseline only (trend)"] = []

for vy in val_years:
    tr, va = dev[dev["year"] < vy], dev[dev["year"] == vy]
    base_tr = loyo_baseline(tr)
    base_va = forward_baseline(tr, va)
    cv["Baseline only (trend)"].append(rmse(va[target_col], base_va))
    for name, params in CANDIDATES.items():
        pipe = fit_residual(tr, base_tr, params)
        pred = np.clip(base_va + pipe.predict(build_X(va, base_va)), 1.0, 9.5)
        cv[name].append(rmse(va[target_col], pred))

cv_df = pd.DataFrame(cv, index=val_years).T
cv_df["mean_RMSE"] = cv_df.mean(axis=1)
print(f"\n=== Time-based validation RMSE by year (train on earlier years only) ===")
print(cv_df.round(4).to_string())

best_name = min(CANDIDATES, key=lambda n: cv_df.loc[n, "mean_RMSE"])
if cv_df.loc[best_name, "mean_RMSE"] >= cv_df.loc["Baseline only (trend)", "mean_RMSE"]:
    print("\nNOTE: no residual model beat the baseline in validation.")
print(f"\nSelected candidate: {best_name}")

# ============================================================
# 5. FINAL: FIT ON ALL DEV YEARS, SCORE THE TEST YEAR ONCE
# ============================================================
base_dev = loyo_baseline(dev)
final = fit_residual(dev, base_dev, CANDIDATES[best_name])

base_test = forward_baseline(dev, test)
pred_test = np.clip(base_test + final.predict(build_X(test, base_test)), 1.0, 9.5)

result = pd.DataFrame({
    "Baseline only (trend)": scores(test[target_col], base_test),
    f"Trend + XGBoost residual ({best_name})": scores(test[target_col], pred_test),
}).T
result.loc["XGBoost benchmark (reference)", ["R2", "RMSE"]] = XGB_REFERENCE

print(f"\n=== Final comparison on {test_year} ===")
print(result.round(4).to_string())

gain = (1 - result.iloc[1]["RMSE"] / result.iloc[0]["RMSE"]) * 100
print(f"\nRMSE change vs trend baseline: {gain:+.1f}% (positive = residual model helps)")

# What does the residual model use?
names = final.named_steps["pre"].get_feature_names_out()
imp = pd.Series(final.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 10 features of the residual model:")
print(imp.head(10).round(4).to_string())