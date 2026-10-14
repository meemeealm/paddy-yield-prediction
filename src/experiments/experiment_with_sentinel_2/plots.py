import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score

N_FOLD_YEARS = 5   # rolling-origin folds: each of the last N years is predicted from earlier years only

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. CONFIG
# ============================================================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_col = config["target_col"]
province_col = config.get("province_col", "province_name")
test_year = config.get("test_year", 2025)
output_dir = config.get("output_dir", "experiments_output")
keys = [province_col, "season"]

model_dir = os.path.join(output_dir, "charts", "linear_trend")
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# 1. LOAD + CLEAN (same steps as the other scripts)
# ============================================================
df = pd.read_parquet(config["data_path"])
df[province_col] = df[province_col].astype(str).str.strip()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df[target_col].between(1.0, 9.5)].copy()
df["year"] = df["year"].astype(int)
df = df.reset_index(drop=True)


# ============================================================
# 2. LINEAR TREND MODEL (one line per province + season)
# ============================================================
def fit_trend(train_df):
    groups = {}
    for key, g in train_df.groupby(keys):
        years, yields = g["year"].values, g[target_col].values
        if len(np.unique(years)) >= 3:
            slope, intercept = np.polyfit(years, yields, 1)
            groups[key] = ("trend", slope, intercept)
        else:
            groups[key] = ("mean", float(yields.mean()), None)
    return {
        "groups": groups,
        "province_mean": train_df.groupby(province_col)[target_col].mean(),
        "global_mean": train_df[target_col].mean(),
    }


def predict_trend(model, frame):
    out = []
    for p, s, yr in zip(frame[province_col], frame["season"], frame["year"]):
        entry = model["groups"].get((p, s))
        if entry is None:                                   # unseen province+season
            v = model["province_mean"].get(p, model["global_mean"])
        elif entry[0] == "trend":
            v = entry[1] * yr + entry[2]
        else:
            v = entry[1]
        out.append(v)
    return np.clip(np.array(out), 1.0, 9.5)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def save_chart(fig, filename):
    fig.tight_layout()
    path = os.path.join(model_dir, filename)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()


# ============================================================
# 3. ROLLING-ORIGIN FOLDS: predict year T from years < T
#    (the model has no tuned parameters, so using test_year as a fold leaks nothing)
# ============================================================
all_years = sorted(df["year"].unique())
first_year = all_years[0]
fold_years = [int(y) for y in all_years if y - first_year >= 3][-N_FOLD_YEARS:]
print(f"Folds (validation years): {fold_years}")

train_r2, val_r2 = [], []
for T in fold_years:
    tr, va = df[df["year"] < T], df[df["year"] == T]
    m = fit_trend(tr)
    train_r2.append(r2_score(tr[target_col], predict_trend(m, tr)))
    val_r2.append(r2_score(va[target_col], predict_trend(m, va)))

# ============================================================
# 4. LEARNING CURVE: how many years of history are needed?
#    For each fold year T, use only the k years right before T.
# ============================================================
curve = {}
for T in fold_years:
    va = df[df["year"] == T]
    for k in range(3, T - first_year + 1):
        tr = df[(df["year"] >= T - k) & (df["year"] < T)]
        if tr.empty:
            continue
        m = fit_trend(tr)
        curve.setdefault(k, []).append((
            r2_score(tr[target_col], predict_trend(m, tr)),
            r2_score(va[target_col], predict_trend(m, va)),
        ))

# ============================================================
# 5. HELD-OUT TEST YEAR (fit on all earlier years)
# ============================================================
train_all, test = df[df["year"] < test_year], df[df["year"] == test_year]
final = fit_trend(train_all)
pred = predict_trend(final, test)
y_test = test[target_col].values
resid = y_test - pred
test_r2, test_rmse = r2_score(y_test, pred), rmse(y_test, pred)
print(f"Test {test_year}: R2={test_r2:.4f} | RMSE={test_rmse:.4f} | bias={-resid.mean():+.4f}")

# ---------- Chart 1: train vs validation R2 per fold ----------
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(fold_years))
ax.bar(x - 0.2, train_r2, 0.4, color="#3b82f6", alpha=0.85, label=f"Train (mean {np.mean(train_r2):.3f})")
ax.bar(x + 0.2, val_r2, 0.4, color="#f97316", alpha=0.85, label=f"Validation (mean {np.mean(val_r2):.3f})")
ax.set_xticks(x)
ax.set_xticklabels(fold_years)
ax.set_xlabel("Validation year (trained on all earlier years)")
ax.set_ylabel("R²")
ax.set_title("Linear Trend: Train vs Validation R² per Fold", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle=":", alpha=0.6)
save_chart(fig, "1_train_vs_val_r2.png")

# ---------- Chart 2: residuals vs predicted ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pred, resid, color="#8b5cf6", alpha=0.6, edgecolors="k", linewidths=0.4)
ax.axhline(0, color="#14b8a6", linestyle="--", linewidth=2)
ax.set_xlabel("Predicted yield")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title(f"Linear Trend: Residuals ({test_year}) | mean = {resid.mean():+.3f}", fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
save_chart(fig, "2_residuals.png")

# ---------- Chart 3: learning curve (years of history) ----------
ks = sorted(curve)
fig, ax = plt.subplots(figsize=(7, 5))
for idx, color, label in [(0, "#3b82f6", "Train R²"), (1, "#f97316", "Validation R²")]:
    mean = np.array([np.mean([v[idx] for v in curve[k]]) for k in ks])
    std = np.array([np.std([v[idx] for v in curve[k]]) for k in ks])
    ax.plot(ks, mean, marker="o", color=color, label=label)
    ax.fill_between(ks, mean - std, mean + std, color=color, alpha=0.15)
ax.set_xlabel("Years of history used for the trend")
ax.set_ylabel("R²")
ax.set_ylim(bottom=max(ax.get_ylim()[0], -1))
ax.set_title("Linear Trend: Learning Curve (rolling folds)", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, linestyle=":", alpha=0.6)
save_chart(fig, "3_learning_curve.png")

# ---------- Chart 4: predicted vs actual ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, pred, color="#8b5cf6", alpha=0.6, edgecolors="k", linewidths=0.4)
lims = [min(y_test.min(), pred.min()) - 0.2, max(y_test.max(), pred.max()) + 0.2]
ax.plot(lims, lims, color="#14b8a6", linestyle="--", linewidth=2, label="Ideal (y = x)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual yield")
ax.set_ylabel("Predicted yield")
ax.set_title(f"Linear Trend: Predicted vs Actual ({test_year})\nR² = {test_r2:.3f}, RMSE = {test_rmse:.3f}", fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)
save_chart(fig, "4_predicted_vs_actual.png")