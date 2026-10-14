import json
import numpy as np
import pandas as pd

def fit_trend_model(df, target_col, province_col="province_name", season_col="season", year_col="year"):
    """Fit one linear trend per province+season on ALL available years."""
    model = {
        "target_col": target_col,
        "last_train_year": int(df[year_col].max()),
        "global_mean": float(df[target_col].mean()),
        "province_mean": df.groupby(province_col)[target_col].mean().to_dict(),
        "trends": {},
    }
    for (prov, season), g in df.groupby([province_col, season_col]):
        years, yields = g[year_col].values, g[target_col].values
        if len(np.unique(years)) >= 3:
            slope, intercept = np.polyfit(years, yields, 1)
            entry = {"type": "trend", "slope": float(slope), "intercept": float(intercept)}
        else:
            entry = {"type": "mean", "mean": float(yields.mean())}
        entry["n_years"] = int(len(np.unique(years)))
        model["trends"][f"{prov}|{season}"] = entry
    return model

def predict_trend(model, province, season, year):
    entry = model["trends"].get(f"{province}|{season}")
    if entry is None:                                   # unseen province+season
        pred = model["province_mean"].get(province, model["global_mean"])
    elif entry["type"] == "trend":
        pred = entry["slope"] * year + entry["intercept"]
    else:
        pred = entry["mean"]
    return float(np.clip(pred, 1.0, 9.5))

# ---- save ----
model = fit_trend_model(df, target_col)        # df = your cleaned dataframe, all years
with open("linear_trend_model.json", "w") as f:
    json.dump(model, f, indent=2)

# ---- load + predict ----
with open("linear_trend_model.json") as f:
    model = json.load(f)
print(predict_trend(model, "Can Tho", "spring", 2026))