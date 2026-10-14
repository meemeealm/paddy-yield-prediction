import json
import numpy as np
import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

target_col = config["target_col"]
province_col = config.get("province_col", "province_name")

# Same row cleaning as the other scripts
df = pd.read_parquet(config["data_path"])
df[province_col] = df[province_col].astype(str).str.strip()
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df[target_col].between(1.0, 9.5)].copy()
df["year"] = df["year"].astype(int)


def fit_trend_model(df, target_col, province_col, season_col="season", year_col="year"):
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
model = fit_trend_model(df, target_col, province_col)
with open("linear_trend_model.json", "w") as f:
    json.dump(model, f, indent=2)
print(f"Saved {len(model['trends'])} province+season trends (last year: {model['last_train_year']})")

# ---- load + predict ----
with open("linear_trend_model.json") as f:
    model = json.load(f)

example_key = next(iter(model["trends"]))
example_province, example_season = example_key.split("|")
print(example_key, "->", predict_trend(model, example_province, example_season, model["last_train_year"] + 1))