import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from province_mappings import PROVINCE_TO_REGION, clean_province_names

logger = logging.getLogger(__name__)

GROUP_COLS = ["province_name", "macro_region", "year", "season"]
TARGET_YEARS = set(range(2005, 2026))

KNOWN_MACRO_REGIONS = [
    "WHOLE COUNTRY",
    "Red River Delta",
    "Northern midlands and mountain areas",
    "North Central and Central coastal areas",
    "Central Highlands",
    "South East",
    "Mekong River Delta",
]

GSO_FILE_MAPPING = {
    "e06.16": ("spring", "planted_area_ha"),
    "e06.17": ("spring", "yield_ton_ha"),
    "e06.18": ("spring", "production_tons"),
    "e06.19": ("autumn", "planted_area_ha"),
    "e06.20": ("autumn", "yield_ton_ha"),
    "e06.21": ("autumn", "production_tons"),
    "e06.22": ("winter", "planted_area_ha"),
    "e06.23": ("winter", "yield_ton_ha"),
    "e06.24": ("winter", "production_tons"),
}

KNOWN_PROVINCES = {
    "Ha Noi", "Ha Tay", "Vinh Phuc", "Bac Ninh", "Quang Ninh", "Hai Duong", "Hai Phong",
    "Hung Yen", "Thai Binh", "Ha Nam", "Nam Dinh", "Ninh Binh", "Ha Giang", "Cao Bang",
    "Bac Kan", "Tuyen Quang", "Lao Cai", "Yen Bai", "Thai  Nguyen", "Thai Nguyen", "Lang Son",
    "Bac Giang", "Phu Tho", "Dien Bien", "Lai Chau", "Son La", "Hoa Binh", "Thanh Hoa",
    "Nghe An", "Ha Tinh", "Quang Binh", "Quang Tri", "Thua Thien-Hue", "Da Nang",
    "Quang  Nam", "Quang Nam", "Quang  Ngai", "Quang Ngai", "Binh Dinh", "Phu Yen",
    "Khanh  Hoa", "Khanh Hoa", "Ninh  Thuan", "Ninh Thuan", "Binh Thuan", "Kon Tum",
    "Gia Lai", "Dak Lak", "Dak Nong", "Lam Dong", "Binh Phuoc", "Tay Ninh", "Binh  Duong",
    "Binh Duong", "Dong Nai", "Ba Ria - Vung Tau", "Ho Chi Minh city", "Long An",
    "Tien Giang", "Ben Tre", "Tra Vinh", "Vinh Long", "Dong Thap", "An Giang",
    "Kien  Giang", "Kien Giang", "Can Tho", "Hau Giang", "Soc Trang", "Bac Lieu", "Ca Mau"
}


def clean_numeric_val(val) -> float:
    """Converts raw cell value to float, treating '..' or empty strings as NaN."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().replace(",", "")
    if val_str in ["..", "", "nan", "None", "-"]:
        return np.nan
    try:
        v = float(val_str)
        return v if v > 0 else np.nan
    except ValueError:
        return np.nan


def extract_season_and_metric(filename: str) -> tuple[str, str]:
    """Extracts crop season and metric type from GSO filename."""
    stem = Path(filename).stem.lower().strip()

    if stem in GSO_FILE_MAPPING:
        return GSO_FILE_MAPPING[stem]

    fname = filename.lower()
    season = "spring" if "spring" in fname else ("autumn" if "autumn" in fname or "summer" in fname else ("winter" if "winter" in fname else "annual"))
    metric = "planted_area_ha" if ("area" in fname or "planted" in fname) else ("yield_ton_ha" if "yield" in fname else ("production_tons" if "production" in fname else stem))

    return season, metric


def impute_missing_agricultural_data(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing target values using Linear Interpolation grouped by Province & Season."""
    target_cols = ["planted_area_ha", "yield_ton_ha", "production_tons"]
    
    # Replace zeros with NaN
    df[target_cols] = df[target_cols].replace(0.0, np.nan)
    
    df = df.sort_values(by=["province_name", "season", "year"]).reset_index(drop=True)
    
    # Linear Interpolation
    for col in target_cols:
        df[col] = df.groupby(["province_name", "season"])[col].transform(
            lambda group: group.interpolate(method="linear", limit_direction="both")
        )
    
    # Forward and Backward fill for remaining edge gaps
    for col in target_cols:
        df[col] = df.groupby(["province_name", "season"])[col].transform(
            lambda group: group.ffill().bfill()
        )
        
    # Mathematical derivation fallback
    zero_area_mask = (df["planted_area_ha"].isna() | (df["planted_area_ha"] == 0)) & (df["yield_ton_ha"] > 0) & (df["production_tons"] > 0)
    df.loc[zero_area_mask, "planted_area_ha"] = (
        df.loc[zero_area_mask, "production_tons"] / df.loc[zero_area_mask, "yield_ton_ha"]
    ).round(1)
    
    return df.fillna(0.0)


def load_and_reshape_gso_files(raw_gso_path: Path) -> pd.DataFrame:
    """Discovers GSO CSV files and combines metrics into long format."""
    files = list(raw_gso_path.glob("*.csv")) if raw_gso_path.is_dir() else [raw_gso_path]

    if not files:
        raise FileNotFoundError(f"No CSV files found at path: {raw_gso_path}")

    all_records = []

    for file_path in files:
        logger.info("Reading raw CSV: %s", file_path)
        df = pd.read_csv(file_path, skiprows=2)

        if df.empty:
            continue

        location_col = df.columns[0]
        season, metric_name = extract_season_and_metric(file_path.name)
        current_region = None

        year_col_map = {}
        for col in df.columns[1:]:
            digits = re.findall(r'\b(20\d{2})\b', str(col))
            if digits:
                yr = int(digits[0])
                if yr in TARGET_YEARS:
                    year_col_map[col] = yr

        for _, row in df.iterrows():
            raw_loc = row[location_col]
            if pd.isna(raw_loc):
                continue

            loc_val = " ".join(str(raw_loc).split())
            if not loc_val or loc_val.lower() == "nan":
                continue

            if loc_val in KNOWN_MACRO_REGIONS:
                current_region = loc_val if loc_val != "WHOLE COUNTRY" else None
                continue

            if loc_val == "WHOLE COUNTRY":
                continue

            cleaned_loc = clean_province_names(pd.Series([loc_val]))[0]

            if loc_val in KNOWN_PROVINCES or cleaned_loc in KNOWN_PROVINCES:
                for orig_col, yr in year_col_map.items():
                    val = clean_numeric_val(row[orig_col])
                    all_records.append({
                        "province_name": cleaned_loc,
                        "macro_region": current_region,
                        "year": yr,
                        "season": season,
                        "metric": metric_name,
                        "value": val
                    })

    if not all_records:
        raise ValueError("No valid data frames were extracted from the CSV files.")

    df_long = pd.DataFrame(all_records)

    # Pivot metrics to wide format
    df_pivoted = df_long.pivot_table(
        index=["province_name", "macro_region", "year", "season"],
        columns="metric",
        values="value",
        aggfunc="max"
    ).reset_index()

    for col in ["planted_area_ha", "yield_ton_ha", "production_tons"]:
        if col not in df_pivoted.columns:
            df_pivoted[col] = np.nan

    # Convert yield from quintals/ha to tons/ha
    df_pivoted["yield_ton_ha"] = df_pivoted["yield_ton_ha"].apply(
        lambda x: round(x / 10.0, 2) if pd.notna(x) and x > 15.0 else (round(x, 2) if pd.notna(x) else np.nan)
    )

    # Apply Linear Time-Series Imputation
    df_clean = impute_missing_agricultural_data(df_pivoted)

    return df_clean


def clean_gso_target_csv(
    raw_gso_path: Path,
    output_path: Path,
) -> None:
    """Loads raw GSO CSVs, standardizes names, applies imputation, and saves cleaned dataset."""
    logger.info("Processing raw GSO data from %s...", raw_gso_path)

    df = load_and_reshape_gso_files(raw_gso_path)

    df["macro_region"] = df["macro_region"].fillna(df["province_name"].map(PROVINCE_TO_REGION))
    df = df.sort_values(by=["province_name", "year", "season"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned GSO target file with %d rows to %s", len(df), output_path)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    default_raw_gso = project_root / "data" / "gso"
    default_output = project_root / "data" / "cleaned_gso.csv"

    parser = argparse.ArgumentParser(
        description="Clean raw GSO target data and align province names with GEE boundaries"
    )
    parser.add_argument(
        "--raw-gso",
        type=Path,
        default=default_raw_gso,
        help=f"Path to folder or file of raw GSO targets CSV (default: {default_raw_gso})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to write cleaned CSV (default: {default_output})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clean_gso_target_csv(args.raw_gso, args.output)


if __name__ == "__main__":
    main()