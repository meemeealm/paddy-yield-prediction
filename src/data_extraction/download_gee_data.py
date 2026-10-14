import argparse
import logging
from pathlib import Path

import ee
import pandas as pd

from auth import initialize_gee
from province_mappings import clean_province_names

logger = logging.getLogger(__name__)

DEFAULT_ASSET_ID = "projects/your-project-name/assets/vietnam_rice_features_2005_2025"

def download_gee_features(asset_id: str) -> pd.DataFrame:
    """Load the exported Asset from GEE and convert it to a local pandas DataFrame."""
    initialize_gee()
    fc = ee.FeatureCollection(asset_id)
    
    # Extract features directly into a list of dictionaries to avoid geemap drop errors
    features = fc.getInfo()["features"]
    data = [f["properties"] for f in features]
        
    df_gee = pd.DataFrame(data)

    # Clean unused GEE system properties if present
    cols_to_drop = [c for c in [".geo", "system:index"] if c in df_gee.columns]
    if cols_to_drop:
        df_gee = df_gee.drop(columns=cols_to_drop)

    # Standardize names and types
    df_gee["province_name"] = clean_province_names(df_gee["province_name"])
    df_gee["year"] = df_gee["year"].astype(int)
    df_gee["season"] = df_gee["season"].astype(str).str.lower()

    return df_gee

def main() -> None:
    parser = argparse.ArgumentParser(description="Download GEE Asset features and save to root/data")
    parser.add_argument("--asset-id", default=DEFAULT_ASSET_ID, help="GEE Asset ID")
    parser.add_argument("--output", type=Path, default=Path("data/gee_features_raw.csv"), help="Output path in root/data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Downloading GEE Asset %s...", args.asset_id)
    df_gee = download_gee_features(args.asset_id)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_gee.to_csv(args.output, index=False)
    logger.info("Saved %d GEE rows to %s", len(df_gee), args.output)

if __name__ == "__main__":
    main()