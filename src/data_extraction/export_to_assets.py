"""
Step 1: Extract seasonal SAR and climate zonal features per province in Google
Earth Engine and export them directly as a GEE Asset table.
"""

import argparse
import logging

import ee

from auth import initialize_gee

logger = logging.getLogger(__name__)

# ==========================================
# 1. PARAMETERS & AOI SETUP
# ==========================================
# Full timeline matching GSO target data (2005-2025)
YEARS = list(range(2005, 2026))

# Vietnam's 3 main rice crop seasons
SEASONS = {
    "spring": {"start_mm_dd": "11-15", "end_mm_dd": "04-15"},  # Winter-Spring / DongXuan
    "autumn": {"start_mm_dd": "04-16", "end_mm_dd": "08-15"},  # Summer-Autumn / HeThu
    "winter": {"start_mm_dd": "08-16", "end_mm_dd": "11-14"},  # Autumn-Winter / ThuDong
}


def extract_seasonal_metrics(
    provinces: ee.FeatureCollection,
    agri_mask: ee.Image,
    year: int,
    season_name: str,
    season_dates: dict,
) -> ee.FeatureCollection:
    """Extracts pre-computed zonal statistics for SAR and Climate per province polygon."""

    # Handle cross-year date window for Spring (DongXuan)
    if season_name == "spring":
        start_date = f"{year - 1}-{season_dates['start_mm_dd']}"
        end_date = f"{year}-{season_dates['end_mm_dd']}"
    else:
        start_date = f"{year}-{season_dates['start_mm_dd']}"
        end_date = f"{year}-{season_dates['end_mm_dd']}"

    # --- A. Sentinel-1 SAR Features (Available from late 2014) ---
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(provinces)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select(["VV", "VH"])
    )

    s1_processed = s1.map(
        lambda img: img.addBands(img.select("VH").subtract(img.select("VV")).rename("VH_VV_ratio"))
    )

    s1_min = s1_processed.select(["VH", "VV"]).reduce(ee.Reducer.min()).rename(["VH_min", "VV_min"])
    s1_max = s1_processed.select(["VH", "VV"]).reduce(ee.Reducer.max()).rename(["VH_max", "VV_max"])
    s1_mean = (
        s1_processed.select(["VH", "VV", "VH_VV_ratio"])
        .reduce(ee.Reducer.mean())
        .rename(["VH_mean", "VV_mean", "VH_VV_ratio_mean"])
    )
    s1_range = s1_max.subtract(s1_min).rename(["VH_range", "VV_range"])

    # --- B. ERA5 Climate Features (Full 2005–2025 coverage) ---
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(provinces)
        .filterDate(start_date, end_date)
    )

    precip_sum = (
        era5.select("total_precipitation_sum")
        .reduce(ee.Reducer.sum())
        .multiply(1000)
        .rename("precip_total_mm")
    )
    temp_mean = (
        era5.select("temperature_2m")
        .reduce(ee.Reducer.mean())
        .subtract(273.15)
        .rename("temp_mean_c")
    )

    # Combine all feature bands and mask non-cropland areas
    composite = ee.Image.cat(
        [s1_min, s1_max, s1_mean, s1_range, precip_sum, temp_mean]
    ).updateMask(agri_mask)

    # --- C. Spatial Zonal Aggregation Over 63 Province Polygons ---
    extracted_data = composite.reduceRegions(
        collection=provinces,
        reducer=ee.Reducer.mean(),
        scale=100,  # 100m scale for fast polygon mean extraction
        crs="EPSG:4326",
    )

    def add_metadata(feature):
        return feature.set(
            {
                "year": year,
                "season": season_name,
                "province_name": feature.get("ADM1_NAME"),
            }
        )

    return extracted_data.map(add_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract satellite/climate zonal features and export as a GEE Asset Table"
    )
    parser.add_argument(
        "--asset-id",
        required=True,
        help="Destination GEE Asset ID (e.g., projects/your-project-name/assets/vietnam_rice_features_2005_2025)",
    )
    parser.add_argument(
        "--description",
        default="export_vietnam_rice_features",
        help="GEE Task description name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Initializing Earth Engine...")
    initialize_gee()

    # Load boundaries (GAUL Level 1 covers all 63 Vietnamese provinces)
    provinces = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
        ee.Filter.eq("ADM0_NAME", "Viet Nam")
    )

    # WorldCover agricultural cropland mask
    worldcover = ee.Image("ESA/WorldCover/v100/2020")
    agri_mask = worldcover.select("Map").eq(40)  # Class 40 = Cropland

    combined_results = ee.FeatureCollection([])

    for year in YEARS:
        for season_name, season_dates in SEASONS.items():
            logger.info("Processing Year: %d, Season: %s", year, season_name)
            seasonal_fc = extract_seasonal_metrics(
                provinces, agri_mask, year, season_name, season_dates
            )
            combined_results = combined_results.merge(seasonal_fc)

    # Submit single batch task to export tabular zonal stats to GEE Asset
    logger.info("Submitting export task to GEE Asset ID: %s", args.asset_id)
    task = ee.batch.Export.table.toAsset(
        collection=combined_results,
        description=args.description,
        assetId=args.asset_id,
    )

    task.start()
    logger.info("Task successfully started! Task ID: %s", task.id)
    logger.info("Monitor status in GEE Code Editor or GEE Tasks tab.")


if __name__ == "__main__":
    main()