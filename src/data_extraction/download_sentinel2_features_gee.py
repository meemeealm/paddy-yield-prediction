import argparse
import logging
from pathlib import Path

import ee
import pandas as pd

from auth import initialize_gee
from province_mappings import clean_province_names

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Season windows: (year_offset, month, day) for start and end (end is exclusive).
# year_offset=-1 means the previous calendar year, +1 the next one.
# IMPORTANT: change these so they match the windows used in your Sentinel-1 pipeline.
# ------------------------------------------------------------
SEASON_WINDOWS = {
    "spring": ((-1, 12, 1), (0, 5, 1)),   # winter-spring crop: Dec (prev year) - Apr
    "autumn": ((0, 5, 1), (0, 9, 1)),     # summer-autumn crop: May - Aug
    "winter": ((0, 9, 1), (1, 1, 1)),     # autumn-winter crop: Sep - Dec
}

INDEX_BANDS = ["NDVI", "EVI", "LSWI"]


def add_indices(img: ee.Image) -> ee.Image:
    """Compute NDVI, EVI and LSWI from a cloud-masked Sentinel-2 SR image."""
    refl = img.select(["B2", "B4", "B8", "B11"]).divide(10000)
    blue, red, nir, swir = refl.select("B2"), refl.select("B4"), refl.select("B8"), refl.select("B11")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = (
        nir.subtract(red).multiply(2.5)
        .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
        .clamp(-1, 1)
        .rename("EVI")
    )
    lswi = nir.subtract(swir).divide(nir.add(swir)).rename("LSWI")  # water / flooding signal

    return ndvi.addBands(evi).addBands(lswi)


def extract_sentinel2_features(
    start_year: int,
    end_year: int,
    scale: int = 100,
    cloud_threshold: float = 0.6,
    mask_cropland: bool = True,
) -> pd.DataFrame:
    """Extracts seasonal Sentinel-2 vegetation/water index statistics per province using GEE."""
    initialize_gee()

    # 1. Vietnam ADM1 Provincial Boundaries from geoBoundaries
    vietnam_adm1 = (
        ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
        .filter(ee.Filter.eq("shapeGroup", "VNM"))
    )

    # 2. Cloud Score+ (cloud/shadow quality band) linked to Sentinel-2 surface reflectance
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    # 3. Optional cropland mask (ESA WorldCover 2021, class 40 = cropland) so that
    #    forests, cities and water do not dilute the province averages
    cropland = ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(40)

    # 4. Per-pixel temporal statistics over the season
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
    )

    all_frames = []

    for year in range(start_year, end_year + 1):
        for season, (start_off, end_off) in SEASON_WINDOWS.items():
            start = f"{year + start_off[0]}-{start_off[1]:02d}-{start_off[2]:02d}"
            end = f"{year + end_off[0]}-{end_off[1]:02d}-{end_off[2]:02d}"

            try:
                s2 = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterDate(start, end)
                    .filterBounds(vietnam_adm1)
                    .linkCollection(cs_plus, ["cs_cdf"])
                    .map(lambda img: img.updateMask(img.select("cs_cdf").gte(cloud_threshold)))
                    .map(add_indices)
                )

                n_images = s2.size().getInfo()
                if n_images == 0:
                    logger.warning("No Sentinel-2 images for %s %s (%s to %s)", year, season, start, end)
                    continue

                season_image = s2.reduce(reducer)
                if mask_cropland:
                    season_image = season_image.updateMask(cropland)

                province_stats = season_image.reduceRegions(
                    collection=vietnam_adm1,
                    reducer=ee.Reducer.mean(),
                    scale=scale,
                    crs="EPSG:4326",
                    tileScale=4,
                )

                features = province_stats.getInfo()["features"]
                df_part = pd.DataFrame([f["properties"] for f in features])
                df_part["year"] = year
                df_part["season"] = season
                df_part["s2_n_images"] = n_images
                all_frames.append(df_part)

                logger.info("Done %s %s: %d images, %d provinces", year, season, n_images, len(df_part))

            except ee.EEException as e:
                logger.error("GEE error for %s %s: %s", year, season, e)
                continue

    if not all_frames:
        logger.warning("No Sentinel-2 data was extracted.")
        return pd.DataFrame()

    df_s2 = pd.concat(all_frames, ignore_index=True)

    # Standardize province names to match your dynamic SAR pipeline
    if "shapeName" in df_s2.columns:
        df_s2 = df_s2.rename(columns={"shapeName": "province_name"})

    if "province_name" in df_s2.columns:
        df_s2["province_name"] = clean_province_names(df_s2["province_name"])

    # Temporal range (max - min) per index, like the VH_range / VV_range features
    for idx in INDEX_BANDS:
        df_s2[f"{idx}_range"] = df_s2[f"{idx}_max"] - df_s2[f"{idx}_min"]

    # Keep only merge keys + features, in a clean order
    feature_cols = [f"{idx}_{stat}" for idx in INDEX_BANDS for stat in ["mean", "min", "max", "range"]]
    keep_cols = ["province_name", "year", "season", "s2_n_images"] + feature_cols
    df_s2 = df_s2[[c for c in keep_cols if c in df_s2.columns]]

    return df_s2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract seasonal Sentinel-2 features per province from GEE")
    parser.add_argument("--output", type=Path, default=Path("data/sentinel2_features_raw.csv"), help="Output path")
    parser.add_argument("--start-year", type=int, default=2015, help="First year to extract")
    parser.add_argument("--end-year", type=int, default=2025, help="Last year to extract")
    parser.add_argument("--scale", type=int, default=100, help="Reduction scale in metres")
    parser.add_argument("--cloud-threshold", type=float, default=0.6, help="Cloud Score+ cs_cdf threshold (0-1)")
    parser.add_argument("--no-cropland-mask", action="store_true", help="Average over all land, not just cropland")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Extracting Sentinel-2 seasonal features from Google Earth Engine...")
    df_s2 = extract_sentinel2_features(
        start_year=args.start_year,
        end_year=args.end_year,
        scale=args.scale,
        cloud_threshold=args.cloud_threshold,
        mask_cropland=not args.no_cropland_mask,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_s2.to_csv(args.output, index=False)
    logger.info("Saved %d Sentinel-2 feature rows to %s", len(df_s2), args.output)


if __name__ == "__main__":
    main()
