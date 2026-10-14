import argparse
import logging
from pathlib import Path

import ee
import pandas as pd

from auth import initialize_gee
from province_mappings import clean_province_names

logger = logging.getLogger(__name__)

def extract_static_environmental_features() -> pd.DataFrame:
    """Extracts static soil, topography, and climate features for Vietnam provinces using GEE."""
    initialize_gee()
    
    # 1. Load Vietnam ADM1 Provincial Boundaries from geoBoundaries
    vietnam_adm1 = (
    ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
    .filter(ee.Filter.eq("shapeGroup", "VNM"))
    )

    # 2. Load Elevation and Slope (SRTM DEM)
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select('elevation').rename('elevation_m')
    slope = ee.Terrain.slope(srtm).rename('slope_deg')

    # 3. Load Climate Normals (WorldClim v1 BioClimatic Variables)
    # BIO1 = Annual Mean Temperature (scaled by 0.1 to get °C), BIO12 = Annual Precipitation (mm)
    worldclim = ee.Image("WORLDCLIM/V1/BIO")
    mean_temp = worldclim.select('bio01').multiply(0.1).rename('mean_temp_c')
    annual_precip = worldclim.select('bio12').rename('annual_precip_mm')

    # 4. Load Static Soil Properties (ISRIC SoilGrids 0-5 cm depth)
    soil_ph = ee.Image("projects/soilgrids-isric/phh2o_mean").select('phh2o_0-5cm_mean').divide(10).rename('soil_ph')
    soil_clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean').divide(10).rename('soil_clay_pct')
    soil_soc = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean').divide(10).rename('soil_soc')

    # 5. Combine all raster layers into a single multi-band image
    combined_image = ee.Image.cat([
        elevation,
        slope,
        mean_temp,
        annual_precip,
        soil_ph,
        soil_clay,
        soil_soc
    ])

    logger.info("Reducing raster layers across Vietnam provincial boundaries...")
    
    # 6. Extract mean values per province polygon using regional reduction
    province_stats = combined_image.reduceRegions(
        collection=vietnam_adm1,
        reducer=ee.Reducer.mean(),
        scale=250,  # Matches SoilGrids 250m resolution
        crs="EPSG:4326"
    )

    # 7. Pull features directly into a pandas DataFrame
    features = province_stats.getInfo()["features"]
    data = [f["properties"] for f in features]
    df_static = pd.DataFrame(data)

    # Drop GEE geometry metadata if present
    if ".geo" in df_static.columns:
        df_static = df_static.drop(columns=[".geo"])

    # Standardize province names to match your dynamic SAR pipeline
    if "shapeName" in df_static.columns:
        df_static = df_static.rename(columns={"shapeName": "province_name"})
    
    if "province_name" in df_static.columns:
        df_static["province_name"] = clean_province_names(df_static["province_name"])

    return df_static

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract static soil and environmental GEE features")
    parser.add_argument("--output", type=Path, default=Path("data/static_features_raw.csv"), help="Output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Extracting static attributes from Google Earth Engine...")
    df_static = extract_static_environmental_features()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_static.to_csv(args.output, index=False)
    logger.info("Saved %d static feature rows to %s", len(df_static), args.output)

if __name__ == "__main__":
    main()