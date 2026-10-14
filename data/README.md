# Datasets Descriptions

## GSO Data Code Mapping 

| GSO Table Code | Crop Season | Target Metric | Metric Column Name |
| :--- | :--- | :--- | :--- |
| **E06.16** | Spring Rice | Planted Area | `planted_area_ha` |
| **E06.17** | Spring Rice | Yield | `yield_ton_ha` |
| **E06.18** | Spring Rice | Production | `production_tons` |
| **E06.19** | Autumn Rice | Planted Area | `planted_area_ha` |
| **E06.20** | Autumn Rice | Yield | `yield_ton_ha` |
| **E06.21** | Autumn Rice | Production | `production_tons` |
| **E06.22** | Winter Rice | Planted Area | `planted_area_ha` |
| **E06.23** | Winter Rice | Yield | `yield_ton_ha` |
| **E06.24** | Winter Rice | Production | `production_tons` |

## Handling Missing Values in GSO data

| Step | Method | Scope | Description |
| :--- | :--- | :--- | :--- |
| **1. Primary Imputation** | Linear Interpolation | Per Province & Season | Fills temporal gaps across continuous years. |
| **2. Edge Case Filling** | Forward & Backward Fill | Per Province & Season | Resolves boundary nulls at 2005/2025 limits. |
| **3. Relation Recovery** | Exact Formula (P/Y) | Per Row | Recovers missing area using known production and yield. |

---

## Feature Description

### Agricultural Production and SAR Features

| Feature | What It Means |
| :--- | :--- |
| `ADM0_NAME` | Name of the country. |
| `ADM1_NAME` | Name of the first-level administrative subdivision. |
| `VH_VV_ratio_mean` | Mean ratio between VH- and VV-polarized satellite radar backscatter measurements. |
| `VH_max` | Maximum observed VH-polarized radar backscatter value during the observation period. |
| `VH_mean` | Mean VH-polarized radar backscatter value during the observation period. |
| `VH_min` | Minimum observed VH-polarized radar backscatter value during the observation period. |
| `VH_range` | Difference between the maximum and minimum VH-polarized radar backscatter values. |
| `VV_max` | Maximum observed VV-polarized radar backscatter value during the observation period. |
| `VV_mean` | Mean VV-polarized radar backscatter value during the observation period. |
| `VV_min` | Minimum observed VV-polarized radar backscatter value during the observation period. |
| `VV_range` | Difference between the maximum and minimum VV-polarized radar backscatter values. |
| `precip_total_mm` | Total precipitation associated with the location and observation period, measured in millimeters (mm). |
| `temp_mean_c` | Mean temperature associated with the location and observation period, measured in degrees Celsius (°C). |
| `composite_key` | Identifier combining the province, year, and season. |
| `macro_region` | Broad geographic region containing the province, such as the Mekong River Delta. |
| `planted_area_ha` | Total area planted with the target crop, measured in hectares (ha). |
| `production_tons` | Total production of the target crop, measured in metric tons. |
| `yield_ton_ha` | Crop yield, measured in tons per hectare (t/ha), representing production relative to the corresponding area measure. |
| `province_name` | Name of the province. |
| `year` | Calendar year associated with the observation. |
| `season` | Growing season associated with the observation, such as winter, spring, or autumn. |

### Environmental and Soil Features

| Feature | What It Means |
| :--- | :--- |
| `annual_precip_mm` | Total annual precipitation at the location, in millimeters (mm). |
| `elevation_m` | Elevation of the location above sea level, measured in meters (m). |
| `mean_temp_c` | Mean temperature, measured in degrees Celsius (°C). |
| `province_name` | Name of the province. |
| `slope_deg` | Terrain slope at the location, measured in degrees (°). |
| `soil_clay_pct` | Percentage of soil composition consisting of clay particles (%). |
| `soil_ph` | Soil pH value indicating soil acidity or alkalinity. Values below 7 are acidic, 7 is neutral, and values above 7 are alkaline. |
| `soil_soc` | Soil organic carbon (SOC) content, representing carbon stored in soil organic matter. |
| `soil` | Soil classification describing the soil group. |
| `latitude` | Latitude in decimal degrees. |
| `longitude` | Longitude in decimal degrees. |

### Sentinel-2 Vegetation Features

| Feature | What It Means |
| :--- | :--- |
| `NDVI_mean` | Mean normalized difference vegetation index measuring overall vegetation greenness and crop vigour. |
| `NDVI_max` | Maximum normalized difference vegetation index capturing peak crop greenness. |
| `EVI_mean` | Mean enhanced vegetation index; similar to NDVI but optimized to remain sensitive in dense canopies. |
| `EVI_max` | Maximum enhanced vegetation index capturing peak canopy greenness without saturation. |
| `LSWI_mean` | Mean land surface water index tracking general moisture levels and flooding conditions in paddies. |
| `LSWI_min` | Minimum land surface water index capturing the driest point in moisture or water coverage. |

---

## Data Sources

* Soil data - [Open Development Mekong](https://data.opendevelopmentmekong.net/dataset/soil-types-in-vietnam)
* Geo Boundaries data - [HDX / GeoBoundaries](https://data.humdata.org/dataset/geoboundaries-admin-boundaries-for-viet-nam)
* Satellite and Remote Sensing Features - GEE