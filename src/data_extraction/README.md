# Vietnam Rice Dataset: GEE Features + GSO Statistics

Builds a province-level, season-level dataset for Vietnamese rice by pairing satellite and climate predictors from Google Earth Engine (GEE) with official GSO statistics.

## Pipeline

1. **Step 1: Feature extraction (`export_to_assets.py`)**
   Aggregates, per province and per season, over cropland pixels only (ESA WorldCover class 40):
   - **Sentinel-1 SAR:** `VH` / `VV` min, max, mean, range, and `VH_VV_ratio_mean`
   - **Sentinel-2 optical:** `NDVI_max`, `EVI_max`, `LSWI_max`
   - **ERA5-Land weather:** `precip_total_mm`, `temp_mean_c`

   The result is exported as a GEE Asset table.
2. **Step 2: Download (`download_and_merge_gso.py`)**
   Loads the exported Asset, converts it to a local pandas DataFrame, and merges it with the cleaned GSO CSV on `province_name`, `year`, `season`. Saves a `.parquet` (or `.csv`) file and logs any rows that did not match.
3. **Helper: GSO cleaning (`prep_gso_data.py`)**
   Converts province names to plain English letters (no Vietnamese diacritics), standardizes them (e.g. `Ha Tay` -> `Ha Noi`), attaches `macro_region`, and consolidates duplicates by `province_name`, `macro_region`, `year`, `season`: **area and production are summed, yields are area-weighted averages**, other columns use a simple mean.

Seasons used: `DongXuan` (Winter-Spring, 15 Nov of the previous year to 15 Apr), `HeThu` (Summer-Autumn, 16 Apr to 15 Aug), `ThuDong` (Autumn-Winter, 16 Aug to 14 Nov). Years: 2020 to 2025.

## Project structure

```text
.
├── export_to_assets.py          # Step 1: GEE feature extraction and Asset export
├── download_gee_data.py         # Step 2: download Asset to local
├── prep_gso_data.py             # Helper: clean raw GSO statistics
├── auth.py                      # Shared helper: initializes Earth Engine automatically
├── province_mappings.py         # Shared name-cleaning map and province -> region lookup
├── download_static_features_gee.py
├── requirements.txt             
└── README.md
```

## Dependencies Installation

```bash
pip install earthengine-api geemap pandas pyarrow python-dotenv
```

or:

```bash
pip install -r requirements.txt
```

| Library | Used for |
|---|---|
| `earthengine-api` | Google Earth Engine API (`import ee`) |
| `geemap` | Converting an Earth Engine FeatureCollection to pandas |
| `pandas` | Cleaning and merging tabular data |
| `pyarrow` | Writing/reading `.parquet` files |
| `python-dotenv` | loads settings from a `.env` file |

---

### Earth Engine authentication

No manual `ee.Initialize()` is needed here. Both GEE scripts call `initialize_gee()` from `auth.py`, which picks the credentials automatically:

If `GOOGLE_APPLICATION_CREDENTIALS` points to a service account JSON key, it is used (no browser login). 

Put settings in a `.env` file next to the scripts -

```text
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
GEE_PROJECT_NAME=gee-project
```

---

## Download GEE Data

### Step 1: Export features to a GEE Asset

```bash
python export_to_assets.py \
  --asset-id "projects/your-gee-project/assets/vietnam_rice_features_2020_2025" \
  --description "vietnam_rice_export_job"
```

`--description` is optional (default: `export_vietnam_rice_features`). The export runs on GEE servers; track it in the GEE Task Manager and wait until it shows `COMPLETED`.

### Helper: Clean the raw GSO CSV

```bash
python prep_gso_data.py \
  --raw-gso path/to/raw_gso.csv \
  --output data/gso_clean.csv
```

The raw CSV must contain `province_name`, `year`, and `season` (values `DongXuan`, `HeThu`, `ThuDong`), plus numeric columns. Column types are detected by name (case-insensitive):

| Name contains | Aggregation for duplicates |
|---|---|
| `area` | sum |
| `production` | sum |
| `yield` | weighted average, weighted by the first area column |
| anything else | mean |


## Notes

- All province names are converted to English letters only (e.g. `Thừa Thiên Huế` -> `Thua Thien Hue`) in both the GSO and GEE data before matching.
- If a province still fails to match (the scripts log a warning), extend `PROVINCE_NAME_MAP` in `province_mappings.py`.

----



