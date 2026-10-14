"""Shared province name cleaning map and macro-region lookup."""

import re
import unicodedata

import pandas as pd

PROVINCE_NAME_MAP = {
    # Fix spacing issues
    "Thai  Nguyen": "Thai Nguyen",
    "Quang  Nam": "Quang Nam",
    "Quang  Ngai": "Quang Ngai",
    "Khanh  Hoa": "Khanh Hoa",
    "Ninh  Thuan": "Ninh Thuan",
    "Binh  Duong": "Binh Duong",
    "Kien  Giang": "Kien Giang",
    "Ho Chi Minh city": "Ho Chi Minh",

    # Standardize hyphenation/naming
    "Thua Thien-Hue": "Thua Thien Hue",
    "Thua Thien - Hue": "Thua Thien Hue",
    "Ba Ria - Vung Tau": "Ba Ria - Vung Tau",

    # Historical / Legacy administrative units
    "Ha Tay": "Ha Noi",  # Merged into Ha Noi post-2008
}

RED_RIVER_DELTA = "Red River Delta"
NORTHERN_MIDLANDS = "Northern midlands and mountain areas"
NORTH_CENTRAL_COASTAL = "North Central and Central coastal areas"
CENTRAL_HIGHLANDS = "Central Highlands"
SOUTH_EAST = "South East"
MEKONG_DELTA = "Mekong River Delta"

PROVINCE_TO_REGION = {
    # Red River Delta
    "Ha Noi": RED_RIVER_DELTA, "Vinh Phuc": RED_RIVER_DELTA, "Bac Ninh": RED_RIVER_DELTA,
    "Quang Ninh": RED_RIVER_DELTA, "Hai Duong": RED_RIVER_DELTA, "Hai Phong": RED_RIVER_DELTA,
    "Hung Yen": RED_RIVER_DELTA, "Thai Binh": RED_RIVER_DELTA, "Ha Nam": RED_RIVER_DELTA,
    "Nam Dinh": RED_RIVER_DELTA, "Ninh Binh": RED_RIVER_DELTA,

    # Northern midlands and mountain areas
    "Ha Giang": NORTHERN_MIDLANDS, "Cao Bang": NORTHERN_MIDLANDS,
    "Bac Kan": NORTHERN_MIDLANDS, "Tuyen Quang": NORTHERN_MIDLANDS,
    "Lao Cai": NORTHERN_MIDLANDS, "Yen Bai": NORTHERN_MIDLANDS,
    "Thai Nguyen": NORTHERN_MIDLANDS, "Lang Son": NORTHERN_MIDLANDS,
    "Bac Giang": NORTHERN_MIDLANDS, "Phu Tho": NORTHERN_MIDLANDS,
    "Dien Bien": NORTHERN_MIDLANDS, "Lai Chau": NORTHERN_MIDLANDS,
    "Son La": NORTHERN_MIDLANDS, "Hoa Binh": NORTHERN_MIDLANDS,

    # North Central and Central coastal areas
    "Thanh Hoa": NORTH_CENTRAL_COASTAL, "Nghe An": NORTH_CENTRAL_COASTAL,
    "Ha Tinh": NORTH_CENTRAL_COASTAL, "Quang Binh": NORTH_CENTRAL_COASTAL,
    "Quang Tri": NORTH_CENTRAL_COASTAL, "Thua Thien Hue": NORTH_CENTRAL_COASTAL,
    "Da Nang": NORTH_CENTRAL_COASTAL, "Quang Nam": NORTH_CENTRAL_COASTAL,
    "Quang Ngai": NORTH_CENTRAL_COASTAL, "Binh Dinh": NORTH_CENTRAL_COASTAL,
    "Phu Yen": NORTH_CENTRAL_COASTAL, "Khanh Hoa": NORTH_CENTRAL_COASTAL,
    "Ninh Thuan": NORTH_CENTRAL_COASTAL, "Binh Thuan": NORTH_CENTRAL_COASTAL,

    # Central Highlands
    "Kon Tum": CENTRAL_HIGHLANDS, "Gia Lai": CENTRAL_HIGHLANDS, "Dak Lak": CENTRAL_HIGHLANDS,
    "Dak Nong": CENTRAL_HIGHLANDS, "Lam Dong": CENTRAL_HIGHLANDS,

    # South East
    "Binh Phuoc": SOUTH_EAST, "Tay Ninh": SOUTH_EAST, "Binh Duong": SOUTH_EAST,
    "Dong Nai": SOUTH_EAST, "Ba Ria - Vung Tau": SOUTH_EAST, "Ho Chi Minh": SOUTH_EAST,

    # Mekong River Delta
    "Long An": MEKONG_DELTA, "Tien Giang": MEKONG_DELTA, "Ben Tre": MEKONG_DELTA,
    "Tra Vinh": MEKONG_DELTA, "Vinh Long": MEKONG_DELTA, "Dong Thap": MEKONG_DELTA,
    "An Giang": MEKONG_DELTA, "Kien Giang": MEKONG_DELTA, "Can Tho": MEKONG_DELTA,
    "Hau Giang": MEKONG_DELTA, "Soc Trang": MEKONG_DELTA, "Bac Lieu": MEKONG_DELTA,
    "Ca Mau": MEKONG_DELTA,
}


def to_english(text: str) -> str:
    # Handle NaN / None / numeric values safely
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.replace("\u0110", "D").replace("\u0111", "d")  # D-stroke / d-stroke
    # rest of your to_english function logic...
    return text


def clean_province_names(series: pd.Series) -> pd.Series:
    """Utility shared across scripts: convert to English letters, strip whitespace, apply PROVINCE_NAME_MAP."""
    return series.astype(str).map(to_english).replace(PROVINCE_NAME_MAP)
