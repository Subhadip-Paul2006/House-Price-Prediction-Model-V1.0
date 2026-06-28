"""
generate_data.py — One-time script to create a synthetic house-price dataset.

Generates ~1000 rows of Indian real-estate data with 16 features and realistic
feature distributions and a few intentional NaN values so that the preprocessing
pipeline has something to clean.

Usage:
    python generate_data.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# SEED fix kar rahe taaki random numbers har baar same hi milein
SEED = 42
N_SAMPLES = 1000
OUTPUT_PATH = os.path.join("data", "house_data.csv")

# Location ke hisab se rates adjust karne ke liye multiplier set kiya hai (Downtown sabse mehanga, Rural sasta)
LOCATION_MULTIPLIERS = {
    "Downtown": 2.0,
    "Urban": 1.6,
    "Suburban": 1.2,
    "Rural": 0.8,
}

# Neighborhoods grouped by location type
NEIGHBORHOODS = {
    "Downtown": ["Central Business", "Downtown Historic"],
    "Urban": ["West Side", "East End", "Old Town", "South Campus", "Cedar Rapids",
              "Oak Park", "East Urban", "Brookside", "Somerset"],
    "Suburban": ["Meadow Village", "SW Ames", "NW Ames", "Mitchell", "Sawyer",
                 "North Ames", "Edwards", "Bluestem", "Northridge Heights",
                 "Stone Brook", "Clear Creek", "College View", "Iowa State Area",
                 "University Heights"],
    "Rural": ["Greens", "Veenker", "Industrial District"],
}

HOUSE_STYLES = ["1Story", "SLvl", "2Story", "1.5Unf", "SFoyer", "1.5Fin"]
KITCHEN_QUALITIES = ["Po", "Fa", "TA", "Gd", "Ex"]
CENTRAL_AIR_OPTIONS = ["Yes", "No"]


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with synthetic house-price data (16 features + Price)."""

    # RNG setup kar rahe hain seed ke sath
    rng = np.random.default_rng(seed)

    # --- Feature values randomly generate kar rahe hain ---
    # Bug Fix #1-9: Ranges now match or exceed UI slider limits
    # np.random.Generator.integers(low, high) is [low, high) — so use high+1 for inclusive max
    area = rng.integers(300, 6001, size=n)                   # Bug Fix #6: matches UI [300, 6000]
    bedrooms = rng.integers(1, 11, size=n)                     # Bug Fix #2: matches UI [1, 10]
    bathrooms = rng.integers(1, 9, size=n)                      # Bug Fix #3: matches UI [1, 8]
    age = rng.integers(0, 51, size=n)                           # Ghar kitna purana hai (0-50 years)
    locations = rng.choice(
        list(LOCATION_MULTIPLIERS.keys()),
        size=n,
        p=[0.15, 0.35, 0.35, 0.15],
    )

    # New features — ranges aligned with app.py sliders
    lot_area = rng.integers(800, 45001, size=n)                # Bug Fix #7: matches UI [800, 45000]
    overall_quality = rng.integers(1, 11, size=n)              # Bug Fix #1: matches UI [1, 10]
    overall_condition = rng.integers(1, 11, size=n)           # Bug Fix #1: matches UI [1, 10]
    garage_cars = rng.integers(0, 5, size=n)                  # Bug Fix #4: matches UI [0, 4]
    garage_area = np.where(
        garage_cars > 0,
        rng.integers(200, 2001, size=n),                       # Bug Fix #8: matches UI [0, 2000]
        0,
    )
    total_basement_sf = np.where(
        rng.random(n) > 0.2,
        rng.integers(200, 2501, size=n),                       # Bug Fix #9: matches UI [0, 2500]
        0,
    )
    fireplaces = rng.integers(0, 5, size=n)                    # Bug Fix #5: matches UI [0, 4]
    central_air = rng.choice(CENTRAL_AIR_OPTIONS, size=n, p=[0.7, 0.3])
    kitchen_quality = rng.choice(KITCHEN_QUALITIES, size=n,
                                 p=[0.05, 0.15, 0.45, 0.25, 0.10])

    # Neighborhood based on location
    neighborhoods = []
    for loc in locations:
        choices = NEIGHBORHOODS[loc]
        neighborhoods.append(rng.choice(choices))

    # House style
    house_style = rng.choice(HOUSE_STYLES, size=n)

    # --- Ek basic price formula set kar rahe hain (Lakhs me) ---
    base_price_per_sqft = 0.035
    price = (
        area * base_price_per_sqft
        + bedrooms * 5.0
        + bathrooms * 3.0
        - age * 0.3
        + lot_area * 0.002
        + overall_quality * 8.0
        + overall_condition * 3.0
        + garage_cars * 6.0
        + garage_area * 0.005
        + total_basement_sf * 0.008
        + fireplaces * 4.0
    )

    # Central air and kitchen quality premium/discount
    air_mult = np.where(np.array(central_air) == "Yes", 1.05, 0.92)
    kitchen_mult_map = {"Po": 0.85, "Fa": 0.92, "TA": 1.0, "Gd": 1.08, "Ex": 1.18}
    kitchen_mult = np.array([kitchen_mult_map[kq] for kq in kitchen_quality])
    price = price * air_mult * kitchen_mult

    # Location multiplier apply kar rahe hain
    location_mult = [LOCATION_MULTIPLIERS[loc] for loc in locations]
    price = price * location_mult

    # Thoda real-world market noise add kar rahe hain (around 10% fluctuation)
    noise = rng.normal(1.0, 0.10, size=n)
    price = price * noise

    # Kuch bhi ho jaye, price 5 Lakh se kam nahi hona chahiye
    price = np.maximum(price, 5.0)

    # Decimal values ko 2 places tak round kar rahe hain
    price = np.round(price, 2)

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": locations,
        "Price": price,
        "Lot Area": lot_area,
        "Overall Quality": overall_quality,
        "Overall Condition": overall_condition,
        "Garage Cars": garage_cars,
        "Garage Area": garage_area,
        "Total Basement SF": total_basement_sf,
        "Fireplaces": fireplaces,
        "Neighborhood": neighborhoods,
        "House Style": house_style,
        "Central Air": central_air,
        "Kitchen Quality": kitchen_quality,
    })

    # --- Jaanबूझkar kuch missing/NaN values daal rahe hain taaki preprocessor clean karna seekhe ---
    n_area_nan = int(0.02 * n)
    n_beds_nan = int(0.01 * n)
    n_loc_nan = int(0.015 * n)
    n_basement_nan = int(0.02 * n)

    df.loc[rng.choice(n, n_area_nan, replace=False), "Area"] = np.nan
    df.loc[rng.choice(n, n_beds_nan, replace=False), "Bedrooms"] = np.nan
    df.loc[rng.choice(n, n_loc_nan, replace=False), "Location"] = np.nan
    df.loc[rng.choice(n, n_basement_nan, replace=False), "Total Basement SF"] = np.nan

    return df


def main() -> None:
    # Target directory exist nahi karti toh pehle folder create karo
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {len(df)} rows → {OUTPUT_PATH}")
    print(f"   Columns: {df.shape[1]}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    print(f"\n   Sample rows:\n{df.head()}")
    print(f"\n   Price statistics:\n{df['Price'].describe()}")


if __name__ == "__main__":
    main()
