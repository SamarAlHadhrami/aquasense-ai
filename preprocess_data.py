"""
AquaSense data preprocessing pipeline (FR1, FR2)
--------------------------------------------------
Loads the raw dataset, cleans it, and produces a ready-to-analyse file.

Steps performed (matching Table 3.10 in your report):
  1. Data Cleaning    - remove duplicates, handle missing values
  2. Data Formatting  - standardize the date column
  3. Data Integration  - (already combined in one file by the generator)
  4. Data Transformation - calculate Specific Energy Consumption (SEC)
  5. Outlier Review   - flag values far outside the normal range
"""

import pandas as pd
import numpy as np

INPUT_FILE = "aquasense_sur_oman_dataset.csv"
OUTPUT_FILE = "aquasense_cleaned.csv"


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows from {path}")
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["Date"], keep="first")
    removed = before - len(df)
    print(f"Removed {removed} duplicate row(s)")
    return df


def handle_missing_values(df):
    missing_before = df["Energy_Consumption_kWh"].isna().sum()
    # For a time series, filling with the nearby days' average is more
    # realistic than dropping rows or using the whole-column mean.
    df["Energy_Consumption_kWh"] = df["Energy_Consumption_kWh"].interpolate(
        method="linear"
    )
    print(f"Filled {missing_before} missing Energy_Consumption_kWh value(s) "
          f"using linear interpolation")
    return df


def format_dates(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def calculate_sec(df):
    # Specific Energy Consumption = Energy Consumption (kWh) / Water Production (m3)
    df["Specific_Energy_Consumption_kWh_per_m3"] = (
        df["Energy_Consumption_kWh"] / df["Water_Production_m3"]
    ).round(3)
    return df


def calculate_ndp(df):
    """Estimate net driving pressure and flag physically implausible
    readings. Only runs when feed pressure and conductivity columns are
    present; otherwise leaves df unchanged."""
    if "Feed_Pressure_psi" not in df.columns or "Feed_Conductivity_mScm" not in df.columns:
        return df

    estimated_osmotic_pressure = df["Feed_Conductivity_mScm"] * 0.777
    df["Net_Driving_Pressure"] = df["Feed_Pressure_psi"] - estimated_osmotic_pressure
    df["Physics_Flag"] = (df["Net_Driving_Pressure"] <= 0).astype(int)
    return df


def flag_outliers(df, column="Specific_Energy_Consumption_kWh_per_m3", z_thresh=3):
    mean = df[column].mean()
    std = df[column].std()
    df["Outlier_Zscore_Flag"] = (
        (df[column] - mean).abs() > z_thresh * std
    ).astype(int)
    n_outliers = df["Outlier_Zscore_Flag"].sum()
    print(f"Flagged {n_outliers} statistical outlier(s) in {column} "
          f"(|z| > {z_thresh})")
    return df


def summarize(df):
    print("\n--- Cleaned dataset summary ---")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Remaining missing values: {df.isna().sum().sum()}")
    print(df.select_dtypes("number").describe().round(2).to_string())


if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    df = remove_duplicates(df)
    df = format_dates(df)
    df = handle_missing_values(df)
    df = calculate_sec(df)
    df = flag_outliers(df)
    summarize(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset -> {OUTPUT_FILE}")