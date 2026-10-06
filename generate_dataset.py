"""
AquaSense synthetic dataset generator
--------------------------------------
Generates a daily energy/water/environment dataset for a Gulf-region
seawater reverse osmosis (SWRO) desalination plant, calibrated to real,
published figures for the Sharqiyah Desalination Plant in Sur, Oman,
and real climate normals for Sur.

Real-world anchors used (sources noted inline):
  - Plant capacity:      131,837 m3/day (Sharqiyah Desalination Co., 2023 press release)
  - Typical delivery:    ~115,000 m3/day average (SDC company site)
  - Specific Energy Consumption (SEC): 3.2-4.25 kWh/m3
      * 3.2 kWh/m3  -> Sur plant extension, energy-recovery optimised (Veolia case study)
      * 3.75-4.25   -> general SWRO literature range (Stillwell & Webber, 2016;
                        PMC8540465 Canary Islands SWRO performance study)
  - Temperature climatology: Sur, Oman monthly average highs/lows
      (weather-atlas.com climate normals; approximate, for realism only)

This is SYNTHETIC data. It is built to be structurally and numerically
realistic (right order of magnitude, right seasonal shape, right SEC
range) so the analysis pipeline can be built and demonstrated now. It
is not a substitute for metered plant data and should be labelled as
synthetic/calibrated in the report.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. Date range: 2 years of daily data
# ---------------------------------------------------------------------
start_date = "2023-01-01"
end_date = "2024-12-31"
dates = pd.date_range(start=start_date, end=end_date, freq="D")
n = len(dates)

# ---------------------------------------------------------------------
# 2. Temperature model, anchored to real Sur, Oman monthly normals
#    (approximate average daily high, deg C, by month)
# ---------------------------------------------------------------------
monthly_avg_high = {
    1: 26.3, 2: 27.5, 3: 31.7, 4: 36.5, 5: 41.3, 6: 41.7,
    7: 40.0, 8: 38.8, 9: 38.4, 10: 34.5, 11: 31.1, 12: 27.5,
}
monthly_avg_low = {
    1: 17.7, 2: 19.5, 3: 22.0, 4: 25.5, 5: 28.5, 6: 30.7,
    7: 30.0, 8: 29.0, 9: 27.5, 10: 24.5, 11: 21.0, 12: 18.5,
}

months = dates.month
avg_high = months.map(monthly_avg_high).astype(float)
avg_low = months.map(monthly_avg_low).astype(float)
daily_mean_temp = (avg_high.values + avg_low.values) / 2

# add day-to-day weather noise on top of the smooth seasonal curve
temperature = daily_mean_temp + rng.normal(0, 1.0, n)
temperature = np.round(temperature, 1)

# ---------------------------------------------------------------------
# 3. Seasonal demand index (0-1), higher in hot months (real pattern:
#    Gulf water demand rises with summer heat / cooling & irrigation use)
# ---------------------------------------------------------------------
seasonal_demand = 0.55 + 0.45 * (temperature - temperature.min()) / (
    temperature.max() - temperature.min()
)
seasonal_demand = np.round(seasonal_demand, 3)

# ---------------------------------------------------------------------
# 4. Water production, anchored to Sharqiyah plant capacity
#    baseline ~115,000 m3/day, capacity ceiling 131,837 m3/day
# ---------------------------------------------------------------------
capacity = 131_837
baseline_production = 115_000

water_production = baseline_production * (0.85 + 0.20 * seasonal_demand)
water_production += rng.normal(0, 2500, n)          # daily operational noise
water_production = np.clip(water_production, 60_000, capacity)
water_production = np.round(water_production, 0)

# ---------------------------------------------------------------------
# 5. Specific Energy Consumption (SEC), anchored to real SWRO benchmarks
#    Slight real physical effect: warmer feedwater -> higher membrane
#    permeability -> marginally lower SEC. Modest noise added.
# ---------------------------------------------------------------------
sec_base = 3.6  # kWh/m3, between the optimised (3.2) and typical (3.75-4.25) benchmarks
temp_effect = -0.004 * (temperature - temperature.mean())  # small negative slope
sec = sec_base + temp_effect + rng.normal(0, 0.08, n)
sec = np.clip(sec, 3.1, 4.3)

# ---------------------------------------------------------------------
# 6. Energy consumption = production x SEC + auxiliary/fixed load
# ---------------------------------------------------------------------
auxiliary_load_kwh = 8_000  # intake pumps, pre-treatment, lighting, etc.
energy_consumption = water_production * sec + auxiliary_load_kwh
energy_consumption += rng.normal(0, 3000, n)
energy_consumption = np.round(energy_consumption, 0)

# ---------------------------------------------------------------------
# 7. Operating period label (mirrors your FR/data field list)
# ---------------------------------------------------------------------
operating_period = np.where(seasonal_demand > 0.75, "Peak", "Normal")

# ---------------------------------------------------------------------
# 8. Inject realistic anomalies (equipment fouling / trip events):
#    energy spikes NOT matched by a production increase - exactly the
#    kind of event your anomaly detection module (FR7) should catch.
# ---------------------------------------------------------------------
anomaly_idx = rng.choice(n, size=12, replace=False)
anomaly_flag = np.zeros(n, dtype=int)
for i in anomaly_idx:
    energy_consumption[i] *= rng.uniform(1.15, 1.35)   # abnormal energy spike
    anomaly_flag[i] = 1
energy_consumption = np.round(energy_consumption, 0)

# ---------------------------------------------------------------------
# 9. Deliberately add a few messy-data artifacts so the preprocessing
#    step (FR2: cleaning missing values / duplicates) has real work to do
# ---------------------------------------------------------------------
df = pd.DataFrame({
    "Date": dates,
    "Energy_Consumption_kWh": energy_consumption,
    "Water_Production_m3": water_production,
    "Temperature_C": temperature,
    "Seasonal_Demand_Index": seasonal_demand,
    "Operating_Period": operating_period,
    "Known_Anomaly_Flag": anomaly_flag,  # ground-truth label for evaluating your detector
})

# a few missing values
missing_idx = rng.choice(n, size=8, replace=False)
df.loc[missing_idx, "Energy_Consumption_kWh"] = np.nan

# a few duplicate rows
dup_rows = df.sample(3, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------------------
# 10. Specific Energy Consumption column (as your report's formula)
# ---------------------------------------------------------------------
df["Specific_Energy_Consumption_kWh_per_m3"] = (
    df["Energy_Consumption_kWh"] / df["Water_Production_m3"]
).round(3)

df.to_csv("aquasense_sur_oman_dataset.csv", index=False)
print(f"Generated {len(df)} rows -> aquasense_sur_oman_dataset.csv")
print(df.head(10).to_string(index=False))
print("\nSummary stats:")
print(df.select_dtypes("number").describe().round(2).to_string())