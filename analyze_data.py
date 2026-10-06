"""
AquaSense analysis modules (FR4, FR5, FR6, FR7)
--------------------------------------------------
Runs on the cleaned dataset (aquasense_cleaned.csv) and produces:
  1. Statistical summary            (FR4)
  2. Time-series trend decomposition (FR5)
  3. Correlation analysis            (FR6)
  4. Anomaly detection               (FR7) - Isolation Forest + Z-score

Each section prints its results and saves a chart as a PNG so you can
drop the images straight into your report or dashboard later.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save charts to file instead of popping up a window
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from scipy import stats

INPUT_FILE = "aquasense_cleaned.csv"


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    return df


# ---------------------------------------------------------------------
# FR4: Statistical Analysis
# ---------------------------------------------------------------------
def statistical_analysis(df):
    print("\n=== FR4: Statistical Analysis ===")
    cols = ["Energy_Consumption_kWh", "Water_Production_m3",
            "Specific_Energy_Consumption_kWh_per_m3"]
    summary = df[cols].agg(["mean", "std", "min", "max"]).round(2)
    print(summary.to_string())
    return summary


# ---------------------------------------------------------------------
# FR5: Time-Series Analysis
# ---------------------------------------------------------------------
def time_series_analysis(df):
    print("\n=== FR5: Time-Series Analysis ===")
    series = df["Energy_Consumption_kWh"].asfreq("D").interpolate()
    result = seasonal_decompose(series, model="additive", period=30)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed Energy Consumption")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal Pattern")
    result.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    # Static chart for report figures only — the live Streamlit dashboard is the source of truth for the demo.
    plt.savefig("chart_time_series_decomposition.png", dpi=120)
    plt.close()
    print("Saved chart_time_series_decomposition.png")

    trend_change = result.trend.dropna().iloc[-1] - result.trend.dropna().iloc[0]
    direction = "increasing" if trend_change > 0 else "decreasing"
    print(f"Overall trend is {direction} "
          f"({trend_change:+.0f} kWh over the observed trend period)")
    return result


# ---------------------------------------------------------------------
# FR6: Correlation Analysis
# ---------------------------------------------------------------------
def correlation_analysis(df):
    print("\n=== FR6: Correlation Analysis ===")
    cols = ["Energy_Consumption_kWh", "Water_Production_m3",
            "Temperature_C", "Seasonal_Demand_Index"]
    corr_matrix = df[cols].corr(method="pearson")
    print(corr_matrix.round(3).to_string())

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    # Static chart for report figures only — the live Streamlit dashboard is the source of truth for the demo.
    plt.savefig("chart_correlation_heatmap.png", dpi=120)
    plt.close()
    print("Saved chart_correlation_heatmap.png")

    # report the strongest relationship with energy consumption
    energy_corr = corr_matrix["Energy_Consumption_kWh"].drop("Energy_Consumption_kWh")
    strongest = energy_corr.abs().idxmax()
    r_val = energy_corr[strongest]
    p_val = stats.pearsonr(df["Energy_Consumption_kWh"], df[strongest])[1]
    print(f"Strongest relationship: Energy vs {strongest} "
          f"(r = {r_val:.3f}, p = {p_val:.4f})")
    return corr_matrix


# ---------------------------------------------------------------------
# FR7: Anomaly Detection (Isolation Forest + Z-score)
# ---------------------------------------------------------------------
def anomaly_detection(df, contamination=0.02):
    print("\n=== FR7: Anomaly Detection ===")

    # Z-score method on Specific Energy Consumption
    sec = df["Specific_Energy_Consumption_kWh_per_m3"]
    z_scores = np.abs(stats.zscore(sec))
    df["Zscore_Anomaly"] = (z_scores > 3).astype(int)

    # Isolation Forest on energy + production together
    features = df[["Energy_Consumption_kWh", "Water_Production_m3"]]
    model = IsolationForest(contamination=contamination, random_state=42)
    df["IsoForest_Anomaly"] = (model.fit_predict(features) == -1).astype(int)

    n_z = df["Zscore_Anomaly"].sum()
    n_if = df["IsoForest_Anomaly"].sum()
    print(f"Z-score flagged {n_z} anomalies")
    print(f"Isolation Forest flagged {n_if} anomalies")

    # compare against the ground-truth flag from the data generator, if present
    # NOTE: this compares against a label the same synthetic generator inserted — not an
    # independent validation. Do not report this number as real-world detection accuracy.
    if "Known_Anomaly_Flag" in df.columns:
        true_positives = ((df["IsoForest_Anomaly"] == 1) &
                           (df["Known_Anomaly_Flag"] == 1)).sum()
        total_known = df["Known_Anomaly_Flag"].sum()
        print(f"Isolation Forest caught {true_positives} of {total_known} "
              f"known injected anomalies")

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Energy_Consumption_kWh"], label="Energy Consumption", alpha=0.7)
    flagged = df[df["IsoForest_Anomaly"] == 1]
    plt.scatter(flagged.index, flagged["Energy_Consumption_kWh"],
                color="red", label="Flagged Anomaly", zorder=5)
    plt.title("Anomaly Detection: Energy Consumption")
    plt.xlabel("Date")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    # Static chart for report figures only — the live Streamlit dashboard is the source of truth for the demo.
    plt.savefig("chart_anomaly_detection.png", dpi=120)
    plt.close()
    print("Saved chart_anomaly_detection.png")
    return df


if __name__ == "__main__":
    df = load_data(INPUT_FILE)

    statistical_analysis(df)
    time_series_analysis(df)
    correlation_analysis(df)
    df = anomaly_detection(df)

    df.to_csv("aquasense_analysed.csv")
    print("\nSaved aquasense_analysed.csv (includes anomaly flags)")
    print("\nAll charts saved as PNG files in this folder.")