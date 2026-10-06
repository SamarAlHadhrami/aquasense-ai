"""
AquaSense machine learning prediction models (FR8)
-----------------------------------------------------
Trains Linear Regression and Random Forest Regression models to predict
energy consumption from water production, temperature, and seasonal
demand, per your report's Modelling stage.

Uses a CHRONOLOGICAL train/test split (not random) as specified in your
Evaluation framework (Table 3.12) - older records train the model, newer
records test it. This avoids data leakage in time-stamped data.

Evaluated with MAE, RMSE, and R^2 - the same metrics named in your report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "aquasense_analysed.csv"

FEATURES = ["Water_Production_m3", "Temperature_C", "Seasonal_Demand_Index"]
TARGET = "Energy_Consumption_kWh"


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def chronological_split(df, test_fraction=0.2):
    split_point = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    print(f"Train: {len(train)} rows ({train['Date'].min().date()} to "
          f"{train['Date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['Date'].min().date()} to "
          f"{test['Date'].max().date()})")
    return train, test


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} performance:")
    print(f"  MAE  = {mae:,.1f} kWh")
    print(f"  RMSE = {rmse:,.1f} kWh")
    print(f"  R^2  = {r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def plot_predictions(test_dates, y_true, predictions_dict, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_true, label="Actual", color="black", linewidth=1.5)
    for name, y_pred in predictions_dict.items():
        plt.plot(test_dates, y_pred, label=name, alpha=0.8)
    plt.title("Predicted vs Actual Energy Consumption (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption (kWh)")
    plt.legend()
    plt.tight_layout()
    # Static chart for report figures only — the live Streamlit dashboard is the source of truth for the demo.
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")


def train_and_predict(df):
    """Fit LR/RF on a chronological train split, predict for every row in
    df, and return df with LR_Prediction, RF_Prediction, and RF_Deviation
    columns added."""
    features = [f for f in FEATURES if f in df.columns]
    train, _ = chronological_split(df)
    X_train, y_train = train[features], train[TARGET]

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

    df = df.copy()
    df["LR_Prediction"] = lr.predict(df[features])
    df["RF_Prediction"] = rf.predict(df[features])
    df["RF_Deviation"] = (df[TARGET] - df["RF_Prediction"]).abs()
    return df


if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    train, test = chronological_split(df)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = []

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results.append(evaluate("Linear Regression", y_test, lr_pred))

    print("\nLinear Regression coefficients:")
    for feat, coef in zip(FEATURES, lr.coef_):
        print(f"  {feat}: {coef:,.2f}")

    # --- Random Forest Regression ---
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, rf_pred))

    print("\nRandom Forest feature importances:")
    for feat, imp in zip(FEATURES, rf.feature_importances_):
        print(f"  {feat}: {imp:.3f}")

    # --- Save comparison chart ---
    plot_predictions(
        test["Date"], y_test.values,
        {"Linear Regression": lr_pred, "Random Forest": rf_pred},
        "chart_ml_predictions.png",
    )

    # --- Save results table ---
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_evaluation_results.csv", index=False)
    print("\nSaved model_evaluation_results.csv")

    better_model = results_df.loc[results_df["R2"].idxmax(), "model"]
    print(f"\nBest performing model on this test set: {better_model}")