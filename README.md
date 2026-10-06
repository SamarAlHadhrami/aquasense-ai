# AquaSense

An interactive dashboard for monitoring and analysing energy use at a seawater reverse osmosis (SWRO) desalination plant. The data is a **synthetic** daily dataset calibrated to published figures for the Sharqiyah Desalination Plant in Sur, Oman and to climate normals for Sur.

**Live app:** _add your Streamlit link here_

## Features

- **Dashboard and Energy Overview** - KPIs for energy, water production and specific energy consumption (SEC, kWh/m³)
- **Trends and Comparisons** - time-series decomposition and period comparisons
- **Anomaly Detection** - Isolation Forest and Z-score, with an Accept/Reject decision log
- **Predictions** - Linear Regression and Random Forest models forecasting energy consumption, evaluated with MAE, RMSE and R² on a chronological train/test split
- **Recommendations and Reports** - suggested actions and CSV export
- **Upload your own CSV** to run the same pipeline on other data

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `generate_dataset.py` | Creates the synthetic raw dataset |
| `preprocess_data.py` | Cleaning, date formatting, SEC calculation, outlier flagging |
| `analyze_data.py` | Summary stats, decomposition, correlation, anomaly detection |
| `train_models.py` | Model training and evaluation |
| `aquasense_*.csv` | Raw, cleaned and analysed datasets |
| `.streamlit/config.toml` | Forces light theme |

To rebuild the data from scratch, run in order: `generate_dataset.py`, `preprocess_data.py`, `analyze_data.py`, `train_models.py`.

## Deployment

Hosted on Streamlit Community Cloud from the `main` branch, with `app.py` as the entry point. The decision log (`decision_log.csv`) is not persisted between restarts on the cloud.

## Notes

The dataset is synthetic and for demonstration and academic use only. It does not contain real plant measurements.
