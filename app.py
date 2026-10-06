"""
AquaSense interactive dashboard (FR9) - matches the report's Initial
Prototype mockups (Figures 3.24 / 3.25).

Run with:  python -m streamlit run app.py

Requires (in addition to the earlier scripts' packages):
    python -m pip install streamlit-option-menu plotly

NOTE ON CARD STYLING: cards use st.container(border=True, key=...)
rather than hand-written <div>...</div> HTML split across multiple
st.markdown() calls. Streamlit renders every st.markdown() call as its
own isolated DOM node, so an opening <div> in one call and a closing
</div> in a later call do NOT visually nest - you get an empty styled
box followed by unstyled content. st.container(border=True) is the
correct, supported way to wrap native widgets (charts, dataframes,
buttons) in a bordered box; the key= argument gives it a stable CSS
class (st-key-<name>) so we can restyle it to match the mockup.
"""

import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess_data import (
    remove_duplicates, format_dates, handle_missing_values,
    calculate_sec, flag_outliers, calculate_ndp,
)
from analyze_data import anomaly_detection
from train_models import chronological_split, train_and_predict

DATA_FILE = "aquasense_analysed.csv"

# Minimum |deviation from expected energy| (%) for a day to be worth flagging as a
# recommendation card. Tune this live if the flagged list is too noisy/too quiet.
MIN_FLAG_DEVIATION_PCT = 5

DECISION_LOG_FILE = "decision_log.csv"
DECISION_LOG_COLUMNS = ["Date", "Deviation_Pct", "Decision", "Timestamp"]


def load_decision_log():
    if os.path.exists(DECISION_LOG_FILE):
        try:
            return pd.read_csv(DECISION_LOG_FILE)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame(columns=DECISION_LOG_COLUMNS)


def log_decision(date_str, deviation_pct, decision):
    file_exists = os.path.exists(DECISION_LOG_FILE)
    row = pd.DataFrame([{
        "Date": date_str,
        "Deviation_Pct": deviation_pct,
        "Decision": decision,
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
    }], columns=DECISION_LOG_COLUMNS)
    row.to_csv(DECISION_LOG_FILE, mode="a", header=not file_exists, index=False)

st.set_page_config(page_title="Desalination Energy Analytics", layout="wide")

# ---------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    header {background: transparent;}
    [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
    [data-testid="stDecoration"] {visibility: hidden; height: 0;}
    .block-container {
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

    .page-title {font-size: 26px; font-weight: 700; color: #111827; margin-bottom: 0;}

    /* Restyle native bordered containers (our "cards") to match the mockup */
    div[class*="st-key-card"] > div {
        border-radius: 16px !important;
        border: 1px solid #f0f1f3 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        padding: 18px 20px !important;
        background: #ffffff;
    }

    .kpi-icon {
        width: 42px; height: 42px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; margin-bottom: 12px;
    }
    .kpi-label {color: #6b7280; font-size: 13px; margin-bottom: 4px;}
    .kpi-value {font-size: 26px; font-weight: 700; color: #111827; line-height: 1.2; white-space: nowrap;}
    .kpi-unit {font-size: 14px; font-weight: 500; color: #6b7280;}
    .kpi-delta-up {color: #16a34a; font-size: 13px; font-weight: 600; margin-top: 6px;}
    .kpi-delta-down {color: #dc2626; font-size: 13px; font-weight: 600; margin-top: 6px;}

    .insight-title {font-weight: 700; font-size: 14px; margin-bottom: 4px; color: #111827;}
    .insight-text {font-size: 13px; color: #6b7280; line-height: 1.5;}

    .badge-increasing {
        background: #dcfce7; color: #16a34a; padding: 5px 14px;
        border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-block;
    }
    .badge-decreasing {
        background: #fee2e2; color: #dc2626; padding: 5px 14px;
        border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-block;
    }

    /* Recommendations page: severity + decision badges */
    .rec-badge {
        padding: 3px 10px; border-radius: 20px; font-size: 12px;
        font-weight: 600; display: inline-block; white-space: nowrap;
    }
    .rec-severity-low {background: #dcfce7; color: #16a34a;}
    .rec-severity-medium {background: #fef3c7; color: #b45309;}
    .rec-severity-high {background: #fee2e2; color: #dc2626;}
    .rec-severity-na {background: #f1f5f9; color: #6b7280;}

    .rec-tag-pending {background: #f1f5f9; color: #6b7280;}
    .rec-tag-accepted {background: #dcfce7; color: #16a34a;}
    .rec-tag-rejected {background: #fee2e2; color: #dc2626;}
    .rec-tag-deferred {background: #fef3c7; color: #b45309;}

    .rec-header-row {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 12px;
    }
    .rec-header-left {display: flex; align-items: center; gap: 10px;}
    .rec-date {font-weight: 700; font-size: 15px; color: #111827;}
    .rec-field-label {color: #6b7280; font-size: 12px;}
    .rec-field-value {color: #111827; font-size: 13px; margin-bottom: 6px;}
    .rec-note {color: #9ca3af; font-size: 13px; font-style: italic;}
    div[class*="st-key-rec_btn_stack_"] {max-width: 140px; margin-left: auto;}
    div[class*="st-key-rec_subbox_"] > div {
        background: #f8fafc !important;
        border: 1px solid #eef2f7 !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        height: 100%;
    }

    .stat-value {font-size: 20px; font-weight: 700; color: #111827;}
    .stat-caption {font-size: 11px; color: #9ca3af; margin-top: 2px;}

    .legend-row {display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 13px;}
    .legend-dot {width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;}
    .legend-text {color: #374151;}
    .legend-pct {margin-left: auto; font-weight: 600; color: #111827;}

    div[data-testid="stButton"] > button {
        border-radius: 10px; border: 1px solid #e5e7eb; background: white;
        color: #374151; font-size: 13px; padding: 6px 14px;
    }

    /* Accept / Reject / Defer decision buttons on the Recommendations page */
    div[class*="st-key-accept_"] button {
        background: #16a34a !important; border-color: #16a34a !important;
        color: white !important; font-weight: 600;
    }
    div[class*="st-key-accept_"] button:hover {
        background: #15803d !important; border-color: #15803d !important;
    }
    div[class*="st-key-reject_"] button {
        background: #dc2626 !important; border-color: #dc2626 !important;
        color: white !important; font-weight: 600;
    }
    div[class*="st-key-reject_"] button:hover {
        background: #b91c1c !important; border-color: #b91c1c !important;
    }
    div[class*="st-key-defer_"] button {
        background: #f59e0b !important; border-color: #f59e0b !important;
        color: white !important; font-weight: 600;
    }
    div[class*="st-key-defer_"] button:hover {
        background: #d97706 !important; border-color: #d97706 !important;
    }

    section[data-testid="stSidebar"] {background-color: #0f172a;}
    section[data-testid="stSidebar"] .stMarkdown {color: white;}

    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 12px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #1e293b;
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 10px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #cbd5e1 !important;
        font-size: 12px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #334155;
        color: #f8fafc;
        border: 1px solid #475569;
        border-radius: 6px;
        font-size: 12px;
        padding: 4px 10px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = {
    "blue": "#2563eb", "blue_bg": "#dbeafe",
    "teal": "#0d9488", "teal_bg": "#ccfbf1",
    "purple": "#7c3aed", "purple_bg": "#ede9fe",
    "orange": "#ea580c", "orange_bg": "#ffedd5",
    "green": "#16a34a", "green_bg": "#dcfce7",
    "red": "#dc2626", "red_bg": "#fee2e2",
    "gray": "#6b7280", "gray_bg": "#f1f5f9",
}

_card_counter = {"n": 0}


def card():
    """A bordered container styled as a white card. Use with `with card():`."""
    _card_counter["n"] += 1
    return st.container(border=True, key=f"card_{_card_counter['n']}")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def train_models(df):
    features = ["Water_Production_m3", "Temperature_C", "Seasonal_Demand_Index"]
    features = [f for f in features if f in df.columns]
    target = "Energy_Consumption_kWh"
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    lr = LinearRegression().fit(train[features], train[target])
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(
        train[features], train[target]
    )
    test = test.copy()
    test["LR_Prediction"] = lr.predict(test[features])
    test["RF_Prediction"] = rf.predict(test[features])

    metrics = {}
    for name, col in [("Linear Regression", "LR_Prediction"), ("Random Forest", "RF_Prediction")]:
        metrics[name] = {
            "MAE": mean_absolute_error(test[target], test[col]),
            "RMSE": np.sqrt(mean_squared_error(test[target], test[col])),
            "R2": r2_score(test[target], test[col]),
        }
    return test, metrics


def pct_change(current, previous):
    if previous == 0 or pd.isna(previous):
        return 0.0
    return (current - previous) / previous * 100


def kpi_card(icon, bg, label, value, unit, delta_pct):
    up = delta_pct >= 0
    arrow = "&#8593;" if up else "&#8595;"
    delta_class = "kpi-delta-up" if up else "kpi-delta-down"

    # Shrink the number's font size when it's long (e.g. large water totals
    # like "3,480,011") so it stays on one line instead of wrapping the
    # unit onto its own line and leaving an awkward gap.
    char_len = len(str(value)) + len(unit)
    if char_len <= 9:
        font_size = 26
    elif char_len <= 13:
        font_size = 21
    else:
        font_size = 17

    with card():
        st.markdown(f"""
        <div class="kpi-icon" style="background:{bg};">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="font-size:{font_size}px;">{value} <span class="kpi-unit">{unit}</span></div>
        <div class="{delta_class}">{arrow} {abs(delta_pct):.1f}% vs previous period</div>
        """, unsafe_allow_html=True)


def insight_card(icon, bg, title, text):
    with card():
        st.markdown(f"""
        <div class="kpi-icon" style="background:{bg};">{icon}</div>
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
        """, unsafe_allow_html=True)


def stat_card(icon, bg, label, value, unit, caption=""):
    with card():
        st.markdown(f"""
        <div class="kpi-icon" style="background:{bg}; width:34px; height:34px; font-size:16px;">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="stat-value">{value} <span class="kpi-unit">{unit}</span></div>
        <div class="stat-caption">{caption}</div>
        """, unsafe_allow_html=True)


def styled_line_chart(x, y, y_title, show_labels=False, value_fmt="{:,.0f}"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers" + ("+text" if show_labels else ""),
        line=dict(color=PALETTE["blue"], width=2.5),
        marker=dict(size=6, color=PALETTE["blue"]),
        text=[value_fmt.format(v) for v in y] if show_labels else None,
        textposition="top center", textfont=dict(size=10, color="#374151"),
        name=y_title,
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title=y_title, gridcolor="#f1f5f9", zeroline=False),
        xaxis=dict(gridcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💧 Desalination\n**Energy Analytics**")
    uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")
    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Energy Overview", "Trends", "Comparisons",
                 "Anomaly Detection", "Predictions", "Recommendations",
                 "Reports", "Settings", "Logout"],
        icons=["grid", "lightning-charge", "graph-up-arrow", "bar-chart",
               "exclamation-triangle", "lightbulb", "star", "file-earmark-text",
               "gear", "box-arrow-right"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#0f172a"},
            "icon": {"color": "#94a3b8", "font-size": "15px"},
            "nav-link": {"font-size": "14px", "color": "#cbd5e1",
                         "--hover-color": "#1e293b", "padding": "9px 14px"},
            "nav-link-selected": {"background-color": "#2563eb", "color": "white"},
        },
    )

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={
        "Date/Time": "Date",
        "Energy Consumption (kW)": "Energy_Consumption_kWh",
        "Water Production (L/min)": "Water_Production_m3",
        "Temperature (C)": "Temperature_C",
        "Feed Pressure (psi)": "Feed_Pressure_psi",
        "Feed Conductivity (mS/cm)": "Feed_Conductivity_mScm",
    })
    df["Date"] = pd.to_datetime(df["Date"])

    df = remove_duplicates(df)
    df = format_dates(df)
    df = handle_missing_values(df)
    df = calculate_sec(df)
    if "Feed_Pressure_psi" in df.columns and "Feed_Conductivity_mScm" in df.columns:
        df = calculate_ndp(df)
    df = flag_outliers(df)

    df = anomaly_detection(df)

    df = train_and_predict(df)
else:
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Could not find {DATA_FILE}. Run generate_dataset.py, "
                 "preprocess_data.py, then analyze_data.py first, in that order.")
        st.stop()

df["Energy_Consumption_MWh"] = df["Energy_Consumption_kWh"] / 1000

# ---------------------------------------------------------------------
# Top bar: title, date-range pill (popover), Filter button (popover)
# ---------------------------------------------------------------------
title_col, spacer, date_col, filter_col = st.columns([4, 3, 2.4, 1])

with title_col:
    page_titles = {
        "Dashboard": "Dashboard Overview", "Energy Overview": "Energy Overview",
        "Trends": "Energy Consumption Trends", "Comparisons": "Water Production Comparison",
        "Anomaly Detection": "Anomaly Highlights", "Predictions": "Machine Learning Predictions",
        "Recommendations": "Recommendation Summary", "Reports": "Reports",
        "Settings": "Settings", "Logout": "Logout",
    }
    st.markdown(f'<div class="page-title">{page_titles.get(page, page)}</div>',
                unsafe_allow_html=True)

default_start = df["Date"].max().date() - pd.Timedelta(days=30)
default_end = df["Date"].max().date()
data_min = df["Date"].min().date()
data_max = df["Date"].max().date()

_current_range = st.session_state.get("date_range_input")
_valid = (
    isinstance(_current_range, tuple) and len(_current_range) == 2
    and data_min <= _current_range[0] <= data_max
    and data_min <= _current_range[1] <= data_max
)
if not _valid:
    st.session_state["date_range_input"] = (default_start, default_end)

label_start, label_end = st.session_state["date_range_input"]

with date_col:
    with st.popover(f"📅 {label_start.strftime('%d %b %Y')} – {label_end.strftime('%d %b %Y')}",
                     use_container_width=True):
        date_range = st.date_input(
            "Select range",
            min_value=data_min, max_value=data_max,
            label_visibility="collapsed", key="date_range_input",
        )

with filter_col:
    with st.popover("🔽 Filter", use_container_width=True):
        if "Operating_Period" in df.columns:
            op_filter = st.multiselect(
                "Operating Period", options=sorted(df["Operating_Period"].dropna().unique()),
                default=list(sorted(df["Operating_Period"].dropna().unique())),
            )

if not isinstance(date_range, tuple) or len(date_range) != 2:
    date_range = (default_start, default_end)
start, end = date_range

mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
if "Operating_Period" in df.columns and "op_filter" in dir() and op_filter:
    mask &= df["Operating_Period"].isin(op_filter)
df_filtered = df.loc[mask].reset_index(drop=True)

period_len = (end - start).days + 1
prev_end = start - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_len - 1)
prev_mask = (df["Date"].dt.date >= prev_start) & (df["Date"].dt.date <= prev_end)
df_prev = df.loc[prev_mask]

st.write("")

# ---------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------
if page == "Dashboard":
    total_energy = df_filtered["Energy_Consumption_MWh"].sum()
    total_water = df_filtered["Water_Production_m3"].sum()
    avg_sec = df_filtered["Specific_Energy_Consumption_kWh_per_m3"].mean()
    avg_temp = df_filtered["Temperature_C"].mean()

    prev_energy = df_prev["Energy_Consumption_MWh"].sum()
    prev_water = df_prev["Water_Production_m3"].sum()
    prev_sec = df_prev["Specific_Energy_Consumption_kWh_per_m3"].mean()
    prev_temp = df_prev["Temperature_C"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("&#9889;", PALETTE["blue_bg"], "Total Energy Consumption",
                  f"{total_energy:,.0f}", "MWh", pct_change(total_energy, prev_energy))
    with c2:
        kpi_card("&#128167;", PALETTE["teal_bg"], "Total Water Production",
                  f"{total_water:,.0f}", "m³", pct_change(total_water, prev_water))
    with c3:
        kpi_card("&#9878;&#65039;", PALETTE["purple_bg"], "Specific Energy Consumption",
                  f"{avg_sec:.2f}", "kWh/m³", pct_change(avg_sec, prev_sec))
    with c4:
        kpi_card("&#9728;&#65039;", PALETTE["orange_bg"], "Avg Temperature",
                  f"{avg_temp:.1f}", "°C", pct_change(avg_temp, prev_temp))

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        with card():
            st.markdown("**Energy Consumption Trend**")
            daily = df_filtered.groupby(df_filtered["Date"].dt.date)["Energy_Consumption_MWh"].sum()
            fig = styled_line_chart(daily.index, daily.values, "MWh")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        with card():
            st.markdown("**Energy by Process**")
            st.caption("Estimated split (no per-process metering in this dataset)")

            proc_labels = ["RO Pumping", "Pre-treatment", "Post-treatment", "Others"]
            proc_pct = [0.55, 0.20, 0.15, 0.10]
            proc_colors = [PALETTE["blue"], "#93c5fd", PALETTE["green"], "#cbd5e1"]
            proc_values = [total_energy * p for p in proc_pct]

            fig = go.Figure(data=[go.Pie(
                labels=proc_labels, values=proc_pct, hole=0.62,
                marker=dict(colors=proc_colors), sort=False, textinfo="none",
            )])
            fig.update_layout(showlegend=False, height=180,
                               margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            for lbl, pct, col, val in zip(proc_labels, proc_pct, proc_colors, proc_values):
                st.markdown(f"""
                <div class="legend-row">
                    <span class="legend-dot" style="background:{col};"></span>
                    <span class="legend-text">{lbl}</span>
                    <span class="legend-pct">{pct*100:.0f}% ({val:,.0f} MWh)</span>
                </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("**Key Insights**")
    i1, i2, i3 = st.columns(3)
    energy_delta = pct_change(total_energy, prev_energy)
    sec_delta = pct_change(avg_sec, prev_sec)
    temp_delta = pct_change(avg_temp, prev_temp)

    with i1:
        direction = "Increased" if energy_delta >= 0 else "Decreased"
        insight_card("&#128200;", PALETTE["blue_bg"], f"Energy Consumption {direction}",
                      f"Total energy consumption {direction.lower()} by {abs(energy_delta):.1f}% "
                      f"compared to the previous period.")
    with i2:
        title = "Increased Specific Energy" if sec_delta >= 0 else "Improved Specific Energy"
        insight_card("&#128167;", PALETTE["teal_bg"], title,
                      f"Specific energy consumption {'increased' if sec_delta >= 0 else 'decreased'} "
                      f"by {abs(sec_delta):.1f}%, indicating "
                      f"{'reduced' if sec_delta >= 0 else 'improved'} operational efficiency.")
    with i3:
        insight_card("&#127777;", PALETTE["orange_bg"], "Temperature Conditions",
                      f"Average temperature changed by {abs(temp_delta):.1f}% vs the "
                      f"previous period. "
                      f"{'Monitor for impact on plant performance.' if abs(temp_delta) > 10 else 'No significant impact expected on plant performance.'}")

# ---------------------------------------------------------------------
# Energy Overview page
# ---------------------------------------------------------------------
elif page == "Energy Overview":
    total_energy = df_filtered["Energy_Consumption_MWh"].sum()
    prev_energy = df_prev["Energy_Consumption_MWh"].sum()
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("&#9889;", PALETTE["blue_bg"], "Total Energy Consumption",
                  f"{total_energy:,.0f}", "MWh", pct_change(total_energy, prev_energy))
    with c2:
        kpi_card("&#128200;", PALETTE["green_bg"], "Peak Day",
                  f"{df_filtered['Energy_Consumption_MWh'].max():,.0f}", "MWh", 0)
    with c3:
        kpi_card("&#128201;", PALETTE["purple_bg"], "Lowest Day",
                  f"{df_filtered['Energy_Consumption_MWh'].min():,.0f}", "MWh", 0)

    st.write("")
    with card():
        st.markdown("**Daily Energy Consumption**")
        daily = df_filtered.groupby(df_filtered["Date"].dt.date)["Energy_Consumption_MWh"].sum()
        fig = styled_line_chart(daily.index, daily.values, "MWh")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------
# Trends page
# ---------------------------------------------------------------------
elif page == "Trends":
    tab1, tab2, tab3 = st.tabs(["Energy Consumption", "SEC", "Temperature"])

    def trend_tab(daily_series, unit, label):
        with card():
            st.markdown(f"**Daily {label}**")
            show_labels = len(daily_series) <= 35
            fig = styled_line_chart(daily_series.index, daily_series.values, unit,
                                     show_labels=show_labels,
                                     value_fmt="{:,.0f}" if unit != "°C" else "{:,.1f}")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.write("")
        stats_col, summary_col = st.columns([2.4, 1.6])

        vals = daily_series.values
        dates = daily_series.index
        max_i, min_i = int(np.argmax(vals)), int(np.argmin(vals))

        with stats_col:
            st.markdown("**Daily Statistics**")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                stat_card("&#128200;", PALETTE["blue_bg"], "Average",
                          f"{vals.mean():,.0f}" if unit != "°C" else f"{vals.mean():,.1f}", unit)
            with s2:
                stat_card("&#11014;&#65039;", PALETTE["green_bg"], "Max",
                          f"{vals[max_i]:,.0f}" if unit != "°C" else f"{vals[max_i]:,.1f}", unit,
                          caption=f"{pd.Timestamp(dates[max_i]):%d %b %Y}")
            with s3:
                stat_card("&#11015;&#65039;", PALETTE["purple_bg"], "Min",
                          f"{vals[min_i]:,.0f}" if unit != "°C" else f"{vals[min_i]:,.1f}", unit,
                          caption=f"{pd.Timestamp(dates[min_i]):%d %b %Y}")
            with s4:
                stat_card("&#9888;&#65039;", PALETTE["orange_bg"], "Std Dev",
                          f"{vals.std():,.0f}" if unit != "°C" else f"{vals.std():,.1f}", unit)

        with summary_col:
            change = vals[-1] - vals[0] if len(vals) > 1 else 0
            pct = pct_change(vals[-1], vals[0]) if len(vals) > 1 else 0
            badge_class = "badge-increasing" if change >= 0 else "badge-decreasing"
            badge_text = "Increasing" if change >= 0 else "Decreasing"
            fmt = "{:,.0f}" if unit != "°C" else "{:,.1f}"
            with card():
                st.markdown("**Trend Summary**")
                st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="insight-text" style="margin-top:10px;">
                {label} shows an overall {badge_text.lower()} trend during the selected period,
                {'increasing' if change >= 0 else 'decreasing'} by {fmt.format(abs(change))} {unit}
                ({abs(pct):.1f}%) from {fmt.format(vals[0])} {unit} on
                {pd.Timestamp(dates[0]):%d %b %Y} to {fmt.format(vals[-1])} {unit} on
                {pd.Timestamp(dates[-1]):%d %b %Y}.
                </div>""", unsafe_allow_html=True)

    with tab1:
        s = df_filtered.groupby(df_filtered["Date"].dt.date)["Energy_Consumption_MWh"].sum()
        trend_tab(s, "MWh", "Energy Consumption")
    with tab2:
        s = df_filtered.groupby(df_filtered["Date"].dt.date)["Specific_Energy_Consumption_kWh_per_m3"].mean()
        trend_tab(s, "kWh/m³", "SEC")
    with tab3:
        s = df_filtered.groupby(df_filtered["Date"].dt.date)["Temperature_C"].mean()
        trend_tab(s, "°C", "Temperature")

# ---------------------------------------------------------------------
# Comparisons page
# ---------------------------------------------------------------------
elif page == "Comparisons":
    with card():
        st.markdown("**Energy vs Water Production**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["Energy_Consumption_MWh"],
                                  name="Energy (MWh)", yaxis="y1", line=dict(color=PALETTE["blue"])))
        fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["Water_Production_m3"],
                                  name="Water Production (m³)", yaxis="y2", line=dict(color=PALETTE["teal"])))
        fig.update_layout(
            yaxis=dict(title="Energy (MWh)", gridcolor="#f1f5f9"),
            yaxis2=dict(title="Water Production (m³)", overlaying="y", side="right"),
            height=350, legend=dict(orientation="h", y=1.1),
            plot_bgcolor="white", margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    with card():
        st.markdown("**Correlation Between Variables**")
        cols = ["Energy_Consumption_MWh", "Water_Production_m3", "Temperature_C", "Seasonal_Demand_Index"]
        cols = [c for c in cols if c in df_filtered.columns]
        corr = df_filtered[cols].corr()
        fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=cols, y=cols,
                                          colorscale="RdBu", zmid=0, text=corr.round(2).values,
                                          texttemplate="%{text}"))
        fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------
# Anomaly Detection page
# ---------------------------------------------------------------------
elif page == "Anomaly Detection":
    if "IsoForest_Anomaly" not in df_filtered.columns:
        st.warning("No anomaly flags found. Run analyze_data.py first.")
    else:
        flagged = df_filtered[df_filtered["IsoForest_Anomaly"] == 1]
        c1, _, _, _ = st.columns(4)
        with c1:
            kpi_card("&#9888;&#65039;", PALETTE["red_bg"], "Anomalies Flagged",
                      str(len(flagged)), "", 0)

        st.write("")
        with card():
            st.markdown("**Energy Consumption with Flagged Anomalies**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["Energy_Consumption_MWh"],
                                      name="Energy Consumption", line=dict(color=PALETTE["blue"])))
            fig.add_trace(go.Scatter(x=flagged["Date"], y=flagged["Energy_Consumption_MWh"],
                                      mode="markers", name="Flagged Anomaly",
                                      marker=dict(color=PALETTE["red"], size=10)))
            fig.update_layout(height=350, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.write("")
        with card():
            st.markdown("**Flagged Dates**")
            st.dataframe(flagged[["Date", "Energy_Consumption_MWh", "Water_Production_m3",
                                   "Specific_Energy_Consumption_kWh_per_m3"]], use_container_width=True)

# ---------------------------------------------------------------------
# Predictions page
# ---------------------------------------------------------------------
elif page == "Predictions":
    test, metrics = train_models(df)
    test["Energy_Consumption_MWh"] = test["Energy_Consumption_kWh"] / 1000
    test["LR_MWh"] = test["LR_Prediction"] / 1000
    test["RF_MWh"] = test["RF_Prediction"] / 1000

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("&#128202;", PALETTE["blue_bg"], "Linear Regression R²",
                  f"{metrics['Linear Regression']['R2']:.3f}", "", 0)
    with c2:
        kpi_card("&#127795;", PALETTE["teal_bg"], "Random Forest R²",
                  f"{metrics['Random Forest']['R2']:.3f}", "", 0)

    st.write("")
    with card():
        st.markdown("**Predicted vs Actual Energy Consumption**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test["Date"], y=test["Energy_Consumption_MWh"],
                                  name="Actual", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=test["Date"], y=test["LR_MWh"],
                                  name="Linear Regression", line=dict(color=PALETTE["blue"])))
        fig.add_trace(go.Scatter(x=test["Date"], y=test["RF_MWh"],
                                  name="Random Forest", line=dict(color=PALETTE["green"])))
        fig.update_layout(height=350, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    with card():
        st.markdown("**Largest Prediction Deviations (possible inefficiencies)**")
        test["RF_Deviation"] = (test["Energy_Consumption_MWh"] - test["RF_MWh"]).abs()
        st.dataframe(test.nlargest(10, "RF_Deviation")[
            ["Date", "Energy_Consumption_MWh", "RF_MWh", "RF_Deviation"]
        ], use_container_width=True)

# ---------------------------------------------------------------------
# Recommendations page
# ---------------------------------------------------------------------
elif page == "Recommendations":
    decision_log_df = load_decision_log()

    avg_sec = df_filtered["Specific_Energy_Consumption_kWh_per_m3"].mean()
    recent_sec = df_filtered["Specific_Energy_Consumption_kWh_per_m3"].iloc[-14:].mean()

    if "LR_Prediction" in df_filtered.columns:
        pred_col = "LR_Prediction"
    elif "RF_Prediction" in df_filtered.columns:
        pred_col = "RF_Prediction"
    else:
        pred_col = None

    # --- Filter bar: date range (from the global picker), live deviation
    # threshold, and sort order. Filtering/sorting only — no calculations
    # change here. ---
    filt_col1, filt_col2, filt_col3 = st.columns([2, 2, 2])
    with filt_col1:
        st.caption("Date range")
        st.markdown(f"📅 {start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}")
    with filt_col2:
        min_dev = st.slider("Minimum deviation %", min_value=0, max_value=30,
                             value=MIN_FLAG_DEVIATION_PCT, key="rec_min_dev")
    with filt_col3:
        sort_choice = st.radio(
            "Sort", ["Most severe first", "Most recent first"],
            index=0, horizontal=True, key="rec_sort_choice",
        )

    if "IsoForest_Anomaly" in df_filtered.columns:
        candidates = df_filtered[df_filtered["IsoForest_Anomaly"] == 1].copy()
    else:
        candidates = df_filtered.iloc[0:0].copy()

    # Only surface days whose deviation from expected energy clears the
    # noise floor set by the slider — small misses aren't worth an
    # operator's time. This is a display/filter step only: the underlying
    # prediction, anomaly-detection, and physics-check values are untouched.
    if pred_col is not None and len(candidates) > 0:
        candidates["Deviation_Pct"] = candidates.apply(
            lambda r: pct_change(r["Energy_Consumption_kWh"], r[pred_col]), axis=1
        )
        flagged = candidates[candidates["Deviation_Pct"].abs() > min_dev].copy()
    else:
        # No prediction column to measure deviation against — fall back to
        # showing the raw anomaly flags rather than silently hiding them.
        flagged = candidates.copy()

    if sort_choice == "Most severe first" and "Deviation_Pct" in flagged.columns:
        flagged = flagged.reindex(flagged["Deviation_Pct"].abs().sort_values(ascending=False).index)
    else:
        flagged = flagged.sort_values("Date", ascending=False)
    flagged = flagged.head(15)

    n_anomalies = len(flagged)

    with card():
        st.markdown("**Recommendation Summary**")
        if recent_sec > avg_sec * 1.05:
            st.write(f"- Recent SEC ({recent_sec:.2f} kWh/m³) is running above the period "
                     f"average ({avg_sec:.2f} kWh/m³) — possible causes include membrane "
                     f"fouling or pump inefficiency; recommend operator inspection.")
        if n_anomalies > 0:
            st.write(f"- {n_anomalies} day(s) with energy use deviating more than "
                     f"{min_dev}% from expected — review the Anomaly Detection "
                     f"page for possible equipment issues.")

    with st.expander("View Decision Log"):
        if len(decision_log_df) > 0:
            st.dataframe(decision_log_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No decisions logged yet.")

    if len(flagged) == 0:
        with card():
            st.write("- No efficiency concerns detected in the selected period.")
    else:
        st.write("")
        st.markdown("**Flagged Anomalies for Review**")

        context_specs = [
            ("Temperature_C", "Temperature", "°C"),
            ("Feed_Pressure_psi", "Feed Pressure", "psi"),
            ("Feed_Conductivity_mScm", "Feed Conductivity", "mS/cm"),
        ]
        context_cols = [c for c in context_specs if c[0] in df_filtered.columns]

        for idx, row in flagged.iterrows():
            decision_key = f"rec_decision_{idx}"
            outcome_key = f"rec_outcome_{idx}"
            date_str = row["Date"].strftime("%Y-%m-%d")

            # Seed the on-page status from previously logged decisions so a
            # fresh session/reload doesn't reset everything back to Pending.
            if decision_key not in st.session_state:
                prior = decision_log_df[decision_log_df["Date"] == date_str]
                if len(prior) > 0:
                    st.session_state[decision_key] = prior.iloc[-1]["Decision"]

            deviation_pct = row["Deviation_Pct"] if "Deviation_Pct" in row.index else None

            if deviation_pct is None:
                severity_class, severity_label = "rec-severity-na", "N/A"
            elif abs(deviation_pct) < 8:
                severity_class, severity_label = "rec-severity-low", "Low"
            elif abs(deviation_pct) <= 15:
                severity_class, severity_label = "rec-severity-medium", "Medium"
            else:
                severity_class, severity_label = "rec-severity-high", "High"
            severity_text = (f"{severity_label} · {abs(deviation_pct):.1f}%"
                              if deviation_pct is not None else severity_label)

            decision = st.session_state.get(decision_key, "Pending")
            outcome_status = st.session_state.get(outcome_key, "Predicted")
            tag_class = {
                "Pending": "rec-tag-pending", "Accepted": "rec-tag-accepted",
                "Rejected": "rec-tag-rejected", "Deferred": "rec-tag-deferred",
            }[decision]
            tag_text = decision if decision != "Accepted" else f"Accepted · {outcome_status}"

            with card():
                st.markdown(f"""
                <div class="rec-header-row">
                    <div class="rec-header-left">
                        <span class="rec-date">{row['Date'].strftime('%d %b %Y')}</span>
                        <span class="rec-badge {severity_class}">{severity_text}</span>
                    </div>
                    <span class="rec-badge {tag_class}">{tag_text}</span>
                </div>
                """, unsafe_allow_html=True)

                left, right = st.columns(2)
                with left:
                    with st.container(border=True, key=f"rec_subbox_left_{idx}"):
                        if pred_col is not None:
                            actual = row["Energy_Consumption_kWh"]
                            expected = row[pred_col]
                            st.markdown('<div class="rec-field-label">Actual energy</div>'
                                        f'<div class="rec-field-value">{actual:,.0f} kWh</div>',
                                        unsafe_allow_html=True)
                            st.markdown(f'<div class="rec-field-label">Expected ({pred_col})</div>'
                                        f'<div class="rec-field-value">{expected:,.0f} kWh</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="rec-field-value">Actual/Expected energy '
                                         'unavailable (no prediction columns).</div>',
                                         unsafe_allow_html=True)
                        st.markdown('<div class="rec-field-label">Specific Energy Consumption</div>'
                                    f'<div class="rec-field-value">'
                                    f'{row["Specific_Energy_Consumption_kWh_per_m3"]:.2f} kWh/m³</div>',
                                    unsafe_allow_html=True)

                with right:
                    with st.container(border=True, key=f"rec_subbox_right_{idx}"):
                        if context_cols:
                            st.markdown('<div class="rec-field-label">Operating conditions</div>',
                                        unsafe_allow_html=True)
                            for col, label, unit in context_cols:
                                st.markdown(f'<div class="rec-field-value">{label}: '
                                            f'{row[col]:.1f} {unit}</div>', unsafe_allow_html=True)
                        if "Physics_Flag" in df_filtered.columns:
                            physics_text = ("implausible reading" if row["Physics_Flag"] == 1
                                             else "plausible")
                            st.markdown('<div class="rec-field-label">Physics check</div>'
                                        f'<div class="rec-field-value">{physics_text}</div>',
                                        unsafe_allow_html=True)

                rec_col, btn_col = st.columns([5, 1])
                with rec_col:
                    if deviation_pct is not None:
                        direction = "above" if deviation_pct >= 0 else "below"
                        if deviation_pct >= 0:
                            causes = ("possible causes include membrane fouling, pump "
                                      "inefficiency, or elevated feed temperature")
                        else:
                            causes = ("possible causes include a sensor/meter fault, reduced "
                                      "water output, or a bypassed treatment stage — verify "
                                      "production volume before assuming efficiency gain")
                        st.write(f"Recommendation: Energy use was {abs(deviation_pct):.1f}% "
                                 f"{direction} expected for these conditions — {causes}; "
                                 f"recommend operator inspection.")
                    else:
                        st.markdown('<div class="rec-note">Recommendation unavailable — no '
                                     'prediction data for this period.</div>',
                                     unsafe_allow_html=True)
                with btn_col:
                    with st.container(key=f"rec_btn_stack_{idx}"):
                        if st.button("Accept", key=f"accept_{idx}", use_container_width=True):
                            st.session_state[decision_key] = "Accepted"
                            log_decision(date_str, deviation_pct, "Accepted")
                        if st.button("Reject", key=f"reject_{idx}", use_container_width=True):
                            st.session_state[decision_key] = "Rejected"
                            log_decision(date_str, deviation_pct, "Rejected")
                        if st.button("Defer", key=f"defer_{idx}", use_container_width=True):
                            st.session_state[decision_key] = "Deferred"
                            log_decision(date_str, deviation_pct, "Deferred")

                if decision == "Accepted":
                    st.selectbox(
                        "Outcome status",
                        ["Predicted", "Observed", "Estimated", "Verified", "Inconclusive"],
                        key=outcome_key,
                    )

# ---------------------------------------------------------------------
# Reports page
# ---------------------------------------------------------------------
elif page == "Reports":
    with card():
        st.markdown("**Export Data**")
        st.write("Download the currently filtered dataset for your report appendix.")
        st.download_button(
            "Download filtered data as CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="aquasense_filtered_export.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------
elif page == "Settings":
    with card():
        st.write("**Data source:**", DATA_FILE)
        st.write("**Rows loaded:**", len(df))
        st.write("**Date range:**", df["Date"].min().date(), "to", df["Date"].max().date())

# ---------------------------------------------------------------------
# Logout page
# ---------------------------------------------------------------------
elif page == "Logout":
    with card():
        st.write("You have been logged out. (Demo dashboard — no authentication is implemented.)")