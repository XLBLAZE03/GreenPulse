import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="INEOS Equipment Carbon Calculator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #f6faf7;
        color: #102A1E;
    }

    h1, h2, h3 {
        color: #0f5132 !important;
        font-weight: 800 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d8e6dc;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #d8e6dc;
        border-left: 6px solid #0f5132;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }

    .input-card {
        background: #ffffff;
        border: 1px solid #d8e6dc;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .small-muted {
        color: #66736b;
        font-size: 0.9rem;
    }

    div.stButton > button {
        background-color: #0f5132;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 700;
    }

    div.stDownloadButton > button {
        background-color: #0f5132;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_INEOS_FILE = os.path.join(
    DATA_DIR,
    "INEOS_Daily_CO2_Dataset_2010_2024_updated.xlsx"
)

DEFAULT_EQUIPMENT_FILE = os.path.join(
    DATA_DIR,
    "equipment_degradation.csv"
)


# -------------------------------------------------
# Internal formula reference
# This is used if data/equipment_degradation.csv is absent.
# Users do NOT need to upload this file.
# -------------------------------------------------
FALLBACK_REFERENCE = pd.DataFrame([
    {
        "Equipment": "Furnace",
        "Key factors to track": "Fuel/air ratio; flame stability; radiant tube temperature; refractory condition; excess O₂; NOx",
        "Common failure modes": "Burner fouling; refractory spalling; tube sagging; flame instability",
        "Monitoring data": "Stack O₂/CO/NOx; flame scanner; tube metal temps; fuel flow; furnace pressure; thermal images",
        "Degradation rate formula": "k = -(1 / U0) × dU/dt, where U = effective heat flux"
    },
    {
        "Equipment": "Boiler",
        "Key factors to track": "Steam pressure/temperature; blowdown rate; feedwater chemistry; flue gas O₂/CO; soot load",
        "Common failure modes": "Soot/fouling; water-side scaling; tube corrosion; low combustion efficiency",
        "Monitoring data": "Stack O₂/CO; flue gas temp; steam flow/pressure; conductivity; blowdown volume; economizer ΔT",
        "Degradation rate formula": "k = -(1 / η0) × dη/dt, where η = boiler thermal efficiency"
    },
    {
        "Equipment": "Reformer",
        "Key factors to track": "Catalyst activity; tube metal temp; pressure drop across bed; feed composition; steam/carbon ratio",
        "Common failure modes": "Catalyst coking/deactivation; tube creep; hot spots; feed poisoning",
        "Monitoring data": "Outlet composition; bed ΔP; tube metal temps; feed impurities; furnace fuel flow",
        "Degradation rate formula": "k = -ln(A(t)/A0) / t, where A = catalyst activity"
    }
])


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def read_ineos_file(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)

    if os.path.exists(DEFAULT_INEOS_FILE):
        return pd.read_excel(DEFAULT_INEOS_FILE)

    return None


def load_equipment_reference():
    """
    Formula CSV is loaded internally from data/equipment_degradation.csv.
    It is not uploaded by the user.
    """
    if os.path.exists(DEFAULT_EQUIPMENT_FILE):
        try:
            return clean_columns(pd.read_csv(DEFAULT_EQUIPMENT_FILE))
        except Exception:
            return FALLBACK_REFERENCE.copy()

    return FALLBACK_REFERENCE.copy()


def detect_required_columns(df: pd.DataFrame):
    required = [
        "Date",
        "Production_Index",
        "Natural_Gas_GJ",
        "KWh_MWh",
        "Steam_tonnes",
        "CO2_Emissions_tCO2e",
        "Total_GHG_tCO2e",
        "boiler",
        "furance",
        "reformer",
        "other_Equpiments",
        "boiler_energy KWh",
        "reforemer_energy KWh",
        "furnace_energy KWh",
        "other_equipment_KWh_KWh",
        "Operating_Hours",
    ]

    existing = set(df.columns)
    return [c for c in required if c not in existing]


def status_from_degradation(value):
    if pd.isna(value):
        return "Unknown"

    if value < 0:
        return "Improving / efficiency gain"

    if value <= 3:
        return "Low degradation"

    if value <= 8:
        return "Moderate degradation"

    return "High degradation risk"


def risk_color(status):
    if "High" in status:
        return "#b42318"
    if "Moderate" in status:
        return "#b54708"
    if "Low" in status:
        return "#0f5132"
    return "#475467"


def load_formula_reference(reference_df: pd.DataFrame, equipment_name: str, field: str):
    if reference_df is None or reference_df.empty:
        return "Not available"

    if "Equipment" not in reference_df.columns:
        return "Not available"

    ref = reference_df.copy()
    ref["Equipment_clean"] = ref["Equipment"].astype(str).str.strip().str.lower()

    row = ref[ref["Equipment_clean"] == equipment_name.lower()]

    if row.empty:
        return "Not available"

    return str(row.iloc[0].get(field, "Not available"))


def linear_degradation_from_user(current_value, baseline_value, total_days):
    """
    Used for:
    Boiler: k = -(1 / η0) × dη/dt
    Furnace: k = -(1 / U0) × dU/dt
    """
    if baseline_value == 0 or total_days <= 0:
        return np.nan

    k_per_day = -((current_value - baseline_value) / total_days) / baseline_value
    return k_per_day * 365 * 100


def log_degradation_from_user(current_value, baseline_value, total_days):
    """
    Used for:
    Reformer: k = -ln(A(t)/A0) / t
    """
    if baseline_value <= 0 or current_value <= 0 or total_days <= 0:
        return np.nan

    k_per_day = -np.log(current_value / baseline_value) / total_days
    return k_per_day * 365 * 100


# -------------------------------------------------
# Core Calculation
# -------------------------------------------------
def calculate_equipment_degradation(
    raw_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    baseline_days: int,
    user_eta: float,
    user_u: float,
    user_a: float,
):
    df = clean_columns(raw_df)

    df["timestamp"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [
        "Production_Index",
        "Natural_Gas_GJ",
        "KWh_MWh",
        "Steam_tonnes",
        "CO2_Emissions_tCO2e",
        "Total_GHG_tCO2e",
        "boiler",
        "furance",
        "reformer",
        "other_Equpiments",
        "boiler_energy KWh",
        "reforemer_energy KWh",
        "furnace_energy KWh",
        "other_equipment_KWh_KWh",
        "Operating_Hours",
    ]

    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    # Safe denominator columns
    df["Operating_Hours_safe"] = df["Operating_Hours"].replace(0, np.nan)
    df["boiler_energy_safe"] = df["boiler_energy KWh"].replace(0, np.nan)
    df["furnace_energy_safe"] = df["furnace_energy KWh"].replace(0, np.nan)
    df["reformer_energy_safe"] = df["reforemer_energy KWh"].replace(0, np.nan)

    # Dataset-derived proxy signals
    df["boiler_efficiency_proxy"] = df["Steam_tonnes"] / df["boiler_energy_safe"]
    df["furnace_heat_flux_proxy"] = df["furnace_energy KWh"] / df["Operating_Hours_safe"]
    df["reformer_activity_proxy"] = df["Production_Index"] / df["reformer_energy_safe"]

    proxy_cols = [
        "boiler_efficiency_proxy",
        "furnace_heat_flux_proxy",
        "reformer_activity_proxy",
    ]

    for col in proxy_cols:
        df[col] = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .interpolate()
            .ffill()
            .bfill()
        )

    baseline_window = max(1, min(baseline_days, len(df)))

    boiler_eta0 = float(df["boiler_efficiency_proxy"].head(baseline_window).mean())
    furnace_u0 = float(df["furnace_heat_flux_proxy"].head(baseline_window).mean())
    reformer_a0 = float(df["reformer_activity_proxy"].head(baseline_window).mean())

    dt = df["timestamp"].diff().dt.days.fillna(1).replace(0, 1)

    # Dataset-based historical degradation trend
    df["boiler_k_per_day"] = -(
        df["boiler_efficiency_proxy"].diff().fillna(0) / dt
    ) / boiler_eta0

    df["furnace_k_per_day"] = -(
        df["furnace_heat_flux_proxy"].diff().fillna(0) / dt
    ) / furnace_u0

    t_days = (df["timestamp"] - df["timestamp"].min()).dt.days
    t_days_safe = t_days.replace(0, np.nan)

    df["reformer_k_per_day"] = -np.log(
        df["reformer_activity_proxy"] / reformer_a0
    ) / t_days_safe

    df["reformer_k_per_day"] = (
        df["reformer_k_per_day"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["boiler_degradation_pct_year"] = df["boiler_k_per_day"] * 365 * 100
    df["furnace_degradation_pct_year"] = df["furnace_k_per_day"] * 365 * 100
    df["reformer_degradation_pct_year"] = df["reformer_k_per_day"] * 365 * 100

    # Equipment emission intensity
    df["boiler_emission_intensity"] = df["boiler"] / df["boiler_energy_safe"]
    df["furnace_emission_intensity"] = df["furance"] / df["furnace_energy_safe"]
    df["reformer_emission_intensity"] = df["reformer"] / df["reformer_energy_safe"]

    # Equipment total emissions and energy
    df["equipment_total_emissions_tCO2e"] = (
        df["boiler"]
        + df["furance"]
        + df["reformer"]
        + df["other_Equpiments"]
    )

    df["equipment_total_energy_kwh"] = (
        df["boiler_energy KWh"]
        + df["furnace_energy KWh"]
        + df["reforemer_energy KWh"]
        + df["other_equipment_KWh_KWh"]
    )

    total_days = max(1, int((df["timestamp"].max() - df["timestamp"].min()).days))

    # User-input-based degradation calculation
    user_boiler_deg = linear_degradation_from_user(user_eta, boiler_eta0, total_days)
    user_furnace_deg = linear_degradation_from_user(user_u, furnace_u0, total_days)
    user_reformer_deg = log_degradation_from_user(user_a, reformer_a0, total_days)

    recent = df.tail(min(365, len(df)))

    summary_config = [
        {
            "Equipment": "Boiler",
            "Parameter": "η / n - Boiler thermal efficiency",
            "User Input": f"η = {user_eta:.6f}",
            "User Input Degradation % / Year": user_boiler_deg,
            "Emission Column": "boiler",
            "Energy Column": "boiler_energy KWh",
            "Proxy Column": "boiler_efficiency_proxy",
            "Degradation Column": "boiler_degradation_pct_year",
            "Proxy Used": "Steam_tonnes / boiler_energy KWh",
            "Baseline Proxy": boiler_eta0,
            "Current Proxy": user_eta,
        },
        {
            "Equipment": "Furnace",
            "Parameter": "U - Effective heat flux",
            "User Input": f"U = {user_u:.6f}",
            "User Input Degradation % / Year": user_furnace_deg,
            "Emission Column": "furance",
            "Energy Column": "furnace_energy KWh",
            "Proxy Column": "furnace_heat_flux_proxy",
            "Degradation Column": "furnace_degradation_pct_year",
            "Proxy Used": "furnace_energy KWh / Operating_Hours",
            "Baseline Proxy": furnace_u0,
            "Current Proxy": user_u,
        },
        {
            "Equipment": "Reformer",
            "Parameter": "A - Catalyst activity",
            "User Input": f"A = {user_a:.6f}",
            "User Input Degradation % / Year": user_reformer_deg,
            "Emission Column": "reformer",
            "Energy Column": "reforemer_energy KWh",
            "Proxy Column": "reformer_activity_proxy",
            "Degradation Column": "reformer_degradation_pct_year",
            "Proxy Used": "Production_Index / reforemer_energy KWh",
            "Baseline Proxy": reformer_a0,
            "Current Proxy": user_a,
        },
    ]

    rows = []

    for item in summary_config:
        equipment = item["Equipment"]
        degradation_col = item["Degradation Column"]
        emission_col = item["Emission Column"]
        energy_col = item["Energy Column"]

        avg_degradation = float(recent[degradation_col].mean())
        latest_dataset_degradation = float(df[degradation_col].iloc[-1])
        user_deg = item["User Input Degradation % / Year"]

        rows.append({
            "Equipment": equipment,
            "Parameter": item["Parameter"],
            "User Input": item["User Input"],
            "Proxy Used": item["Proxy Used"],
            "Baseline Proxy": round(float(item["Baseline Proxy"]), 6),
            "Current User Proxy": round(float(item["Current Proxy"]), 6),
            "Dataset Latest Degradation % / Year": round(latest_dataset_degradation, 3),
            "Dataset Avg Degradation % / Year": round(avg_degradation, 3),
            "User Input Degradation % / Year": round(float(user_deg), 3) if not pd.isna(user_deg) else np.nan,
            "Latest Emissions tCO2e": round(float(df[emission_col].iloc[-1]), 3),
            "Total Emissions tCO2e": round(float(df[emission_col].sum()), 3),
            "Latest Energy kWh": round(float(df[energy_col].iloc[-1]), 3),
            "Total Energy kWh": round(float(df[energy_col].sum()), 3),
            "Status": status_from_degradation(user_deg),
            "Formula": load_formula_reference(reference_df, equipment, "Degradation rate formula"),
            "Key Factors": load_formula_reference(reference_df, equipment, "Key factors to track"),
            "Common Failure Modes": load_formula_reference(reference_df, equipment, "Common failure modes"),
            "Monitoring Data": load_formula_reference(reference_df, equipment, "Monitoring data"),
        })

    return df, pd.DataFrame(rows)


# -------------------------------------------------
# Graph Filtering
# -------------------------------------------------
WEEKDAY_MAP = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

ORDINAL_MAP = {
    "1st": 1, "first": 1,
    "2nd": 2, "second": 2,
    "3rd": 3, "third": 3,
    "4th": 4, "fourth": 4,
    "5th": 5, "fifth": 5,
}


def add_date_features(df):
    df = df.copy()
    df["weekday_num"] = df["timestamp"].dt.weekday
    df["weekday_name"] = df["timestamp"].dt.day_name()
    df["month_num"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.month_name()
    df["nth_weekday_of_month"] = ((df["timestamp"].dt.day - 1) // 7) + 1
    return df


def apply_prompt_filter(df, prompt):
    filtered = add_date_features(df)
    p = prompt.lower().strip()

    granularity = "Daywise"
    descriptions = []

    if "week" in p:
        granularity = "Weekwise"
    if "month" in p:
        granularity = "Monthwise"
    if "day" in p:
        granularity = "Daywise"

    selected_month = None
    for month_name, month_num in MONTH_MAP.items():
        if re.search(rf"\b{month_name}\b", p):
            selected_month = month_num
            break

    if selected_month is not None:
        filtered = filtered[filtered["month_num"] == selected_month]
        descriptions.append(f"month = {selected_month}")

    selected_weekday = None
    selected_weekday_name = None
    for weekday_name, weekday_num in WEEKDAY_MAP.items():
        if re.search(rf"\b{weekday_name}\b", p):
            selected_weekday = weekday_num
            selected_weekday_name = weekday_name.title()
            break

    selected_ordinal = None
    for ordinal_text, ordinal_num in ORDINAL_MAP.items():
        if re.search(rf"\b{ordinal_text}\b", p):
            selected_ordinal = ordinal_num
            break

    if selected_weekday is not None:
        filtered = filtered[filtered["weekday_num"] == selected_weekday]
        descriptions.append(f"weekday = {selected_weekday_name}")

    if selected_ordinal is not None:
        filtered = filtered[filtered["nth_weekday_of_month"] == selected_ordinal]
        descriptions.append(f"occurrence = {selected_ordinal}")

    if not descriptions:
        descriptions.append("no filter")

    return filtered, granularity, ", ".join(descriptions)


def apply_manual_filter(df, filter_mode, weekday, ordinal, month):
    filtered = add_date_features(df)
    desc = "no filter"

    if filter_mode == "Specific weekday":
        weekday_num = WEEKDAY_MAP[weekday.lower()]
        filtered = filtered[filtered["weekday_num"] == weekday_num]
        desc = f"weekday = {weekday}"

    elif filter_mode == "Nth weekday of each month":
        weekday_num = WEEKDAY_MAP[weekday.lower()]
        filtered = filtered[
            (filtered["weekday_num"] == weekday_num)
            & (filtered["nth_weekday_of_month"] == ordinal)
        ]
        desc = f"{ordinal} occurrence of {weekday}"

    elif filter_mode == "Specific month":
        month_num = MONTH_MAP[month.lower()]
        filtered = filtered[filtered["month_num"] == month_num]
        desc = f"month = {month}"

    return filtered, desc


def aggregate_for_graph(df, metric_col, granularity):
    if df.empty:
        return df

    graph_df = df.copy()

    if granularity == "Daywise":
        graph_df["period"] = graph_df["timestamp"]
    elif granularity == "Weekwise":
        graph_df["period"] = graph_df["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    else:
        graph_df["period"] = graph_df["timestamp"].dt.to_period("M").dt.to_timestamp()

    if any(word in metric_col.lower() for word in ["emissions", "energy", "kwh"]):
        agg_func = "sum"
    else:
        agg_func = "mean"

    out = (
        graph_df
        .groupby("period", as_index=False)[metric_col]
        .agg(agg_func)
        .sort_values("period")
    )

    return out


# -------------------------------------------------
# Report Generator
# -------------------------------------------------
def generate_markdown_report(summary_df, result_df):
    latest_date = result_df["timestamp"].max().strftime("%Y-%m-%d")
    start_date = result_df["timestamp"].min().strftime("%Y-%m-%d")

    total_co2 = result_df["CO2_Emissions_tCO2e"].sum()
    total_ghg = result_df["Total_GHG_tCO2e"].sum()
    total_equipment = result_df["equipment_total_emissions_tCO2e"].sum()

    report = f"""
# INEOS Equipment Degradation & Carbon Report

## Reporting Period
- Start Date: {start_date}
- End Date: {latest_date}
- Records Analysed: {len(result_df)}

## Carbon Summary
- Total CO2 Emissions: {total_co2:,.2f} tCO2e
- Total GHG Emissions: {total_ghg:,.2f} tCO2e
- Equipment-attributed Emissions: {total_equipment:,.2f} tCO2e

## User-Entered Equipment Parameters
"""

    for _, row in summary_df.iterrows():
        report += f"""
### {row['Equipment']}
- Parameter: {row['Parameter']}
- User Input: {row['User Input']}
- Baseline Proxy: {row['Baseline Proxy']}
- User Input Degradation % / Year: {row['User Input Degradation % / Year']}
- Dataset Average Degradation % / Year: {row['Dataset Avg Degradation % / Year']}
- Status: {row['Status']}
- Total Emissions: {row['Total Emissions tCO2e']} tCO2e
- Total Energy: {row['Total Energy kWh']} kWh
- Formula: {row['Formula']}
- Key Factors: {row['Key Factors']}
- Common Failure Modes: {row['Common Failure Modes']}
- Monitoring Data: {row['Monitoring Data']}
"""

    report += """

## Methodology
The app uses equipment-specific degradation formulas from the internal equipment degradation reference file.

User-provided parameters:
- Furnace: U = effective heat flux
- Boiler: η / n = boiler thermal efficiency
- Reformer: A = catalyst activity

Baseline values are calculated from the first selected baseline window of the INEOS dataset.
"""

    return report


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("🏭 INEOS Calculator")
st.sidebar.caption("Equipment degradation + carbon impact")

st.sidebar.markdown("---")
st.sidebar.markdown("### Upload INEOS Dataset")

uploaded_ineos = st.sidebar.file_uploader(
    "Upload INEOS Excel dataset",
    type=["xlsx", "xls"]
)

baseline_days = st.sidebar.slider(
    "Baseline window days",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Equipment degradation formulas are loaded internally from the system file. "
    "Users only need to upload the INEOS Excel dataset and enter equipment parameters."
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------
ineos_df = read_ineos_file(uploaded_ineos)
reference_df = load_equipment_reference()

if ineos_df is None:
    st.warning(
        "Please upload the INEOS Excel file, or place it in "
        "`data/INEOS_Daily_CO2_Dataset_2010_2024_updated.xlsx`."
    )
    st.stop()

ineos_df = clean_columns(ineos_df)
reference_df = clean_columns(reference_df)

missing_cols = detect_required_columns(ineos_df)

if missing_cols:
    st.error("The INEOS dataset is missing required columns:")
    st.code("\n".join(missing_cols))
    st.stop()


# -------------------------------------------------
# Main UI Header
# -------------------------------------------------
st.title("🏭 INEOS Equipment Carbon Footprint Calculator")
st.write(
    "Calculate equipment-wise carbon emissions and degradation for Boiler, Furnace, and Reformer "
    "using INEOS daily CO₂ data and user-entered equipment parameters."
)


# -------------------------------------------------
# User Parameter Inputs - visible in main page
# -------------------------------------------------
st.markdown("## ⚙️ Enter Equipment Parameters")

st.markdown(
    """
    <div class="input-card">
    Enter the current measured values for each equipment parameter. These values are used directly in degradation calculation.
    </div>
    """,
    unsafe_allow_html=True
)

param_col1, param_col2, param_col3 = st.columns(3)

with param_col1:
    user_u = st.number_input(
        "Furnace U - Effective Heat Flux",
        min_value=0.000001,
        value=1.000000,
        step=0.001,
        format="%.6f",
        help="Enter current effective heat flux value for the furnace."
    )

with param_col2:
    user_eta = st.number_input(
        "Boiler η / n - Boiler Thermal Efficiency",
        min_value=0.000001,
        value=1.000000,
        step=0.001,
        format="%.6f",
        help="Enter current boiler thermal efficiency value."
    )

with param_col3:
    user_a = st.number_input(
        "Reformer A - Catalyst Activity",
        min_value=0.000001,
        value=1.000000,
        step=0.001,
        format="%.6f",
        help="Enter current catalyst activity value for the reformer."
    )


# -------------------------------------------------
# Calculate
# -------------------------------------------------
try:
    result_df, summary_df = calculate_equipment_degradation(
        raw_df=ineos_df,
        reference_df=reference_df,
        baseline_days=baseline_days,
        user_eta=user_eta,
        user_u=user_u,
        user_a=user_a,
    )
except Exception as e:
    st.error(f"Could not calculate equipment degradation: {str(e)}")
    st.stop()


# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
total_co2 = result_df["CO2_Emissions_tCO2e"].sum()
total_ghg = result_df["Total_GHG_tCO2e"].sum()
equipment_total = result_df["equipment_total_emissions_tCO2e"].sum()
records = len(result_df)

st.markdown("---")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Records Analysed", f"{records:,}")

with kpi2:
    st.metric("Total CO₂", f"{total_co2:,.0f} tCO₂e")

with kpi3:
    st.metric("Total GHG", f"{total_ghg:,.0f} tCO₂e")

with kpi4:
    st.metric("Equipment Emissions", f"{equipment_total:,.0f} tCO₂e")

st.markdown("---")


# -------------------------------------------------
# Tabs - Formula Mapping tab removed
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Summary",
    "📈 Trends",
    "📄 Report",
    "🧾 Data"
])


# -------------------------------------------------
# Summary Tab
# -------------------------------------------------
with tab1:
    st.subheader("Equipment Degradation Summary")

    display_cols = [
        "Equipment",
        "Parameter",
        "User Input",
        "Baseline Proxy",
        "Current User Proxy",
        "User Input Degradation % / Year",
        "Dataset Avg Degradation % / Year",
        "Latest Emissions tCO2e",
        "Total Emissions tCO2e",
        "Status",
    ]

    st.dataframe(summary_df[display_cols], use_container_width=True)

    cols = st.columns(3)

    for idx, row in summary_df.iterrows():
        color = risk_color(row["Status"])

        with cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color:{color};">
                    <h3>{row['Equipment']}</h3>
                    <p><b>Parameter:</b> {row['Parameter']}</p>
                    <p><b>User Input:</b> {row['User Input']}</p>
                    <p><b>Status:</b> {row['Status']}</p>
                    <p><b>User Input Degradation:</b> {row['User Input Degradation % / Year']}% / year</p>
                    <p><b>Total Emissions:</b> {row['Total Emissions tCO2e']} tCO₂e</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Equipment Emissions Split")

    split_df = pd.DataFrame({
        "Equipment": ["Boiler", "Furnace", "Reformer", "Other Equipment"],
        "Emissions": [
            result_df["boiler"].sum(),
            result_df["furance"].sum(),
            result_df["reformer"].sum(),
            result_df["other_Equpiments"].sum(),
        ],
    })

    fig_pie = px.pie(
        split_df,
        names="Equipment",
        values="Emissions",
        hole=0.45,
        title="Equipment-wise Total Emissions Share"
    )

    st.plotly_chart(fig_pie, use_container_width=True)


# -------------------------------------------------
# Trends Tab
# -------------------------------------------------
with tab2:
    st.subheader("Daywise / Weekwise / Monthwise Graphs")

    selected_equipment = st.selectbox(
        "Select equipment",
        ["Boiler", "Furnace", "Reformer"]
    )

    if selected_equipment == "Boiler":
        metric_options = {
            "Degradation % / Year": "boiler_degradation_pct_year",
            "Efficiency Proxy": "boiler_efficiency_proxy",
            "Emissions": "boiler",
            "Energy kWh": "boiler_energy KWh",
        }
    elif selected_equipment == "Furnace":
        metric_options = {
            "Degradation % / Year": "furnace_degradation_pct_year",
            "Effective Heat Flux Proxy": "furnace_heat_flux_proxy",
            "Emissions": "furance",
            "Energy kWh": "furnace_energy KWh",
        }
    else:
        metric_options = {
            "Degradation % / Year": "reformer_degradation_pct_year",
            "Catalyst Activity Proxy": "reformer_activity_proxy",
            "Emissions": "reformer",
            "Energy kWh": "reforemer_energy KWh",
        }

    selected_metric_name = st.selectbox(
        "Select graph metric",
        list(metric_options.keys())
    )

    metric_col = metric_options[selected_metric_name]

    st.markdown("### Graph Prompt")
    graph_prompt = st.text_input(
        "Type graph instruction",
        placeholder="Examples: daywise, weekwise, monthwise, 1st Monday, first Monday, January, weekwise Monday"
    )

    if graph_prompt.strip():
        filtered_df, granularity, filter_desc = apply_prompt_filter(result_df, graph_prompt)
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            granularity = st.selectbox(
                "Graph frequency",
                ["Daywise", "Weekwise", "Monthwise"]
            )

        with col_b:
            filter_mode = st.selectbox(
                "Date filter",
                [
                    "All",
                    "Specific weekday",
                    "Nth weekday of each month",
                    "Specific month",
                ]
            )

        weekday = "Monday"
        ordinal = 1
        month = "January"

        if filter_mode in ["Specific weekday", "Nth weekday of each month"]:
            weekday = st.selectbox(
                "Select weekday",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )

        if filter_mode == "Nth weekday of each month":
            ordinal = st.selectbox(
                "Select occurrence",
                [1, 2, 3, 4, 5],
                format_func=lambda x: f"{x}st" if x == 1 else f"{x}nd" if x == 2 else f"{x}rd" if x == 3 else f"{x}th"
            )

        if filter_mode == "Specific month":
            month = st.selectbox(
                "Select month",
                [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
            )

        filtered_df, filter_desc = apply_manual_filter(
            result_df,
            filter_mode,
            weekday,
            ordinal,
            month
        )

    st.info(
        f"Graph mode: **{granularity}** | Filter: **{filter_desc}** | Rows: **{len(filtered_df)}**"
    )

    graph_df = aggregate_for_graph(filtered_df, metric_col, granularity)

    if graph_df.empty:
        st.warning("No data found for the selected graph/filter.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=graph_df["period"],
            y=graph_df[metric_col],
            mode="lines+markers",
            name=selected_metric_name,
            line=dict(width=2)
        ))

        fig.update_layout(
            title=f"{selected_equipment} - {selected_metric_name} ({granularity})",
            xaxis_title="Date",
            yaxis_title=selected_metric_name,
            height=430,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Filtered Graph Data")
        st.dataframe(graph_df, use_container_width=True)


# -------------------------------------------------
# Report Tab
# -------------------------------------------------
with tab3:
    st.subheader("Download Report")

    report_md = generate_markdown_report(summary_df, result_df)

    st.markdown(report_md)

    st.download_button(
        label="📥 Download Degradation Report",
        data=report_md,
        file_name="ineos_equipment_degradation_report.md",
        mime="text/markdown"
    )

    csv_summary = summary_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Summary CSV",
        data=csv_summary,
        file_name="equipment_degradation_summary.csv",
        mime="text/csv"
    )


# -------------------------------------------------
# Data Tab
# -------------------------------------------------
with tab4:
    st.subheader("Processed Dataset")

    st.dataframe(result_df, use_container_width=True)

    processed_csv = result_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Full Processed Dataset",
        data=processed_csv,
        file_name="ineos_equipment_degradation_full_results.csv",
        mime="text/csv"
    )