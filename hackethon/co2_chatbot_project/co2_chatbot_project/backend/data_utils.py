"""
data_utils.py
--------------
Helper functions that answer historical/statistical questions
about the CO2 dataset (used by the chatbot).
"""

import pandas as pd
import numpy as np

DATA_PATH = "../data/INEOS_Daily_CO2_Dataset_2010_2024.xlsx"

_df = None


def load_data():
    """Load (and cache) the dataset once per process."""
    global _df
    if _df is None:
        df = pd.read_excel(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        _df = df
    return _df


def overall_stats():
    df = load_data()
    col = "CO2_Emissions_tCO2e"
    return {
        "records": int(len(df)),
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "mean": round(df[col].mean(), 2),
        "min": round(df[col].min(), 2),
        "max": round(df[col].max(), 2),
        "total": round(df[col].sum(), 2),
        "std": round(df[col].std(), 2),
    }


def year_stats(year: int):
    df = load_data()
    sub = df[df["Year"] == year]
    if sub.empty:
        return None
    col = "CO2_Emissions_tCO2e"
    return {
        "year": year,
        "mean": round(sub[col].mean(), 2),
        "min": round(sub[col].min(), 2),
        "max": round(sub[col].max(), 2),
        "total": round(sub[col].sum(), 2),
        "days": int(len(sub)),
    }


def compare_years(year_a: int, year_b: int):
    a = year_stats(year_a)
    b = year_stats(year_b)
    if a is None or b is None:
        return None
    diff = round(a["mean"] - b["mean"], 2)
    pct = round((diff / b["mean"]) * 100, 2) if b["mean"] else None
    return {"year_a": a, "year_b": b, "mean_diff": diff, "pct_change": pct}


def yearly_trend():
    """Average CO2 per year, used to describe long-term trend."""
    df = load_data()
    trend = df.groupby("Year")["CO2_Emissions_tCO2e"].mean().round(2)
    return trend.to_dict()


def monthly_seasonality():
    """Average CO2 per calendar month, across all years."""
    df = load_data()
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    seasonal = df.groupby("Month")["CO2_Emissions_tCO2e"].mean().round(2)
    return seasonal.to_dict()


def top_correlations():
    """Which factors correlate most strongly with CO2 emissions."""
    df = load_data()
    numeric_cols = [
        "Production_Index", "Natural_Gas_GJ", "Energy_MWh",
        "Steam_tonnes", "Flare_Emissions_tCO2e", "CH4_tCO2e", "N2O_tCO2e",
    ]
    corr = df[numeric_cols + ["CO2_Emissions_tCO2e"]].corr()["CO2_Emissions_tCO2e"]
    corr = corr.drop("CO2_Emissions_tCO2e").sort_values(ascending=False)
    return corr.round(3).to_dict()


def highest_lowest_days(n=3):
    df = load_data()
    col = "CO2_Emissions_tCO2e"
    top = df.nlargest(n, col)[["Date", col]]
    bottom = df.nsmallest(n, col)[["Date", col]]
    top["Date"] = top["Date"].dt.strftime("%Y-%m-%d")
    bottom["Date"] = bottom["Date"].dt.strftime("%Y-%m-%d")
    return {
        "highest": top.to_dict("records"),
        "lowest": bottom.to_dict("records"),
    }
