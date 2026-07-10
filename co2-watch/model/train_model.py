"""
train_model.py
---------------
Trains a forecasting model for daily CO2 emissions (tCO2e) using the
INEOS Daily CO2 dataset. The model uses only DATE-DERIVED features
(year, cyclical day-of-year) so it can be used to forecast any future
date without needing future values of production/energy variables.

Run:
    python train_model.py

Produces:
    co2_forecast_model.pkl   -> trained sklearn model
    co2_metadata.pkl         -> min/max year, feature list, historical stats
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "../data/INEOS_Daily_CO2_Dataset_2010_2024.xlsx"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical + trend features from the Date column."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["DayOfYear"] = df["Date"].dt.dayofyear
    # cyclical encoding so Dec 31 and Jan 1 are "close" to the model
    df["doy_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    # linear year trend, normalised so the model generalises to future years
    df["year_trend"] = df["Year"] - df["Year"].min()
    return df


def main():
    print("Loading data...")
    df = pd.read_excel(DATA_PATH)
    df = build_features(df)

    feature_cols = ["year_trend", "doy_sin", "doy_cos"]
    target_col = "CO2_Emissions_tCO2e"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )

    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE : {mae:.2f} tCO2e")
    print(f"Test R^2 : {r2:.3f}")

    # Save model + metadata needed by the chatbot backend
    joblib.dump(model, "co2_forecast_model.pkl")

    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "min_year": int(df["Year"].min()),
        "max_year": int(df["Year"].max()),
        "historical_mean": float(df[target_col].mean()),
        "historical_std": float(df[target_col].std()),
        "historical_min": float(df[target_col].min()),
        "historical_max": float(df[target_col].max()),
        "test_mae": float(mae),
        "test_r2": float(r2),
    }
    joblib.dump(metadata, "co2_metadata.pkl")
    print("Saved co2_forecast_model.pkl and co2_metadata.pkl")


if __name__ == "__main__":
    main()
