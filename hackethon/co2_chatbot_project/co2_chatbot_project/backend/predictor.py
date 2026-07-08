"""
predictor.py
-------------
Loads the trained model and metadata, and exposes a simple
`forecast(start_date, days)` function used by the chatbot and API.
"""

import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "co2_forecast_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "co2_metadata.pkl")

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _meta = joblib.load(META_PATH)
    return _model, _meta


def _make_features(dates: pd.DatetimeIndex, min_year: int) -> pd.DataFrame:
    year_trend = dates.year - min_year
    doy = dates.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame({
        "year_trend": year_trend,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    })


def forecast(start_date: str, days: int = 30):
    """
    Forecast CO2 emissions (tCO2e) for `days` days starting at `start_date`.
    start_date: 'YYYY-MM-DD' string
    Returns list of {date, predicted_co2}
    """
    model, meta = _load()
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    X = _make_features(dates, meta["min_year"])
    preds = model.predict(X)
    return [
        {"date": d.strftime("%Y-%m-%d"), "predicted_co2_tCO2e": round(float(p), 2)}
        for d, p in zip(dates, preds)
    ]


def forecast_single_date(date_str: str):
    result = forecast(date_str, days=1)
    return result[0]


def model_info():
    _, meta = _load()
    return meta
