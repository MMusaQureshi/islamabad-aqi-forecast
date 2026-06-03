from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api_client import fetch_realtime_plus_forecast_hourly
from src.config import FEATURE_COLUMNS
from src.feature_engineering import latest_feature_frame_for_prediction


def load_local_best_model(model_dir: str | Path = "model_artifacts/best_model") -> dict[str, Any]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    best_model = metadata["best_model"]

    if best_model == "tensorflow_mlp":
        import tensorflow as tf

        model = tf.keras.models.load_model(model_dir / "keras_model")
        scaler = joblib.load(model_dir / "scaler.joblib")
        return {"model": model, "scaler": scaler, "metadata": metadata}

    model = joblib.load(model_dir / "model.joblib")
    return {"model": model, "metadata": metadata}


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def recursive_daily_forecast(model_bundle: dict[str, Any], days: int = 3) -> pd.DataFrame:
    hourly = fetch_realtime_plus_forecast_hourly(past_days=14, forecast_days=days+1)
    daily = latest_feature_frame_for_prediction(hourly).sort_values("date").reset_index(drop=True)

    metadata = model_bundle["metadata"]
    selected_features = metadata["selected_features"]
    model_name = metadata["best_model"]
    model = model_bundle["model"]

    today = pd.Timestamp.today(tz="Asia/Karachi").normalize().tz_localize(None)
    future_dates = sorted(daily[daily["date"] > today]["date"].unique())[:days]

    predictions = []

    # Replace future AQI values with predictions as we move forward.
    working = daily.copy()

    for forecast_date in future_dates:
        idx = working.index[working["date"] == forecast_date]
        if len(idx) == 0:
            continue
        idx = idx[0]

        # Recalculate lag features from previous actual/predicted AQI.
        for lag in [1, 2, 3, 7]:
            previous_date = forecast_date - pd.Timedelta(days=lag)
            prev_values = working.loc[working["date"] == previous_date, "aqi_mean"]
            working.loc[idx, f"aqi_lag_{lag}d"] = float(prev_values.iloc[0]) if len(prev_values) else None

        prev_3 = working[(working["date"] < forecast_date) & (working["date"] >= forecast_date - pd.Timedelta(days=3))]["aqi_mean"]
        prev_7 = working[(working["date"] < forecast_date) & (working["date"] >= forecast_date - pd.Timedelta(days=7))]["aqi_mean"]
        working.loc[idx, "aqi_rolling_mean_3d"] = prev_3.mean()
        working.loc[idx, "aqi_rolling_mean_7d"] = prev_7.mean()
        working.loc[idx, "aqi_change_1d"] = working.loc[idx, "aqi_lag_1d"] - working.loc[idx, "aqi_lag_2d"]

        X = working.loc[[idx], selected_features]

        if model_name == "tensorflow_mlp":
            scaler = model_bundle["scaler"]
            pred = float(model.predict(scaler.transform(X), verbose=0).ravel()[0])
        else:
            pred = float(model.predict(X)[0])

        pred = max(0.0, min(500.0, pred))

        # This makes the next loop recursive.
        working.loc[idx, "aqi_mean"] = pred

        predictions.append(
            {
                "date": pd.to_datetime(forecast_date).date().isoformat(),
                "predicted_daily_aqi": round(pred, 1),
                "category": aqi_category(pred),
            }
        )

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    bundle = load_local_best_model()
    print(recursive_daily_forecast(bundle, days=3))
