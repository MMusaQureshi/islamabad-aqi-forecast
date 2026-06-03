from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api_client import fetch_realtime_plus_forecast_hourly
from src.feature_engineering import latest_feature_frame_for_prediction


def load_local_best_model(model_dir: str | Path = "model_artifacts/best_model") -> dict[str, Any]:
    model_dir = Path(model_dir)

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    best_model = metadata["best_model"]

    if best_model == "tensorflow_mlp":
        import tensorflow as tf

        model = tf.keras.models.load_model(model_dir / "keras_model.keras")
        scaler = joblib.load(model_dir / "scaler.joblib")

        return {
            "model": model,
            "scaler": scaler,
            "metadata": metadata,
        }

    model = joblib.load(model_dir / "model.joblib")

    return {
        "model": model,
        "metadata": metadata,
    }


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


def direct_three_day_forecast(model_bundle: dict[str, Any], days: int = 3) -> pd.DataFrame:
    """
    Direct multi-horizon AQI forecasting.

    This does NOT use recursive forecasting.
    The model directly predicts:
    - Day 1 AQI
    - Day 2 AQI
    - Day 3 AQI

    This avoids prediction error accumulation.
    """

    hourly = fetch_realtime_plus_forecast_hourly(
        past_days=30,
        forecast_days=0,
    )

    daily = latest_feature_frame_for_prediction(hourly).sort_values("date").reset_index(drop=True)

    if daily.empty:
        raise RuntimeError("No daily features available for prediction.")

    metadata = model_bundle["metadata"]
    selected_features = metadata["selected_features"]
    model_name = metadata["best_model"]
    model = model_bundle["model"]

    # Use the latest available row.
    # This may include today's partial data, which makes the forecast more real-time.
    latest_row = daily.iloc[-2].copy()
    latest_date = pd.to_datetime(latest_row["date"])

    X = pd.DataFrame([latest_row[selected_features].to_dict()])

    if model_name == "tensorflow_mlp":
        scaler = model_bundle["scaler"]
        preds = model.predict(scaler.transform(X), verbose=0).ravel()
    else:
        preds = model.predict(X).ravel()

    preds = preds[:days]

    predictions = []

    today = pd.Timestamp.today().normalize()

    for i, pred in enumerate(preds, start=1):
        pred = float(max(0.0, min(500.0, pred)))

        # Forecast dates are from today, not from the latest row date.
        forecast_date = today + pd.Timedelta(days=i)

        predictions.append(
            {
                "date": forecast_date.date().isoformat(),
                "predicted_daily_aqi": round(pred, 1),
                "category": aqi_category(pred),
            }
        )

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    bundle = load_local_best_model()
    print(direct_three_day_forecast(bundle, days=3))