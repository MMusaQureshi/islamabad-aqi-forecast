from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.api_client import fetch_realtime_plus_forecast_hourly
from src.config import CFG
from src.feature_engineering import latest_feature_frame_for_prediction
from src.hopsworks_utils import read_daily_features


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


def load_local_best_model(model_dir: str | Path = "model_artifacts/best_model") -> dict[str, Any]:
    model_dir = Path(model_dir)

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    # New format: horizon-specific models.
    if metadata.get("forecast_method") == "horizon_specific_direct_multi_horizon":
        horizon_models = {}

        for horizon_info in metadata["horizon_models"]:
            horizon = int(horizon_info["horizon"])
            framework = horizon_info["framework"]
            selected_model = horizon_info["selected_model"]

            payload = {
                "framework": framework,
                "selected_model": selected_model,
                "selected_features": horizon_info["selected_features"],
            }

            if framework == "sklearn":
                payload["model"] = joblib.load(model_dir / horizon_info["model_path"])

            elif framework == "tensorflow":
                import tensorflow as tf

                payload["model"] = tf.keras.models.load_model(
                    model_dir / horizon_info["model_path"]
                )
                payload["scaler"] = joblib.load(model_dir / horizon_info["scaler_path"])

            elif framework == "baseline":
                payload["model"] = None

            horizon_models[horizon] = payload

        return {
            "metadata": metadata,
            "horizon_models": horizon_models,
            "model": None,
        }

    # Old format: single model.
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


def _latest_completed_row(selected_features: list[str]):
    """
    First tries live API.
    If live API fails, falls back to Hopsworks Feature Store.
    """

    # 1. Try live API first.
    try:
        hourly = fetch_realtime_plus_forecast_hourly(
            past_days=30,
            forecast_days=1,
        )

        daily = latest_feature_frame_for_prediction(hourly)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        today = pd.Timestamp.now(tz=CFG.timezone).normalize().tz_localize(None)
        completed = daily[daily["date"] < today].copy()

        if completed.empty:
            raise RuntimeError("No completed daily row available from live API.")

        latest_row = completed.iloc[-1]
        latest_date = pd.to_datetime(latest_row["date"])

        missing = [col for col in selected_features if col not in latest_row.index]
        if missing:
            raise RuntimeError(f"Live API row missing selected features: {missing}")

        return latest_row, latest_date, "live API (Open-Meteo)"

    except Exception as live_exc:
        print(f"Live API failed: {live_exc}")
        print("Trying Hopsworks Feature Store fallback...")

    # 2. Hopsworks fallback.
    try:
        cloud_df = read_daily_features()
        cloud_df["date"] = pd.to_datetime(cloud_df["date"])
        cloud_df = cloud_df.sort_values("date").reset_index(drop=True)

        if cloud_df.empty:
            raise RuntimeError("Hopsworks returned no rows.")

        latest_row = cloud_df.iloc[-1]
        latest_date = pd.to_datetime(latest_row["date"])

        missing = [col for col in selected_features if col not in latest_row.index]
        if missing:
            raise RuntimeError(f"Hopsworks row missing selected features: {missing}")

        return latest_row, latest_date, "Hopsworks Feature Store"

    except Exception as cloud_exc:
        raise RuntimeError(
            "Both live API and Hopsworks Feature Store failed. "
            f"Hopsworks error: {cloud_exc}"
        )


def _predict_single_horizon(horizon_payload: dict, latest_row: pd.Series) -> float:
    framework = horizon_payload["framework"]
    selected_model = horizon_payload["selected_model"]
    selected_features = horizon_payload["selected_features"]

    if framework == "baseline":
        if selected_model == "baseline_persistence":
            return float(latest_row["aqi_mean"])

        if selected_model == "baseline_rolling_3d":
            return float(latest_row["aqi_rolling_mean_3d"])

        if selected_model == "baseline_rolling_7d":
            return float(latest_row["aqi_rolling_mean_7d"])

        return float(latest_row["aqi_mean"])

    X = pd.DataFrame([latest_row[selected_features].to_dict()])

    if framework == "sklearn":
        return float(horizon_payload["model"].predict(X).ravel()[0])

    if framework == "tensorflow":
        X_scaled = horizon_payload["scaler"].transform(X)
        return float(horizon_payload["model"].predict(X_scaled, verbose=0).ravel()[0])

    raise RuntimeError(f"Unsupported horizon framework: {framework}")


def direct_three_day_forecast(model_bundle: dict[str, Any], days: int = 3) -> pd.DataFrame:
    """
    Predict next 3 days AQI.

    Supports:
    - New horizon-specific direct models
    - Old single direct multi-output model
    """

    metadata = model_bundle["metadata"]

    # New horizon-specific format.
    if metadata.get("forecast_method") == "horizon_specific_direct_multi_horizon":
        horizon_models = model_bundle["horizon_models"]

        all_features = sorted(
            {
                feature
                for payload in horizon_models.values()
                for feature in payload["selected_features"]
            }
        )

        latest_row, latest_date, source = _latest_completed_row(all_features)
        print(f"Forecast anchored on {latest_date:%Y-%m-%d} using features from {source}.")

        today = pd.Timestamp.now(tz=CFG.timezone).normalize().tz_localize(None)

        predictions = []

        for horizon in range(1, days + 1):
            horizon_payload = horizon_models[horizon]
            pred = _predict_single_horizon(horizon_payload, latest_row)

            value = max(0.0, min(500.0, float(pred)))
            forecast_date = today + pd.Timedelta(days=horizon)

            predictions.append(
                {
                    "date": forecast_date.date().isoformat(),
                    "predicted_daily_aqi": round(value, 1),
                    "category": aqi_category(value),
                }
            )

        return pd.DataFrame(predictions)

    # Old single-model format.
    selected_features = metadata["selected_features"]
    model_name = metadata["best_model"]
    model = model_bundle["model"]

    latest_row, latest_date, source = _latest_completed_row(selected_features)
    print(f"Forecast anchored on {latest_date:%Y-%m-%d} using features from {source}.")

    X = pd.DataFrame([latest_row[selected_features].to_dict()])

    if model_name == "tensorflow_mlp":
        scaler = model_bundle["scaler"]
        preds = model.predict(scaler.transform(X), verbose=0).ravel()
    else:
        preds = model.predict(X).ravel()

    preds = preds[:days]

    today = pd.Timestamp.now(tz=CFG.timezone).normalize().tz_localize(None)

    predictions = []

    for i, pred in enumerate(preds, start=1):
        value = max(0.0, min(500.0, float(pred)))
        forecast_date = today + pd.Timedelta(days=i)

        predictions.append(
            {
                "date": forecast_date.date().isoformat(),
                "predicted_daily_aqi": round(value, 1),
                "category": aqi_category(value),
            }
        )

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    bundle = load_local_best_model()
    print(direct_three_day_forecast(bundle, days=3))