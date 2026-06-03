from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from src.config import CFG, AIR_QUALITY_HOURLY_VARS, WEATHER_HOURLY_VARS


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data.get("reason", "Unknown Open-Meteo API error"))
    return data


def _hourly_json_to_df(data: dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("API response does not contain hourly data.")
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


def fetch_air_quality_hourly(
    start_date: str | None = None,
    end_date: str | None = None,
    past_days: int | None = None,
    forecast_days: int | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "timezone": CFG.timezone,
        "domains": "cams_global",
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    if past_days is not None:
        params["past_days"] = past_days
    if forecast_days is not None:
        params["forecast_days"] = forecast_days

    return _hourly_json_to_df(_get_json(CFG.open_meteo_air_quality_url, params))


def fetch_weather_hourly_archive(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": CFG.timezone,
    }
    return _hourly_json_to_df(_get_json(CFG.open_meteo_weather_archive_url, params))


def fetch_weather_hourly_forecast(past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    params = {
        "latitude": CFG.latitude,
        "longitude": CFG.longitude,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": CFG.timezone,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    return _hourly_json_to_df(_get_json(CFG.open_meteo_weather_forecast_url, params))


def fetch_historical_hourly(days: int = 90) -> pd.DataFrame:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)

    start_str = start.isoformat()
    end_str = end.isoformat()

    aq = fetch_air_quality_hourly(start_date=start_str, end_date=end_str)
    weather = fetch_weather_hourly_archive(start_date=start_str, end_date=end_str)

    df = pd.merge(aq, weather, on="timestamp", how="inner")
    df["city"] = CFG.city
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_realtime_plus_forecast_hourly(past_days: int = 10, forecast_days: int = 3) -> pd.DataFrame:
    aq = fetch_air_quality_hourly(past_days=past_days, forecast_days=forecast_days)
    weather = fetch_weather_hourly_forecast(past_days=past_days, forecast_days=forecast_days)

    df = pd.merge(aq, weather, on="timestamp", how="inner")
    df["city"] = CFG.city
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_realtime_plus_forecast_hourly(past_days=2, forecast_days=3)
    print(df.head())
    print(df.tail())
    print(df.shape)
