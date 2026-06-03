from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    city: str = os.getenv("CITY", "Islamabad")
    latitude: float = float(os.getenv("LATITUDE", "33.6844"))
    longitude: float = float(os.getenv("LONGITUDE", "73.0479"))
    timezone: str = os.getenv("TIMEZONE", "Asia/Karachi")

    hopsworks_project_name: str | None = os.getenv("HOPSWORKS_PROJECT_NAME")
    hopsworks_api_key: str | None = os.getenv("HOPSWORKS_API_KEY")

    feature_group_name: str = "aqi_islamabad_daily_features"
    feature_group_version: int = 3

    feature_view_name: str = "aqi_islamabad_daily_fv"
    feature_view_version: int = 1

    model_name: str = "islamabad_daily_aqi_forecaster"

    open_meteo_air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    open_meteo_weather_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_weather_forecast_url: str = "https://api.open-meteo.com/v1/forecast"


CFG = Config()

AIR_QUALITY_HOURLY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]

TARGET_COLUMN = "aqi_mean"
FEATURE_COLUMNS = [
    # pollutant forecast/observed daily aggregates
    "pm2_5_mean",
    "pm2_5_max",
    "pm10_mean",
    "pm10_max",
    "carbon_monoxide_mean",
    "nitrogen_dioxide_mean",
    "sulphur_dioxide_mean",
    "ozone_mean",

    # weather daily aggregates
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "precipitation_sum",
    "cloud_cover_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_mean",

    # lag/rolling AQI features
    "aqi_lag_1d",
    "aqi_lag_2d",
    "aqi_lag_3d",
    "aqi_lag_7d",
    "aqi_rolling_mean_3d",
    "aqi_rolling_mean_7d",
    "aqi_change_1d",

    # derived interaction features
    "pm2_5_to_pm10_ratio",
    "wind_pm2_5_interaction",
    "humidity_pm2_5_interaction",
    "temperature_ozone_interaction",

    # time features
    "day_of_week",
    "month",
    "is_weekend",
    
    "aqi_change_2d",
    "aqi_accel",
    "aqi_rolling_mean_14d",
    "aqi_deviation_from_7d",
    "season",
]