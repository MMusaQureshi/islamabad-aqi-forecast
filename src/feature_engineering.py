from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def hourly_to_daily_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    agg = {
        "pm2_5": ["mean", "max"],
        "pm10": ["mean", "max"],
        "carbon_monoxide": ["mean"],
        "nitrogen_dioxide": ["mean"],
        "sulphur_dioxide": ["mean"],
        "ozone": ["mean"],
        "us_aqi": ["mean", "max"],
        "temperature_2m": ["mean", "max", "min"],
        "relative_humidity_2m": ["mean"],
        "surface_pressure": ["mean"],
        "precipitation": ["sum"],
        "cloud_cover": ["mean"],
        "wind_speed_10m": ["mean"],
        "wind_direction_10m": ["mean"],
    }

    daily = df.groupby(["city", "date"], as_index=False).agg(agg)

    daily.columns = [
        "_".join([str(c) for c in col if c]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in daily.columns
    ]

    daily = daily.rename(
        columns={
            "us_aqi_mean": "aqi_mean",
            "us_aqi_max": "aqi_max",
        }
    )

    daily["date"] = pd.to_datetime(daily["date"])

    return daily.sort_values(["city", "date"]).reset_index(drop=True)


def add_daily_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    grouped = df.groupby("city", group_keys=False)

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int32")
    df["month"] = df["date"].dt.month.astype("int32")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")

    # Season feature
    # 0 = winter, 1 = spring, 2 = summer, 3 = fall
    df["season"] = ((df["month"] % 12) // 3).astype("int32")

    # -----------------------------------------------------
    # AQI lag features
    # -----------------------------------------------------
    df["aqi_lag_1d"] = grouped["aqi_mean"].shift(1)
    df["aqi_lag_2d"] = grouped["aqi_mean"].shift(2)
    df["aqi_lag_3d"] = grouped["aqi_mean"].shift(3)
    df["aqi_lag_7d"] = grouped["aqi_mean"].shift(7)

    # -----------------------------------------------------
    # AQI rolling features
    # Important: rolling features use shifted AQI only.
    # This prevents leakage from the current day.
    # -----------------------------------------------------
    df["aqi_rolling_mean_3d"] = grouped["aqi_mean"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=3).mean()
    )

    df["aqi_rolling_mean_7d"] = grouped["aqi_mean"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=7).mean()
    )

    df["aqi_rolling_mean_14d"] = grouped["aqi_mean"].transform(
        lambda s: s.shift(1).rolling(window=14, min_periods=14).mean()
    )

    # -----------------------------------------------------
    # AQI change / momentum features
    # These must come AFTER lag features.
    # -----------------------------------------------------
    df["aqi_change_1d"] = df["aqi_lag_1d"] - df["aqi_lag_2d"]
    df["aqi_change_2d"] = df["aqi_lag_2d"] - df["aqi_lag_3d"]

    # Acceleration must come AFTER aqi_change_1d and aqi_change_2d.
    df["aqi_accel"] = df["aqi_change_1d"] - df["aqi_change_2d"]

    # Deviation must come AFTER aqi_rolling_mean_7d.
    df["aqi_deviation_from_7d"] = df["aqi_lag_1d"] - df["aqi_rolling_mean_7d"]

    # -----------------------------------------------------
    # Derived pollutant/weather features
    # -----------------------------------------------------
    df["pm2_5_to_pm10_ratio"] = df["pm2_5_mean"] / df["pm10_mean"].replace(0, np.nan)
    df["wind_pm2_5_interaction"] = df["wind_speed_10m_mean"] * df["pm2_5_mean"]
    df["humidity_pm2_5_interaction"] = df["relative_humidity_2m_mean"] * df["pm2_5_mean"]
    df["temperature_ozone_interaction"] = df["temperature_2m_mean"] * df["ozone_mean"]

    df = df.replace([np.inf, -np.inf], np.nan)

    # -----------------------------------------------------
    # Hopsworks-compatible integer types
    # -----------------------------------------------------
    df["day_of_week"] = df["day_of_week"].astype("int32")
    df["month"] = df["month"].astype("int32")
    df["is_weekend"] = df["is_weekend"].astype("int64")
    df["season"] = df["season"].astype("int32")

    return df


def make_training_frame(hourly_df: pd.DataFrame) -> pd.DataFrame:
    daily = hourly_to_daily_features(hourly_df)
    features = add_daily_features(daily)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    return features.dropna(subset=required).reset_index(drop=True)


def latest_feature_frame_for_prediction(hourly_df: pd.DataFrame) -> pd.DataFrame:
    daily = hourly_to_daily_features(hourly_df)

    return add_daily_features(daily)


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y