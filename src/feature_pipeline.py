from __future__ import annotations

import logging

import pandas as pd

from src.api_client import fetch_realtime_plus_forecast_hourly
from src.config import CFG
from src.feature_engineering import make_training_frame
from src.hopsworks_utils import insert_daily_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def run_hourly_feature_pipeline(past_days: int = 30) -> None:
    """
    Hourly feature pipeline.

    This pipeline refreshes recent completed daily feature rows in Hopsworks.

    Important:
    - The model is daily, not hourly.
    - We fetch hourly data, aggregate it into daily features, then store daily rows.
    - Today's row is excluded because today's AQI mean is incomplete until the day ends.
    """

    logging.info("Starting hourly feature pipeline.")
    logging.info("City: %s", CFG.city)
    logging.info("Fetching recent hourly data for past %s days.", past_days)

    hourly = fetch_realtime_plus_forecast_hourly(
        past_days=past_days,
        forecast_days=1,
    )

    logging.info("Fetched hourly rows: %s", len(hourly))

    daily_features = make_training_frame(hourly)

    if daily_features.empty:
        logging.warning("No daily features generated. Nothing to insert.")
        return

    daily_features["date"] = pd.to_datetime(daily_features["date"])

    today = pd.Timestamp.now(tz=CFG.timezone).normalize().tz_localize(None)

    # Keep only completed days.
    completed_daily_features = daily_features[daily_features["date"] < today].copy()

    if completed_daily_features.empty:
        logging.warning("No completed daily feature rows available. Nothing to insert.")
        return

    # Match Hopsworks schema types.
    if "day_of_week" in completed_daily_features.columns:
        completed_daily_features["day_of_week"] = completed_daily_features["day_of_week"].astype("int32")

    if "month" in completed_daily_features.columns:
        completed_daily_features["month"] = completed_daily_features["month"].astype("int32")

    if "is_weekend" in completed_daily_features.columns:
        completed_daily_features["is_weekend"] = completed_daily_features["is_weekend"].astype("int64")

    logging.info(
        "Prepared completed daily rows for insert: %s",
        len(completed_daily_features),
    )

    logging.info(
        "Date range: %s to %s",
        completed_daily_features["date"].min(),
        completed_daily_features["date"].max(),
    )

    insert_daily_features(completed_daily_features, wait=False)
    
    logging.info(
        "Inserted/updated %s completed daily rows in Hopsworks Feature Store.",
        len(completed_daily_features),
    )


if __name__ == "__main__":
    run_hourly_feature_pipeline()