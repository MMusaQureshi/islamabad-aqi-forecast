from __future__ import annotations


import argparse

from src.api_client import fetch_historical_hourly
from src.feature_engineering import make_training_frame
from src.hopsworks_utils import insert_daily_features


def run_backfill(days: int = 90, write: bool = False):
    hourly = fetch_historical_hourly(days=days)
    daily_features = make_training_frame(hourly)

    print(f"Fetched hourly rows: {len(hourly)}")
    print(f"Prepared daily feature rows: {len(daily_features)}")
    print(daily_features.tail())

    if write:
        insert_daily_features(daily_features,wait=True)
        print("Inserted daily features into Hopsworks Feature Store.")

    return daily_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    run_backfill(days=args.days, write=args.write)
