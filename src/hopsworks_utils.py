from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from src.config import CFG


def login_project():
    import hopsworks

    api_key = CFG.hopsworks_api_key
    project_name = CFG.hopsworks_project_name

    if not api_key:
        raise RuntimeError("Missing HOPSWORKS_API_KEY. Add it to .env or GitHub Secrets.")

    # project can be omitted if the API key is scoped to one project,
    # but keeping it explicit is safer for team projects.
    if project_name:
        return hopsworks.login(project=project_name, api_key_value=api_key)

    return hopsworks.login(api_key_value=api_key)


def get_feature_store():
    project = login_project()
    return project.get_feature_store()


def get_model_registry():
    project = login_project()
    return project.get_model_registry()


def get_or_create_daily_feature_group():
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=CFG.feature_group_name,
        version=CFG.feature_group_version,
        description="Daily AQI forecasting features for Islamabad generated from hourly weather and pollutant data.",
        primary_key=["city", "date"],
        event_time="date",
        online_enabled=False,
    )
    return fg


def insert_daily_features(df, wait: bool = True):
    fg = get_or_create_daily_feature_group()
    fg.insert(df, operation="upsert", wait=wait)


def read_daily_features() -> pd.DataFrame:
    fg = get_or_create_daily_feature_group()
    return fg.read()


def read_recent_daily_features() -> pd.DataFrame:
    """Read daily features from the cloud, sorted oldest -> newest by date.

    Used by the prediction path so the dashboard can serve forecasts straight
    from the Feature Store (the cloud) even while the live weather API is down.
    """
    df = read_daily_features()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["city", "date"]) if "city" in df.columns else df.sort_values("date")
    return df.reset_index(drop=True)


def try_read_recent_daily_features() -> pd.DataFrame | None:
    """Best-effort cloud read. Returns None on any failure (no creds, network,
    empty store) so callers can fall back to the live API gracefully."""
    try:
        df = read_recent_daily_features()
        return df if len(df) else None
    except Exception as exc:
        print(f"Could not read features from the cloud ({exc}). Falling back to live API.")
        return None


def register_sklearn_model(model_dir: str | Path, metrics: dict, input_example) -> None:
    mr = get_model_registry()
    model_meta = mr.sklearn.create_model(
        name=CFG.model_name,
        metrics=metrics,
        description="Best selected daily AQI forecasting model for Islamabad.",
        input_example=input_example,
    )
    model_meta.save(str(model_dir))
