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


def register_sklearn_model(model_dir: str | Path, metrics: dict, input_example) -> None:
    mr = get_model_registry()
    model_meta = mr.sklearn.create_model(
        name=CFG.model_name,
        metrics=metrics,
        description="Best selected daily AQI forecasting model for Islamabad.",
        input_example=input_example,
    )
    model_meta.save(str(model_dir))
