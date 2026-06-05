from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.predict import (
    load_local_best_model,
    direct_three_day_forecast,
    _latest_completed_row,
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon="🌫️",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS FOR BIGGER TEXT / READABLE TABLES
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 96% !important;
    }

    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    h1 {
        font-size: 46px !important;
        font-weight: 850 !important;
        margin-bottom: 0.5rem !important;
    }

    h2, h3 {
        font-size: 32px !important;
        font-weight: 800 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    .stMarkdown {
        font-size: 19px !important;
        line-height: 1.6 !important;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 17px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 19px !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 40px !important;
        font-weight: 850 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 17px !important;
        font-weight: 750 !important;
    }

    [data-testid="stAlert"] {
        font-size: 19px !important;
        font-weight: 650 !important;
        padding: 1rem !important;
    }

    table {
        font-size: 19px !important;
        width: 100% !important;
    }

    th {
        font-size: 19px !important;
        font-weight: 850 !important;
        padding: 12px !important;
        text-align: left !important;
    }

    td {
        font-size: 19px !important;
        padding: 12px !important;
        text-align: left !important;
    }

    [data-testid="stDataFrame"] {
        font-size: 19px !important;
    }

    pre, code {
        font-size: 17px !important;
    }

    hr {
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PATHS / CACHE HELPERS
# ---------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "model_artifacts" / "best_model"


def get_model_signature() -> str:
    """
    Used to refresh cached model if model artifacts change after retraining.
    """
    paths = [
        MODEL_DIR / "metadata.json",
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "keras_model.keras",
        MODEL_DIR / "scaler.joblib",
    ]

    # Horizon-specific artifacts
    for i in [1, 2, 3]:
        paths.extend(
            [
                MODEL_DIR / f"horizon_{i}_model.joblib",
                MODEL_DIR / f"horizon_{i}_keras_model.keras",
                MODEL_DIR / f"horizon_{i}_scaler.joblib",
            ]
        )

    parts = []

    for path in paths:
        if path.exists():
            parts.append(f"{path.name}:{path.stat().st_mtime}")

    return "|".join(parts)


@st.cache_resource(show_spinner="Loading trained model...")
def cached_model_bundle(model_signature: str) -> dict:
    """
    Cache model loading so Streamlit does not reload model artifacts every refresh.
    """
    return load_local_best_model(MODEL_DIR)


@st.cache_data(ttl=900, show_spinner="Generating next 3 days AQI forecast...")
def cached_direct_forecast(model_signature: str) -> pd.DataFrame:
    """
    Cache forecast for 15 minutes to avoid repeated live API calls.
    """
    bundle = load_local_best_model(MODEL_DIR)
    return direct_three_day_forecast(bundle, days=3)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_short_category(category: str) -> str:
    mapping = {
        "Good": "Good",
        "Moderate": "Moderate",
        "Unhealthy for Sensitive Groups": "Sensitive Groups",
        "Unhealthy": "Unhealthy",
        "Very Unhealthy": "Very Unhealthy",
        "Hazardous": "Hazardous",
    }
    return mapping.get(category, category)


def get_alert_message(aqi: float) -> tuple[str, str]:
    if aqi <= 50:
        return "Good air quality. Normal outdoor activity is safe.", "success"
    if aqi <= 100:
        return "Moderate AQI. Acceptable for most people.", "info"
    if aqi <= 150:
        return "Sensitive groups should limit outdoor activity.", "warning"
    if aqi <= 200:
        return "AQI alert: Unhealthy. Reduce outdoor exposure.", "error"
    if aqi <= 300:
        return "AQI alert: Very Unhealthy. Avoid outdoor activity.", "error"
    return "AQI alert: Hazardous. Stay indoors if possible.", "error"


def show_alert(message: str, level: str) -> None:
    if level == "success":
        st.success(message)
    elif level == "info":
        st.info(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


def readable_table(df: pd.DataFrame) -> None:
    styled = (
        df.style
        .set_properties(
            **{
                "font-size": "19px",
                "padding": "12px",
                "text-align": "left",
                "white-space": "normal",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "19px"),
                        ("font-weight", "bold"),
                        ("padding", "12px"),
                        ("text-align", "left"),
                        ("background-color", "#1f2937"),
                        ("color", "white"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "19px"),
                        ("padding", "12px"),
                        ("text-align", "left"),
                    ],
                },
            ]
        )
    )

    st.table(styled)


def _average_shap_values(shap_values: object) -> np.ndarray | None:
    """
    Converts SHAP output into one importance vector.

    Supports:
    - list of outputs
    - ndarray: (samples, features)
    - ndarray: (samples, features, outputs)
    - ndarray: (outputs, samples, features)
    """
    if isinstance(shap_values, list):
        return np.mean(
            [np.abs(output_values[0]) for output_values in shap_values],
            axis=0,
        )

    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        # Common multi-output shape: (samples, features, outputs)
        if shap_array.shape[0] == 1:
            return np.mean(np.abs(shap_array[0]), axis=1)

        # Alternative multi-output shape: (outputs, samples, features)
        if shap_array.shape[1] == 1:
            return np.mean(np.abs(shap_array[:, 0, :]), axis=0)

        return np.mean(np.abs(shap_array), axis=(0, 2))

    if shap_array.ndim == 2:
        return np.abs(shap_array[0])

    if shap_array.ndim == 1:
        return np.abs(shap_array)

    return None


def _get_single_model_importance(
    selected_model: str,
    model,
    selected_features: list[str],
) -> pd.DataFrame | None:
    """
    Normal feature importance for one model.
    Supports Ridge Regression and Random Forest.
    """
    if selected_model == "ridge_regression":
        ridge_model = model.named_steps["model"]
        values = np.abs(ridge_model.coef_)

        if values.ndim == 2:
            values = values.mean(axis=0)

    elif selected_model == "random_forest":
        values = model.feature_importances_

    else:
        return None

    return pd.DataFrame(
        {
            "Feature": selected_features,
            "Importance": values,
        }
    )


def get_feature_importance(bundle: dict) -> pd.DataFrame | None:
    """
    Returns normal feature importance.

    Supports:
    - Old single Ridge Regression model
    - Old single Random Forest model
    - New horizon-specific ensemble with Ridge/Random Forest horizon models
    """
    metadata = bundle["metadata"]

    # -----------------------------------------------------
    # New horizon-specific ensemble
    # -----------------------------------------------------
    if metadata.get("forecast_method") == "horizon_specific_direct_multi_horizon":
        horizon_models = bundle.get("horizon_models", {})
        frames = []

        for horizon, payload in horizon_models.items():
            selected_model = payload.get("selected_model")
            framework = payload.get("framework")

            if framework != "sklearn":
                continue

            model = payload.get("model")
            selected_features = payload.get("selected_features", [])

            frame = _get_single_model_importance(
                selected_model=selected_model,
                model=model,
                selected_features=selected_features,
            )

            if frame is not None:
                frame["Horizon"] = f"Day {horizon}"
                frame["Model"] = selected_model
                frames.append(frame)

        if not frames:
            return None

        all_importance = pd.concat(frames, ignore_index=True)

        final_importance = (
            all_importance.groupby("Feature", as_index=False)["Importance"]
            .mean()
            .sort_values("Importance", ascending=False)
        )

        final_importance["Importance"] = final_importance["Importance"].round(4)

        return final_importance

    # -----------------------------------------------------
    # Old single-model format
    # -----------------------------------------------------
    model_name = metadata["best_model"]
    selected_features = metadata["selected_features"]
    model = bundle["model"]

    importance = _get_single_model_importance(
        selected_model=model_name,
        model=model,
        selected_features=selected_features,
    )

    if importance is None:
        return None

    importance = importance.sort_values("Importance", ascending=False)
    importance["Importance"] = importance["Importance"].round(4)

    return importance


@st.cache_data(ttl=900, show_spinner="Computing SHAP explanation...")
def get_shap_explanation(model_signature: str) -> pd.DataFrame | None:
    """
    Computes SHAP explanation for the latest prediction row.

    Supports:
    - Old single Ridge Regression model
    - Old single Random Forest model
    - New horizon-specific ensemble:
        Day 1 model
        Day 2 model
        Day 3 model

    For horizon-specific models:
    - SHAP is computed separately for each horizon.
    - Final SHAP importance is averaged across supported horizons.
    """

    bundle = load_local_best_model(MODEL_DIR)
    metadata = bundle["metadata"]

    # -----------------------------------------------------
    # Case 1: New horizon-specific ensemble
    # -----------------------------------------------------
    if metadata.get("forecast_method") == "horizon_specific_direct_multi_horizon":
        horizon_models = bundle.get("horizon_models", {})

        supported_horizons = []

        for horizon, payload in horizon_models.items():
            selected_model = payload.get("selected_model")
            framework = payload.get("framework")

            if framework == "sklearn" and selected_model in [
                "ridge_regression",
                "random_forest",
            ]:
                supported_horizons.append((horizon, payload))

        if not supported_horizons:
            return None

        all_features = sorted(
            {
                feature
                for _, payload in supported_horizons
                for feature in payload["selected_features"]
            }
        )

        latest_row, latest_date, source = _latest_completed_row(all_features)

        horizon_shap_frames = []

        for horizon, payload in supported_horizons:
            selected_model = payload["selected_model"]
            selected_features = payload["selected_features"]
            model = payload["model"]

            X = pd.DataFrame([latest_row[selected_features].to_dict()])

            if selected_model == "ridge_regression":
                scaler = model.named_steps["scaler"]
                ridge_model = model.named_steps["model"]

                X_scaled = scaler.transform(X)

                # In standardized space, zero represents average training conditions.
                background = np.zeros((1, X_scaled.shape[1]))

                explainer = shap.LinearExplainer(ridge_model, background)
                shap_values = explainer.shap_values(X_scaled)
                explainer_name = "SHAP LinearExplainer"

            elif selected_model == "random_forest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                explainer_name = "SHAP TreeExplainer"

            else:
                continue

            values = _average_shap_values(shap_values)

            if values is None:
                continue

            horizon_df = pd.DataFrame(
                {
                    "Feature": selected_features,
                    "SHAP Importance": values,
                    "Horizon": f"Day {horizon}",
                    "Model": selected_model,
                    "Explainer": explainer_name,
                    "Source Row": f"{latest_date:%Y-%m-%d} from {source}",
                }
            )

            horizon_shap_frames.append(horizon_df)

        if not horizon_shap_frames:
            return None

        all_shap = pd.concat(horizon_shap_frames, ignore_index=True)

        final_shap = (
            all_shap.groupby("Feature", as_index=False)["SHAP Importance"]
            .mean()
            .sort_values("SHAP Importance", ascending=False)
        )

        final_shap["SHAP Importance"] = final_shap["SHAP Importance"].round(4)
        final_shap["Explainer"] = "Horizon-specific SHAP"
        final_shap["Source Row"] = all_shap["Source Row"].iloc[0]

        return final_shap

    # -----------------------------------------------------
    # Case 2: Old single-model format
    # -----------------------------------------------------
    model_name = metadata["best_model"]
    selected_features = metadata["selected_features"]
    model = bundle["model"]

    latest_row, latest_date, source = _latest_completed_row(selected_features)
    X = pd.DataFrame([latest_row[selected_features].to_dict()])

    if model_name == "ridge_regression":
        scaler = model.named_steps["scaler"]
        ridge_model = model.named_steps["model"]

        X_scaled = scaler.transform(X)
        background = np.zeros((1, X_scaled.shape[1]))

        explainer = shap.LinearExplainer(ridge_model, background)
        shap_values = explainer.shap_values(X_scaled)
        explainer_name = "SHAP LinearExplainer"

    elif model_name == "random_forest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        explainer_name = "SHAP TreeExplainer"

    else:
        return None

    values = _average_shap_values(shap_values)

    if values is None:
        return None

    shap_df = pd.DataFrame(
        {
            "Feature": selected_features,
            "SHAP Importance": values,
        }
    ).sort_values("SHAP Importance", ascending=False)

    shap_df["SHAP Importance"] = shap_df["SHAP Importance"].round(4)
    shap_df["Source Row"] = f"{latest_date:%Y-%m-%d} from {source}"
    shap_df["Explainer"] = explainer_name

    return shap_df


def get_forecast_uncertainty(metadata: dict) -> int:
    """
    Uses average test MAE internally for uncertainty range.
    The dashboard does not show RMSE/MAE/R² metrics separately.
    """
    test_metrics = metadata.get("test_metrics", {})
    avg_mae = test_metrics.get("avg_mae", test_metrics.get("mae", 13.0))

    try:
        return round(float(avg_mae))
    except Exception:
        return 13


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Dashboard Controls")

st.sidebar.info(
    "Forecast and SHAP explanations are cached for 15 minutes. "
    "Use refresh only when needed."
)

if st.sidebar.button("🔄 Refresh forecast"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.title("🌫️ Islamabad AQI Forecast — Next 3 Days")

st.caption(
    "Daily AQI forecast generated from weather and pollutant features using direct multi-horizon forecasting."
)

model_signature = get_model_signature()

try:
    bundle = cached_model_bundle(model_signature)
    metadata = bundle["metadata"]

except Exception as exc:
    st.error("Failed to load local trained model.")
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# LOAD FORECAST SAFELY
# ---------------------------------------------------------

forecast = None

try:
    forecast = cached_direct_forecast(model_signature)

except Exception as exc:
    st.error("Could not generate the 3-day forecast.")
    st.warning(
        "This usually happens when live API data is temporarily unavailable, returns 502, "
        "or the network has SSL/time-out issues. Try Refresh forecast after 1–2 minutes."
    )
    with st.expander("Show forecast error details"):
        st.exception(exc)


uncertainty = get_forecast_uncertainty(metadata)


# ---------------------------------------------------------
# FORECAST CARDS
# ---------------------------------------------------------

st.subheader("Next 3 Days AQI Forecast")

if forecast is not None and not forecast.empty:
    cols = st.columns(3)

    for i, row in forecast.iterrows():
        predicted_aqi = float(row["predicted_daily_aqi"])
        category = row["category"]
        short_category = get_short_category(category)
        message, level = get_alert_message(predicted_aqi)

        with cols[i]:
            st.metric(
                label=row["date"],
                value=f"{predicted_aqi:.1f} ± {uncertainty}",
                delta=short_category,
            )
            show_alert(message, level)

    st.caption(
        "Forecast values are estimates and may change as new API data arrives."
    )

else:
    st.info(
        "3-day forecast is temporarily unavailable because live feature data could not be fetched."
    )

st.divider()


# ---------------------------------------------------------
# FORECAST CHART AND RAW TABLE
# ---------------------------------------------------------

if forecast is not None and not forecast.empty:
    st.subheader("Forecast Trend")

    chart_df = forecast.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.set_index("date")

    st.line_chart(chart_df["predicted_daily_aqi"])

    st.subheader("Raw Forecast Table")

    forecast_display = forecast.copy()
    forecast_display = forecast_display.rename(
        columns={
            "date": "Date",
            "predicted_daily_aqi": "Predicted Daily AQI",
            "category": "Category",
        }
    )

    readable_table(forecast_display)

st.divider()


# ---------------------------------------------------------
# FEATURE SELECTION AND IMPORTANCE
# ---------------------------------------------------------

st.subheader("Feature Selection and Importance")

selected_features = metadata.get("selected_features", [])

# For horizon-specific model, metadata["selected_features"] is the union across horizons.
st.write("Selected features used by the selected model/system:")

selected_features_df = pd.DataFrame(
    {
        "No.": range(1, len(selected_features) + 1),
        "Selected Feature": selected_features,
    }
)

readable_table(selected_features_df)

importance = get_feature_importance(bundle)

if importance is not None:
    st.subheader("Feature Importance Chart")

    chart_importance = importance.set_index("Feature")["Importance"]
    st.bar_chart(chart_importance)

    st.subheader("Feature Importance Table")
    readable_table(importance)

else:
    st.info(
        "Normal feature importance is available for Ridge Regression and Random Forest models. "
        "If TensorFlow or baseline models are selected for all horizons, use SHAP/LIME separately."
    )

st.divider()


# ---------------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------------

st.subheader("SHAP Explainability")

try:
    shap_df = get_shap_explanation(model_signature)

    if shap_df is not None and not shap_df.empty:
        explainer_used = shap_df["Explainer"].iloc[0]

        st.write(
            f"{explainer_used} was used to explain the selected model/system. "
            "For the horizon-specific 3-day forecast, SHAP values are averaged across supported Day 1, Day 2, and Day 3 models."
        )

        shap_chart = shap_df.set_index("Feature")["SHAP Importance"]
        st.bar_chart(shap_chart)

        st.subheader("SHAP Explanation Table")
        readable_table(shap_df)

    else:
        st.info(
            "SHAP explanation is available for Ridge Regression and Random Forest horizon models. "
            "If TensorFlow or baseline models are selected for all horizons, use a separate SHAP/LIME notebook explanation."
        )

except Exception as exc:
    st.warning("SHAP explanation could not be generated.")
    with st.expander("Show SHAP error details"):
        st.exception(exc)

st.divider()


# ---------------------------------------------------------
# PROJECT PIPELINE SUMMARY
# ---------------------------------------------------------

st.subheader("Project Pipeline Summary")

st.markdown(
    """
    **Pipeline used in this project:**

    1. Fetch weather and pollutant data for Islamabad.
    2. Convert hourly data into daily AQI forecasting features.
    3. Store processed features in **Hopsworks Feature Store**.
    4. Train **Ridge Regression**, **Random Forest**, and **TensorFlow MLP**.
    5. Select the best model for direct daily AQI forecasting.
    6. Register the selected model in **Hopsworks Model Registry**.
    7. Generate the next 3 days AQI forecast using **direct multi-horizon forecasting**.
    8. Display AQI forecast, AQI categories, health alerts, selected features, feature importance, and **SHAP explainability**.
    """
)

st.info(
    "Direct multi-horizon forecasting predicts Day 1, Day 2, and Day 3 AQI directly. "
    "This avoids error accumulation that can occur in recursive forecasting."
)