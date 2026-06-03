from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.api_client import fetch_realtime_plus_forecast_hourly
from src.predict import load_local_best_model, recursive_daily_forecast


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
    /* Full page */
    .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 96% !important;
    }

    /* General text */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    /* Main title */
    h1 {
        font-size: 46px !important;
        font-weight: 850 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Section headings */
    h2, h3 {
        font-size: 32px !important;
        font-weight: 800 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* Paragraph / markdown */
    .stMarkdown {
        font-size: 19px !important;
        line-height: 1.6 !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        font-size: 17px !important;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 19px !important;
        font-weight: 750 !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 40px !important;
        font-weight: 850 !important;
    }

    /* Metric delta/category */
    [data-testid="stMetricDelta"] {
        font-size: 17px !important;
        font-weight: 750 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        font-size: 19px !important;
        font-weight: 650 !important;
        padding: 1rem !important;
    }

    /* Normal HTML tables */
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

    /* Streamlit dataframe containers */
    [data-testid="stDataFrame"] {
        font-size: 19px !important;
    }

    /* Code / JSON text */
    pre, code {
        font-size: 17px !important;
    }

    /* Divider spacing */
    hr {
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_aqi_category(aqi: float) -> str:
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
    """
    Shows a bigger, cleaner table than default st.dataframe.
    Good for final project dashboard screenshots.
    """
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


def get_latest_observed_aqi() -> dict:
    """
    Gets the latest available AQI and pollutant values from the API.
    """
    hourly = fetch_realtime_plus_forecast_hourly(past_days=2, forecast_days=3)
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])

    now = pd.Timestamp.now()
    observed = hourly[hourly["timestamp"] <= now].copy()

    if observed.empty:
        latest = hourly.iloc[-1]
    else:
        latest = observed.iloc[-1]

    aqi = float(latest["us_aqi"])

    return {
        "timestamp": latest["timestamp"],
        "aqi": round(aqi, 1),
        "category": get_aqi_category(aqi),
        "pm2_5": round(float(latest["pm2_5"]), 1),
        "pm10": round(float(latest["pm10"]), 1),
        "ozone": round(float(latest["ozone"]), 1),
        "nitrogen_dioxide": round(float(latest["nitrogen_dioxide"]), 1),
        "carbon_monoxide": round(float(latest["carbon_monoxide"]), 1),
    }


def get_feature_importance(bundle: dict) -> pd.DataFrame | None:
    """
    Returns feature importance for Ridge Regression or Random Forest.
    For Ridge, absolute coefficient values are used.
    For Random Forest, built-in feature importance is used.
    """
    metadata = bundle["metadata"]
    model_name = metadata["best_model"]
    selected_features = metadata["selected_features"]
    model = bundle["model"]

    if model_name == "ridge_regression":
        ridge_model = model.named_steps["model"]
        values = abs(ridge_model.coef_)

    elif model_name == "random_forest":
        values = model.feature_importances_

    else:
        return None

    importance = pd.DataFrame(
        {
            "Feature": selected_features,
            "Importance": values,
        }
    ).sort_values("Importance", ascending=False)

    importance["Importance"] = importance["Importance"].round(4)

    return importance


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.title("🌫️ Islamabad AQI Forecast — Next 3 Days")

st.caption(
    "Daily AQI forecast generated from weather and pollutant features using recursive forecasting."
)

try:
    # Load local best model saved by src.train_pipeline
    bundle = load_local_best_model()
    metadata = bundle["metadata"]

    # Generate forecast
    forecast = recursive_daily_forecast(bundle, days=3)

    # Latest observed AQI
    latest = get_latest_observed_aqi()

    # Model metadata
    test_metrics = metadata.get("test_metrics", {})
    validation_comparison = pd.DataFrame(metadata.get("validation_comparison", []))

    test_mae = test_metrics.get("mae", None)
    uncertainty = round(test_mae) if test_mae is not None else 13

    # -----------------------------------------------------
    # CURRENT AQI
    # -----------------------------------------------------

    st.subheader("Current / Latest Observed AQI")

    current_cols = st.columns(6)

    with current_cols[0]:
        st.metric(
            label="Latest AQI",
            value=latest["aqi"],
            delta=get_short_category(latest["category"]),
        )

    with current_cols[1]:
        st.metric("PM2.5", latest["pm2_5"])

    with current_cols[2]:
        st.metric("PM10", latest["pm10"])

    with current_cols[3]:
        st.metric("Ozone", latest["ozone"])

    with current_cols[4]:
        st.metric("NO₂", latest["nitrogen_dioxide"])

    with current_cols[5]:
        st.metric("CO", latest["carbon_monoxide"])

    st.caption(f"Latest API timestamp: {latest['timestamp']}")

    current_message, current_level = get_alert_message(latest["aqi"])
    show_alert(current_message, current_level)

    st.divider()

    # -----------------------------------------------------
    # FORECAST CARDS
    # -----------------------------------------------------

    st.subheader("Next 3 Days AQI Forecast")

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
        f"±{uncertainty} AQI is based on the selected model's test MAE. "
        "Forecast values are estimates and may change as new API data arrives."
    )

    st.divider()

    # -----------------------------------------------------
    # FORECAST CHART
    # -----------------------------------------------------

    st.subheader("Forecast Trend")

    chart_df = forecast.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.set_index("date")

    st.line_chart(chart_df["predicted_daily_aqi"])

    # -----------------------------------------------------
    # RAW FORECAST TABLE
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    st.subheader("Model Performance")

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Best Model", metadata.get("best_model", "N/A"))

    with metric_cols[1]:
        st.metric("Test RMSE", round(test_metrics.get("rmse", 0), 2))

    with metric_cols[2]:
        st.metric("Test MAE", round(test_metrics.get("mae", 0), 2))

    with metric_cols[3]:
        st.metric("Test R²", round(test_metrics.get("r2", 0), 2))

    if not validation_comparison.empty:
        st.subheader("Validation Comparison of 3 Models")

        validation_display = validation_comparison.copy()
        validation_display = validation_display.rename(
            columns={
                "model": "Model",
                "rmse": "Validation RMSE",
                "mae": "Validation MAE",
                "r2": "Validation R²",
            }
        )

        numeric_cols = validation_display.select_dtypes(include="number").columns
        validation_display[numeric_cols] = validation_display[numeric_cols].round(3)

        readable_table(validation_display)

    st.divider()

    # -----------------------------------------------------
    # FEATURE SELECTION AND IMPORTANCE
    # -----------------------------------------------------

    st.subheader("Feature Selection and Importance")

    selected_features = metadata.get("selected_features", [])

    st.write("Selected features used by the best model:")

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
            "Feature importance is not directly available for the TensorFlow model. "
            "Use SHAP separately for deep learning explanations."
        )

    st.divider()

    # -----------------------------------------------------
    # PROJECT PIPELINE SUMMARY
    # -----------------------------------------------------

    st.subheader("Project Pipeline Summary")

    st.markdown(
        """
        **Pipeline used in this project:**

        1. Fetch weather and pollutant data for Islamabad.
        2. Convert hourly data into daily AQI forecasting features.
        3. Store processed features in **Hopsworks Feature Store**.
        4. Train **Ridge Regression**, **Random Forest**, and **TensorFlow MLP**.
        5. Select the best model using **RMSE**, **MAE**, and **R²**.
        6. Register the best model in **Hopsworks Model Registry**.
        7. Generate next 3 days AQI forecast using **recursive forecasting**.
        8. Display AQI forecast, categories, alerts, model metrics, and feature importance.
        """
    )

    # -----------------------------------------------------
    # RAW METADATA
    # -----------------------------------------------------

    with st.expander("Show Raw Model Metadata"):
        st.json(metadata)

except Exception as exc:
    st.error("Dashboard failed to load.")
    st.exception(exc)