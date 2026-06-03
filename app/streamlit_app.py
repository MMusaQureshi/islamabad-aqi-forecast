from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.api_client import fetch_realtime_plus_forecast_hourly
from src.predict import load_local_best_model, direct_three_day_forecast


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
    Gets latest available AQI and pollutant values from the API.
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

    For Ridge Regression:
    - Direct forecasting has 3 outputs.
    - Ridge coefficients become a 2D matrix: horizons × features.
    - We average absolute coefficient values across Day 1, Day 2, and Day 3.

    For Random Forest:
    - Built-in feature_importances_ is used.
    """
    metadata = bundle["metadata"]
    model_name = metadata["best_model"]
    selected_features = metadata["selected_features"]
    model = bundle["model"]

    if model_name == "ridge_regression":
        ridge_model = model.named_steps["model"]
        values = abs(ridge_model.coef_)

        if values.ndim == 2:
            values = values.mean(axis=0)

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


def get_metric(test_metrics: dict, new_key: str, old_key: str, default: float = 0.0) -> float:
    """
    Supports both old recursive metadata and new direct multi-horizon metadata.
    New direct model uses avg_rmse, avg_mae, avg_r2.
    Old model used rmse, mae, r2.
    """
    return float(test_metrics.get(new_key, test_metrics.get(old_key, default)))


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

st.title("🌫️ Islamabad AQI Forecast — Next 3 Days")

st.caption(
    "Daily AQI forecast generated from weather and pollutant features using direct multi-horizon forecasting."
)

try:
    # Load local best model saved by src.train_pipeline
    bundle = load_local_best_model()
    metadata = bundle["metadata"]

    # Generate direct 3-day forecast
    forecast = direct_three_day_forecast(bundle, days=3)

    # Latest observed AQI
    latest = get_latest_observed_aqi()

    # Model metadata
    test_metrics = metadata.get("test_metrics", {})
    validation_comparison = pd.DataFrame(metadata.get("validation_comparison", []))

    avg_test_rmse = get_metric(test_metrics, "avg_rmse", "rmse")
    avg_test_mae = get_metric(test_metrics, "avg_mae", "mae", default=13.0)
    avg_test_r2 = get_metric(test_metrics, "avg_r2", "r2")

    uncertainty = round(avg_test_mae) if avg_test_mae else 13

    forecast_method = metadata.get("forecast_method", "direct_multi_horizon")
    training_rows = metadata.get("training_rows", "N/A")

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
        f"±{uncertainty} AQI is based on the selected model's average test MAE. "
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

    metric_cols = st.columns(5)

    with metric_cols[0]:
        st.metric("Best Model", metadata.get("best_model", "N/A"))

    with metric_cols[1]:
        st.metric("Avg Test RMSE", round(avg_test_rmse, 2))

    with metric_cols[2]:
        st.metric("Avg Test MAE", round(avg_test_mae, 2))

    with metric_cols[3]:
        st.metric("Avg Test R²", round(avg_test_r2, 2))

    with metric_cols[4]:
        st.metric("Training Rows", training_rows)

    st.caption(f"Forecast method: {forecast_method}")

    # Day-wise test metrics for direct forecasting
    day_metric_rows = []

    for day in [1, 2, 3]:
        day_metric_rows.append(
            {
                "Forecast Horizon": f"Day {day}",
                "Test RMSE": round(test_metrics.get(f"day_{day}_rmse", 0), 3),
                "Test MAE": round(test_metrics.get(f"day_{day}_mae", 0), 3),
                "Test R²": round(test_metrics.get(f"day_{day}_r2", 0), 3),
            }
        )

    day_metrics_df = pd.DataFrame(day_metric_rows)

    if day_metrics_df[["Test RMSE", "Test MAE", "Test R²"]].sum().sum() != 0:
        st.subheader("Day-wise Test Metrics")
        readable_table(day_metrics_df)

    # Validation comparison
    if not validation_comparison.empty:
        st.subheader("Validation Comparison of 3 Models")

        validation_display = validation_comparison.copy()

        validation_display = validation_display.rename(
            columns={
                "model": "Model",
                "avg_rmse": "Avg Validation RMSE",
                "avg_mae": "Avg Validation MAE",
                "avg_r2": "Avg Validation R²",
                "day_1_mae": "Day 1 MAE",
                "day_2_mae": "Day 2 MAE",
                "day_3_mae": "Day 3 MAE",
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
        5. Select the best model using **average RMSE**, **average MAE**, and **average R²** across the 3 forecast days.
        6. Register the best model in **Hopsworks Model Registry**.
        7. Generate next 3 days AQI forecast using **direct multi-horizon forecasting**.
        8. Display AQI forecast, categories, alerts, model metrics, and feature importance.
        """
    )

    st.info(
        "Direct multi-horizon forecasting predicts Day 1, Day 2, and Day 3 AQI directly. "
        "This avoids error accumulation that can occur in recursive forecasting."
    )

    # -----------------------------------------------------
    # RAW METADATA
    # -----------------------------------------------------

    with st.expander("Show Raw Model Metadata"):
        st.json(metadata)

except Exception as exc:
    st.error("Dashboard failed to load.")
    st.exception(exc)