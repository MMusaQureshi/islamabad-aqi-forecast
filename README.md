# Islamabad AQI Forecast — Next 3 Days

A serverless machine learning project for forecasting the **Air Quality Index (AQI) of Islamabad for the next 3 days** using weather and air pollution data.

The project includes automated data collection, feature engineering, cloud feature storage, model training, model registry, CI/CD automation, Streamlit dashboard deployment, AQI health alerts, and SHAP explainability.

🔗 **Live Demo:** https://islamabad-aqi-forecast-krk8fy8vtnudqqeey2abhr.streamlit.app/

---

## Project Overview

This project predicts Islamabad’s AQI for the next 3 days using a daily forecasting approach.

The system collects hourly weather and pollutant data, converts it into daily machine learning features, stores processed features in Hopsworks Feature Store, trains multiple ML models, selects the best model, registers it in Hopsworks Model Registry, and displays predictions through a Streamlit dashboard.

---

## Key Features

* Fetches hourly weather and air quality data from Open-Meteo APIs
* Converts hourly data into daily AQI forecasting features
* Uses Hopsworks Feature Store for cloud-based feature storage
* Performs historical data backfill for model training
* Trains multiple models:

  * Ridge Regression
  * Random Forest
  * TensorFlow MLP
* Uses direct multi-horizon forecasting for the next 3 days
* Supports horizon-specific forecasting models
* Registers trained models in Hopsworks Model Registry
* Automates feature and training pipelines using GitHub Actions
* Provides a Streamlit dashboard for AQI predictions
* Shows AQI category and health alerts
* Includes feature importance and SHAP explainability

---

## Forecasting Approach

The project uses **direct multi-horizon forecasting**.

Instead of recursively predicting one day at a time, the model directly predicts:

* Day 1 AQI
* Day 2 AQI
* Day 3 AQI

This avoids recursive error accumulation, where an incorrect Day 1 prediction can negatively affect Day 2 and Day 3 predictions.

The latest version also supports **horizon-specific models**, meaning each forecast horizon can have its own selected model.

Example:

* Day 1 model predicts tomorrow’s AQI
* Day 2 model predicts the day after tomorrow’s AQI
* Day 3 model predicts the third day’s AQI

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Hopsworks Feature Store
* Hopsworks Model Registry
* GitHub Actions
* Streamlit
* SHAP
* Open-Meteo APIs
* Git / GitHub

---

## Project Structure

```text
aqi_islamabad_daily_forecast/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── api_client.py
│   ├── backfill.py
│   ├── config.py
│   ├── feature_engineering.py
│   ├── feature_pipeline.py
│   ├── hopsworks_utils.py
│   ├── predict.py
│   └── train_pipeline.py
│
├── .github/
│   └── workflows/
│       ├── hourly_feature_pipeline.yml
│       └── daily_training_pipeline.yml
│
├── experiments/
│   └── direct_rf_v2_results.md
│
├── model_artifacts/
│   └── best_model/
│       ├── metadata.json
│       └── model files
│
├── notebooks/
│   └── README.md
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Variables

Create a `.env` file locally using `.env.example`.

Example:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=aqi_musa

CITY=Islamabad
LATITUDE=33.6844
LONGITUDE=73.0479
TIMEZONE=Asia/Karachi
```

Do not commit `.env` to GitHub.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MMusaQureshi/islamabad-aqi-forecast.git
cd islamabad-aqi-forecast
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally

Test the API client:

```bash
python -m src.api_client
```

Run historical backfill:

```bash
python -m src.backfill --days 365 --write
```

Run training pipeline:

```bash
python -m src.train_pipeline
```

Run prediction:

```bash
python -m src.predict
```

Run Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

## Feature Pipeline

The feature pipeline fetches recent hourly weather and pollutant data, converts it into completed daily feature rows, and inserts those rows into Hopsworks Feature Store.

Run manually:

```bash
python -m src.feature_pipeline
```

The hourly feature pipeline is also automated using GitHub Actions.

---

## Training Pipeline

The training pipeline reads processed features from Hopsworks Feature Store, creates direct 3-day targets, trains multiple ML models, compares their performance, selects the best model, saves model artifacts, and registers the selected model in Hopsworks Model Registry.

Run manually:

```bash
python -m src.train_pipeline
```

---

## Prediction Pipeline

The prediction pipeline loads the selected local model artifacts and generates the next 3 days AQI forecast.

Run manually:

```bash
python -m src.predict
```

Example output:

```text
         date  predicted_daily_aqi                        category
0  2026-06-06                108.1  Unhealthy for Sensitive Groups
1  2026-06-07                115.5  Unhealthy for Sensitive Groups
2  2026-06-08                107.0  Unhealthy for Sensitive Groups
```

---

## Dashboard

The Streamlit dashboard displays:

* Next 3 days AQI forecast
* AQI category
* Health alert messages
* Forecast trend chart
* Raw forecast table
* Selected model features
* Feature importance
* SHAP explainability

The dashboard does not display today’s AQI section or visible RMSE/MSE/R² metrics in the final UI.

---

## SHAP Explainability

The project uses SHAP for model explanation.

Supported explainers:

* SHAP LinearExplainer for Ridge Regression
* SHAP TreeExplainer for Random Forest

For horizon-specific models, SHAP values are calculated separately for supported Day 1, Day 2, and Day 3 models, then averaged to show overall feature contribution.

---

## CI/CD Automation

This project uses GitHub Actions for pipeline automation.

### Hourly Feature Pipeline

Workflow file:

```text
.github/workflows/hourly_feature_pipeline.yml
```

Purpose:

* Runs automatically every hour
* Fetches recent hourly data
* Generates daily features
* Updates Hopsworks Feature Store

### Daily Training Pipeline

Workflow file:

```text
.github/workflows/daily_training_pipeline.yml
```

Purpose:

* Runs automatically once per day
* Reads features from Hopsworks Feature Store
* Retrains models
* Registers updated model artifacts in Hopsworks Model Registry

Both workflows can also be triggered manually from the GitHub Actions tab using `workflow_dispatch`.

---

## GitHub Actions Secrets

Add these secrets in GitHub:

```text
Repository → Settings → Secrets and variables → Actions
```

Required secrets:

```text
HOPSWORKS_API_KEY
HOPSWORKS_PROJECT_NAME
```

Example project name:

```text
aqi_musa
```

---

## Streamlit Community Cloud Deployment

Deploy using Streamlit Community Cloud.

Deployment settings:

```text
Repository: MMusaQureshi/islamabad-aqi-forecast
Branch: main
Main file path: app/streamlit_app.py
Python version: 3.11
```

Add secrets in Streamlit Cloud:

```toml
HOPSWORKS_API_KEY = "your_real_hopsworks_api_key"
HOPSWORKS_PROJECT_NAME = "aqi_musa"

CITY = "Islamabad"
LATITUDE = "33.6844"
LONGITUDE = "73.0479"
TIMEZONE = "Asia/Karachi"
```

Do not upload `.env` to GitHub or Streamlit.

---

## Important Notes

* The model is trained for Islamabad only.
* Forecasting is daily, not hourly.
* The dashboard predicts the next 3 days, starting from tomorrow.
* AQI predictions are estimates and may change as new API data arrives.
* API failures from Open-Meteo or Hopsworks may happen temporarily, so retry and fallback logic is included.
* Model artifacts are included for Streamlit deployment because the dashboard loads local model files.

---

## Final Project Status

Implemented requirements:

* Feature pipeline
* Historical data backfill
* Hopsworks Feature Store
* Multiple ML models
* Best model selection
* Hopsworks Model Registry
* GitHub Actions CI/CD
* Streamlit dashboard
* AQI alerts
* SHAP explainability
* Direct next 3-day AQI forecasting

---

## Author

Developed by **MMusaQureshi** as an end-to-end serverless AQI forecasting machine learning project for Islamabad.
