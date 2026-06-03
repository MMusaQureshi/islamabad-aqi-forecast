# Islamabad AQI Daily Forecasting Project

This scaffold implements the project as **daily AQI forecasting for the next 3 days**, not 72 hourly outputs.

Important design decision:

- The APIs still return hourly data.
- We aggregate hourly weather and pollutant data into **daily features**.
- The model predicts **daily AQI** for:
  - Tomorrow
  - Day after tomorrow
  - Third day
- The dashboard displays daily forecast cards and AQI alerts.

## First run

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
```

Then edit `.env` and add your Hopsworks project name and API key.

## Step 1: test data fetching

```bash
python -m src.api_client
```

## Step 2: create 90-day backfill in Hopsworks Feature Store

```bash
python -m src.backfill --days 90 --write
```

## Step 3: train 3 models and register the best one

```bash
python -m src.train_pipeline
```

## Step 4: run dashboard

```bash
streamlit run app/streamlit_app.py
```
