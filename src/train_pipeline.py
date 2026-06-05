from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS
from src.hopsworks_utils import read_daily_features, register_sklearn_model


TARGET_COLUMNS = [
    "target_aqi_day_1",
    "target_aqi_day_2",
    "target_aqi_day_3",
]


def add_direct_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).copy()

    df["target_aqi_day_1"] = df.groupby("city")["aqi_mean"].shift(-1)
    df["target_aqi_day_2"] = df.groupby("city")["aqi_mean"].shift(-2)
    df["target_aqi_day_3"] = df.groupby("city")["aqi_mean"].shift(-3)

    return df


def chronological_split_df(df: pd.DataFrame):
    n = len(df)

    # More training data because dataset is still small.
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_features_for_horizon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_features: int = 10,
) -> list[str]:
    selector_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=4,
        max_depth=5,
        max_features="sqrt",
        bootstrap=True,
    )

    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    selected_features = X_train.columns[selector.get_support()].tolist()

    if len(selected_features) < min_features:
        importances = pd.Series(selector_model.feature_importances_, index=X_train.columns)
        selected_features = importances.sort_values(ascending=False).head(min_features).index.tolist()

    return selected_features


def build_tensorflow_model(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                16,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(
                8,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],
    )

    return model


def train_single_horizon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> dict:
    X_train_full = train_df[FEATURE_COLUMNS].copy()
    X_val_full = val_df[FEATURE_COLUMNS].copy()
    X_test_full = test_df[FEATURE_COLUMNS].copy()

    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    y_test = test_df[target_col].copy()

    selected_features = select_features_for_horizon(X_train_full, y_train)

    X_train = X_train_full[selected_features]
    X_val = X_val_full[selected_features]
    X_test = X_test_full[selected_features]

    candidates = {}

    # -----------------------------------------------------
    # Baseline 1: persistence
    # Predict future AQI = latest/current AQI.
    # -----------------------------------------------------
    if "aqi_mean" in val_df.columns:
        val_pred = val_df["aqi_mean"].values
        test_pred = test_df["aqi_mean"].values

        candidates["baseline_persistence"] = {
            "framework": "baseline",
            "model": None,
            "selected_features": selected_features,
            "val_metrics": regression_metrics(y_val, val_pred),
            "test_metrics": regression_metrics(y_test, test_pred),
        }

    # -----------------------------------------------------
    # Baseline 2: 3-day rolling mean
    # -----------------------------------------------------
    if "aqi_rolling_mean_3d" in val_df.columns:
        val_pred = val_df["aqi_rolling_mean_3d"].values
        test_pred = test_df["aqi_rolling_mean_3d"].values

        if not np.isnan(val_pred).any() and not np.isnan(test_pred).any():
            candidates["baseline_rolling_3d"] = {
                "framework": "baseline",
                "model": None,
                "selected_features": selected_features,
                "val_metrics": regression_metrics(y_val, val_pred),
                "test_metrics": regression_metrics(y_test, test_pred),
            }

    # -----------------------------------------------------
    # Baseline 3: 7-day rolling mean
    # -----------------------------------------------------
    if "aqi_rolling_mean_7d" in val_df.columns:
        val_pred = val_df["aqi_rolling_mean_7d"].values
        test_pred = test_df["aqi_rolling_mean_7d"].values

        if not np.isnan(val_pred).any() and not np.isnan(test_pred).any():
            candidates["baseline_rolling_7d"] = {
                "framework": "baseline",
                "model": None,
                "selected_features": selected_features,
                "val_metrics": regression_metrics(y_val, val_pred),
                "test_metrics": regression_metrics(y_test, test_pred),
            }

    # -----------------------------------------------------
    # Model 1: Ridge Regression
    # -----------------------------------------------------
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RidgeCV(
                    alphas=[0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
                ),
            ),
        ]
    )

    ridge.fit(X_train, y_train)
    ridge_val_pred = ridge.predict(X_val)
    ridge_test_pred = ridge.predict(X_test)

    candidates["ridge_regression"] = {
        "framework": "sklearn",
        "model": ridge,
        "selected_features": selected_features,
        "val_metrics": regression_metrics(y_val, ridge_val_pred),
        "test_metrics": regression_metrics(y_test, ridge_test_pred),
    }

    # -----------------------------------------------------
    # Model 2: Random Forest
    # -----------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=5,
        max_depth=5,
        max_features="sqrt",
        max_samples=0.8,
        bootstrap=True,
    )

    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_test_pred = rf.predict(X_test)

    candidates["random_forest"] = {
        "framework": "sklearn",
        "model": rf,
        "selected_features": selected_features,
        "val_metrics": regression_metrics(y_val, rf_val_pred),
        "test_metrics": regression_metrics(y_test, rf_test_pred),
    }

    # -----------------------------------------------------
    # Model 3: TensorFlow MLP
    # -----------------------------------------------------
    try:
        import tensorflow as tf

        tf_scaler = StandardScaler()
        X_train_scaled = tf_scaler.fit_transform(X_train)
        X_val_scaled = tf_scaler.transform(X_val)
        X_test_scaled = tf_scaler.transform(X_test)

        tf_model = build_tensorflow_model(input_dim=X_train.shape[1])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
        )

        tf_model.fit(
            X_train_scaled,
            y_train.values,
            validation_data=(X_val_scaled, y_val.values),
            epochs=250,
            batch_size=8,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        tf_val_pred = tf_model.predict(X_val_scaled, verbose=0).ravel()
        tf_test_pred = tf_model.predict(X_test_scaled, verbose=0).ravel()

        candidates["tensorflow_mlp"] = {
            "framework": "tensorflow",
            "model": tf_model,
            "scaler": tf_scaler,
            "selected_features": selected_features,
            "val_metrics": regression_metrics(y_val, tf_val_pred),
            "test_metrics": regression_metrics(y_test, tf_test_pred),
        }

    except Exception as exc:
        print(f"TensorFlow model failed for Day {horizon}: {exc}")

    # Select best by validation MAE.
    comparison = pd.DataFrame(
    [
        {
            "horizon": horizon,
            "model": name,
            "framework": payload["framework"],
            "val_rmse": payload["val_metrics"]["rmse"],
            "val_mse": payload["val_metrics"]["mse"],
            "val_mae": payload["val_metrics"]["mae"],
            "val_r2": payload["val_metrics"]["r2"],
            "test_rmse": payload["test_metrics"]["rmse"],
            "test_mse": payload["test_metrics"]["mse"],
            "test_mae": payload["test_metrics"]["mae"],
            "test_r2": payload["test_metrics"]["r2"],
        }
        for name, payload in candidates.items()
    ]
).sort_values(["val_mae", "val_rmse"], ascending=[True, True])

# Important:
# Baselines are kept for comparison only.
# Final selected model must be an actual ML model.
    ml_comparison = comparison[comparison["framework"] != "baseline"].copy()

    if ml_comparison.empty:
        raise RuntimeError(f"No ML model trained successfully for Day {horizon}.")

    best_name = ml_comparison.iloc[0]["model"]
    best_payload = candidates[best_name]

    return {
        "horizon": horizon,
        "target_col": target_col,
        "best_name": best_name,
        "best_payload": best_payload,
        "comparison": comparison,
    }


def train_horizon_specific_models(df: pd.DataFrame) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    df = add_direct_forecast_targets(df)

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS + ["aqi_mean"]
    df = df.dropna(subset=required_columns).reset_index(drop=True)

    train_df, val_df, test_df = chronological_split_df(df)

    horizon_results = []

    for horizon, target_col in enumerate(TARGET_COLUMNS, start=1):
        result = train_single_horizon(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            target_col=target_col,
            horizon=horizon,
        )
        horizon_results.append(result)

    comparison = pd.concat(
        [result["comparison"] for result in horizon_results],
        ignore_index=True,
    )

    selected_rows = []

    for result in horizon_results:
        best_payload = result["best_payload"]
        selected_rows.append(
            {
                "horizon": result["horizon"],
                "selected_model": result["best_name"],
                "test_rmse": best_payload["test_metrics"]["rmse"],
                "test_mse": best_payload["test_metrics"]["mse"],
                "test_mae": best_payload["test_metrics"]["mae"],
                "test_r2": best_payload["test_metrics"]["r2"],
            }
        )

    selected_df = pd.DataFrame(selected_rows)

    overall_test_metrics = {
        "avg_rmse": float(selected_df["test_rmse"].mean()),
        "avg_mse": float(selected_df["test_mse"].mean()),
        "avg_mae": float(selected_df["test_mae"].mean()),
        "avg_r2": float(selected_df["test_r2"].mean()),
    }

    for _, row in selected_df.iterrows():
        horizon = int(row["horizon"])
        overall_test_metrics[f"day_{horizon}_rmse"] = float(row["test_rmse"])
        overall_test_metrics[f"day_{horizon}_mse"] = float(row["test_mse"])
        overall_test_metrics[f"day_{horizon}_mae"] = float(row["test_mae"])
        overall_test_metrics[f"day_{horizon}_r2"] = float(row["test_r2"])

    return {
        "horizon_results": horizon_results,
        "comparison": comparison,
        "selected_df": selected_df,
        "test_metrics": overall_test_metrics,
        "training_rows": len(df),
    }


def save_and_register_best(result: dict):
    out_dir = Path("model_artifacts") / "best_model"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_horizons = []

    for horizon_result in result["horizon_results"]:
        horizon = horizon_result["horizon"]
        best_name = horizon_result["best_name"]
        best_payload = horizon_result["best_payload"]

        horizon_metadata = {
            "horizon": horizon,
            "selected_model": best_name,
            "framework": best_payload["framework"],
            "selected_features": best_payload["selected_features"],
            "test_metrics": best_payload["test_metrics"],
            "val_metrics": best_payload["val_metrics"],
        }

        if best_payload["framework"] == "sklearn":
            model_path = f"horizon_{horizon}_model.joblib"
            joblib.dump(best_payload["model"], out_dir / model_path)
            horizon_metadata["model_path"] = model_path

        elif best_payload["framework"] == "tensorflow":
            model_path = f"horizon_{horizon}_keras_model.keras"
            scaler_path = f"horizon_{horizon}_scaler.joblib"

            best_payload["model"].save(out_dir / model_path)
            joblib.dump(best_payload["scaler"], out_dir / scaler_path)

            horizon_metadata["model_path"] = model_path
            horizon_metadata["scaler_path"] = scaler_path

        elif best_payload["framework"] == "baseline":
            horizon_metadata["model_path"] = None

        metadata_horizons.append(horizon_metadata)

    # Union of selected features across all horizons.
    selected_features_union = sorted(
        {
            feature
            for horizon_info in metadata_horizons
            for feature in horizon_info["selected_features"]
        }
    )

    metadata = {
        "best_model": "horizon_specific_ensemble",
        "forecast_method": "horizon_specific_direct_multi_horizon",
        "forecast_horizon_days": 3,
        "forecast_granularity": "daily",
        "city": "Islamabad",
        "training_rows": result["training_rows"],
        "selected_features": selected_features_union,
        "horizon_models": metadata_horizons,
        "validation_comparison": result["comparison"].to_dict(orient="records"),
        "selected_models_summary": result["selected_df"].to_dict(orient="records"),
        "test_metrics": result["test_metrics"],
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Hopsworks model registry still receives the folder.
    metrics = {
        "test_avg_rmse": float(result["test_metrics"]["avg_rmse"]),
        "test_avg_mse": float(result["test_metrics"]["avg_mse"]),
        "test_avg_mae": float(result["test_metrics"]["avg_mae"]),
        "test_avg_r2": float(result["test_metrics"]["avg_r2"]),
    }

    input_example = pd.DataFrame(
        [{feature: 0 for feature in selected_features_union}]
    )

    register_sklearn_model(out_dir, metrics=metrics, input_example=input_example)

    print("Registered horizon-specific direct model in Hopsworks Model Registry.")


def run_training_pipeline():
    df = read_daily_features()
    result = train_horizon_specific_models(df)

    print("\nFull horizon-specific validation/test comparison:")
    print(result["comparison"])

    print("\nSelected model per forecast horizon:")
    print(result["selected_df"])

    print("\nOverall test metrics:")
    print(result["test_metrics"])

    print("\nTraining rows:", result["training_rows"])

    save_and_register_best(result)


if __name__ == "__main__":
    run_training_pipeline()