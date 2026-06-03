from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.feature_engineering import get_model_matrix
from src.hopsworks_utils import read_daily_features, register_sklearn_model


def chronological_split(X: pd.DataFrame, y: pd.Series):
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return (
        X.iloc[:train_end],
        X.iloc[train_end:val_end],
        X.iloc[val_end:],
        y.iloc[:train_end],
        y.iloc[train_end:val_end],
        y.iloc[val_end:],
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_tensorflow_model(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_three_models(df: pd.DataFrame):
    df = df.sort_values("date").dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    X, y = get_model_matrix(df)

    # Feature selection using Random Forest importance.
    selector_base = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
    )
    selector_base.fit(X, y)

    selector = SelectFromModel(selector_base, threshold="median", prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()

    # Make sure selected list is not too tiny on small data.
    if len(selected_features) < 8:
        importances = pd.Series(selector_base.feature_importances_, index=X.columns)
        selected_features = importances.sort_values(ascending=False).head(12).index.tolist()

    X = X[selected_features]

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

    results = {}

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge.fit(X_train, y_train)
    results["ridge_regression"] = {
        "model": ridge,
        "metrics": evaluate(y_val, ridge.predict(X_val)),
        "framework": "sklearn",
    }

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "model": rf,
        "metrics": evaluate(y_val, rf.predict(X_val)),
        "framework": "sklearn",
    }

    # TensorFlow model wrapped as saved Keras model; selection still based on validation metrics.
    import tensorflow as tf

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    tf_model = build_tensorflow_model(input_dim=X_train.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    tf_model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=200,
        batch_size=8,
        callbacks=[early_stop],
        verbose=0,
    )
    tf_pred = tf_model.predict(X_val_scaled, verbose=0).ravel()
    results["tensorflow_mlp"] = {
        "model": tf_model,
        "scaler": scaler,
        "metrics": evaluate(y_val, tf_pred),
        "framework": "tensorflow",
    }

    comparison = pd.DataFrame(
        [
            {"model": name, **payload["metrics"]}
            for name, payload in results.items()
        ]
    ).sort_values(["rmse", "mae"], ascending=[True, True])

    best_name = comparison.iloc[0]["model"]
    best_payload = results[best_name]

    # Final test metrics for selected model.
    if best_name == "tensorflow_mlp":
        X_test_scaled = best_payload["scaler"].transform(X_test)
        test_pred = best_payload["model"].predict(X_test_scaled, verbose=0).ravel()
    else:
        test_pred = best_payload["model"].predict(X_test)

    test_metrics = evaluate(y_test, test_pred)

    return {
        "best_name": best_name,
        "best_payload": best_payload,
        "selected_features": selected_features,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "input_example": X.head(1),
    }


def save_and_register_best(result: dict):
    out_dir = Path("model_artifacts") / "best_model"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_name = result["best_name"]
    payload = result["best_payload"]

    if best_name == "tensorflow_mlp":
        # To keep registry loading simple for this class project, save a generic Python bundle.
        # The model registry still stores the best model artifact on cloud.
        payload["model"].save(out_dir / "keras_model")
        joblib.dump(payload["scaler"], out_dir / "scaler.joblib")
    else:
        joblib.dump(payload["model"], out_dir / "model.joblib")

    metadata = {
        "best_model": best_name,
        "selected_features": result["selected_features"],
        "validation_comparison": result["comparison"].to_dict(orient="records"),
        "test_metrics": result["test_metrics"],
        "forecast_granularity": "daily",
        "forecast_horizon_days": 3,
        "city": "Islamabad",
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        **{f"test_{k}": v for k, v in result["test_metrics"].items()},
        "validation_rmse": float(result["comparison"].iloc[0]["rmse"]),
        "validation_mae": float(result["comparison"].iloc[0]["mae"]),
        "validation_r2": float(result["comparison"].iloc[0]["r2"]),
    }

    register_sklearn_model(out_dir, metrics=metrics, input_example=result["input_example"])
    print("Registered best model in Hopsworks Model Registry.")


def run_training_pipeline():
    df = read_daily_features()
    result = train_three_models(df)

    print("\nModel comparison on validation set:")
    print(result["comparison"])

    print("\nSelected best model:", result["best_name"])
    print("Test metrics:", result["test_metrics"])
    print("Selected features:", result["selected_features"])

    save_and_register_best(result)


if __name__ == "__main__":
    run_training_pipeline()
