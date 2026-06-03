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
from src.feature_engineering import get_model_matrix
from src.hopsworks_utils import read_daily_features, register_sklearn_model


TARGET_COLUMNS = [
    "target_aqi_day_1",
    "target_aqi_day_2",
    "target_aqi_day_3",
]


def add_direct_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates direct multi-horizon targets.

    target_aqi_day_1 = AQI 1 day ahead
    target_aqi_day_2 = AQI 2 days ahead
    target_aqi_day_3 = AQI 3 days ahead

    This removes recursive forecasting and prevents error accumulation.
    """
    df = df.sort_values(["city", "date"]).copy()

    df["target_aqi_day_1"] = df.groupby("city")["aqi_mean"].shift(-1)
    df["target_aqi_day_2"] = df.groupby("city")["aqi_mean"].shift(-2)
    df["target_aqi_day_3"] = df.groupby("city")["aqi_mean"].shift(-3)

    return df


def chronological_split(X: pd.DataFrame, y: pd.DataFrame):
    n = len(X)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    return (
        X.iloc[:train_end],
        X.iloc[train_end:val_end],
        X.iloc[val_end:],
        y.iloc[:train_end],
        y.iloc[train_end:val_end],
        y.iloc[val_end:],
    )


def evaluate_multi_output(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """
    Evaluates Day 1, Day 2, and Day 3 separately, then also calculates average metrics.
    """
    metrics = {}

    for i, col in enumerate(TARGET_COLUMNS):
        actual = y_true[col].values
        pred = y_pred[:, i]

        horizon = i + 1

        metrics[f"day_{horizon}_rmse"] = float(np.sqrt(mean_squared_error(actual, pred)))
        metrics[f"day_{horizon}_mae"] = float(mean_absolute_error(actual, pred))
        metrics[f"day_{horizon}_r2"] = float(r2_score(actual, pred))

    metrics["avg_rmse"] = float(np.mean([metrics[f"day_{i}_rmse"] for i in [1, 2, 3]]))
    metrics["avg_mae"] = float(np.mean([metrics[f"day_{i}_mae"] for i in [1, 2, 3]]))
    metrics["avg_r2"] = float(np.mean([metrics[f"day_{i}_r2"] for i in [1, 2, 3]]))

    return metrics


def build_tensorflow_model(input_dim: int, output_dim: int = 3):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(
                16,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dense(output_dim),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=["mae"],
    )

    return model


def train_three_models(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    df = add_direct_forecast_targets(df)

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    df = df.dropna(subset=required_columns).reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    # Feature selection using Random Forest importance on multi-output targets.
    selector_base = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        max_depth=8,
    )
    selector_base.fit(X, y)

    selector = SelectFromModel(selector_base, threshold="median", prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()

    if len(selected_features) < 8:
        importances = pd.Series(selector_base.feature_importances_, index=X.columns)
        selected_features = importances.sort_values(ascending=False).head(12).index.tolist()

    X = X[selected_features]

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

    results = {}

    # -------------------------
    # Model 1: Ridge Regression
    # -------------------------
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
    ridge_pred = ridge.predict(X_val)

    results["ridge_regression"] = {
        "model": ridge,
        "metrics": evaluate_multi_output(y_val, ridge_pred),
        "framework": "sklearn",
    }

    # -------------------------
    # Model 2: Random Forest
    # -------------------------
    # REPLACE your RF block with this:
    rf = RandomForestRegressor(
    n_estimators=200,       # Fewer trees — faster, less overfit on small data
    random_state=42,
    min_samples_leaf=5,     # Was 2 — key fix, forces more generalization
    max_depth=5,            # Was 8 — shallower trees
    max_features="sqrt",
    max_samples=0.8,        # Bagging: use 80% of data per tree
    )

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)

    results["random_forest"] = {
        "model": rf,
        "metrics": evaluate_multi_output(y_val, rf_pred),
        "framework": "sklearn",
    }

    # -------------------------
    # Model 3: TensorFlow MLP
    # -------------------------
    import tensorflow as tf

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    tf_model = build_tensorflow_model(
        input_dim=X_train.shape[1],
        output_dim=len(TARGET_COLUMNS),
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
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
        epochs=300,
        batch_size=8,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

    tf_pred = tf_model.predict(X_val_scaled, verbose=0)

    results["tensorflow_mlp"] = {
        "model": tf_model,
        "scaler": scaler,
        "metrics": evaluate_multi_output(y_val, tf_pred),
        "framework": "tensorflow",
    }

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "avg_rmse": payload["metrics"]["avg_rmse"],
                "avg_mae": payload["metrics"]["avg_mae"],
                "avg_r2": payload["metrics"]["avg_r2"],
                "day_1_mae": payload["metrics"]["day_1_mae"],
                "day_2_mae": payload["metrics"]["day_2_mae"],
                "day_3_mae": payload["metrics"]["day_3_mae"],
            }
            for name, payload in results.items()
        ]
    ).sort_values(["avg_mae", "avg_rmse"], ascending=[True, True])

    best_name = comparison.iloc[0]["model"]
    best_payload = results[best_name]

    # Final test metrics for selected model.
    if best_name == "tensorflow_mlp":
        X_test_scaled = best_payload["scaler"].transform(X_test)
        test_pred = best_payload["model"].predict(X_test_scaled, verbose=0)
    else:
        test_pred = best_payload["model"].predict(X_test)

    test_metrics = evaluate_multi_output(y_test, test_pred)

    return {
        "best_name": best_name,
        "best_payload": best_payload,
        "selected_features": selected_features,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "input_example": X.head(1),
        "training_rows": len(df),
    }


def save_and_register_best(result: dict):
    out_dir = Path("model_artifacts") / "best_model"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    best_name = result["best_name"]
    payload = result["best_payload"]

    if best_name == "tensorflow_mlp":
        payload["model"].save(out_dir / "keras_model.keras")
        joblib.dump(payload["scaler"], out_dir / "scaler.joblib")
    else:
        joblib.dump(payload["model"], out_dir / "model.joblib")

    metadata = {
        "best_model": best_name,
        "selected_features": result["selected_features"],
        "target_columns": TARGET_COLUMNS,
        "validation_comparison": result["comparison"].to_dict(orient="records"),
        "test_metrics": result["test_metrics"],
        "forecast_method": "direct_multi_horizon",
        "forecast_horizon_days": 3,
        "forecast_granularity": "daily",
        "city": "Islamabad",
        "training_rows": result["training_rows"],
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "test_avg_rmse": float(result["test_metrics"]["avg_rmse"]),
        "test_avg_mae": float(result["test_metrics"]["avg_mae"]),
        "test_avg_r2": float(result["test_metrics"]["avg_r2"]),
        "validation_avg_rmse": float(result["comparison"].iloc[0]["avg_rmse"]),
        "validation_avg_mae": float(result["comparison"].iloc[0]["avg_mae"]),
        "validation_avg_r2": float(result["comparison"].iloc[0]["avg_r2"]),
    }

    register_sklearn_model(out_dir, metrics=metrics, input_example=result["input_example"])

    print("Registered direct multi-horizon model in Hopsworks Model Registry.")


def run_training_pipeline():
    df = read_daily_features()
    result = train_three_models(df)

    print("\nDirect 3-day model comparison on validation set:")
    print(result["comparison"])

    print("\nSelected best model:", result["best_name"])
    print("Test metrics:", result["test_metrics"])
    print("Selected features:", result["selected_features"])
    print("Training rows:", result["training_rows"])

    save_and_register_best(result)


if __name__ == "__main__":
    run_training_pipeline()