# ==========================================================
# Training Pipeline (Seq2Seq + Linear Compatible)
# ==========================================================

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.models.factory import create_model


# ==========================================================
# Time-series split (NO SHUFFLE)
# ==========================================================

def time_series_split(series, lag, test_ratio=0.2):

    split_index = int(len(series) * (1 - test_ratio))

    train_series = series[:split_index]
    test_series = series[split_index - lag:]

    return train_series, test_series


# ==========================================================
# Main training function
# ==========================================================

def run_training(
    df,
    target_col,
    model_type,
    lag,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    epochs=None,
    forecast_horizon=None
):

    series = df[target_col].values.astype(np.float32)

    train_series, test_series = time_series_split(
        series,
        lag=lag,
        test_ratio=0.2
    )

    # ===============================
    # Create model
    # ===============================
    model = create_model(
        model_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs
    )

    # ===============================
    # Train
    # ===============================
    if model_type == "lstm":
        learning_curve = model.fit(
            series=train_series,
            lag=lag,
            horizon=forecast_horizon
        )
    else:
        learning_curve = None
        model.fit(train_series, lag=lag)

    # ===============================
    # Safe evaluation
    # ===============================
    predictions = []
    targets = []

    max_i = len(test_series) - lag - forecast_horizon

    if max_i <= 0:
        raise ValueError(
            "Test set too small for selected lag and forecast_horizon"
        )

    for i in range(max_i):

        input_window = test_series[i:i+lag]
        true_future = test_series[i+lag:i+lag+forecast_horizon]

        pred_future = model.forecast(
            input_window,
            steps=forecast_horizon
        )

        predictions.append(pred_future)
        targets.append(true_future)

    predictions = np.array(predictions)
    targets = np.array(targets)

    # ===============================
    # Metrics
    # ===============================
    r2 = r2_score(
        targets.reshape(-1),
        predictions.reshape(-1)
    )

    mse = mean_squared_error(
        targets.reshape(-1),
        predictions.reshape(-1)
    )

    # ===============================
    # Artifact
    # ===============================
    artifact = {
        "model": model,
        "config": {
            "model_type": model_type,
            "lag": lag,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "forecast_horizon": forecast_horizon,
            "train_length": len(train_series)
        },
        "metrics": {
            "r2": float(r2),
            "mse": float(mse)
        },
        "test_true": targets.tolist(),
        "test_pred": predictions.tolist()
    }

    if learning_curve is not None:
        artifact["learning_curve"] = learning_curve

    return artifact


# ==========================================================
# Forecast helper
# ==========================================================

def forecast_future(artifact, series, steps):

    model = artifact["model"]
    lag = artifact["config"]["lag"]

    last_window = series[-(lag+1):]

    future = model.forecast(
        last_window,
        steps
    )

    return future