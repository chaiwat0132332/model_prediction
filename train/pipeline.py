import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src.models.factory import create_model


def time_series_split(series, test_ratio=0.2, lag=1):
    split_index = int(len(series) * (1 - test_ratio))

    train_series = series[:split_index]
    test_series = series[split_index - lag:]

    return train_series, test_series


def run_training(
    df,
    target_col,
    model_type,
    lag,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    epochs=None,
    batch_size=None,
    lr=None,
    forecast_horizon=120,
    progress_callback=None,
):
    series = df[target_col].astype(float).values

    if len(series) <= lag:
        raise ValueError("ข้อมูลมีน้อยกว่าค่า lag")
    if forecast_horizon is None or forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be > 0")

    train_series, test_series = time_series_split(series, test_ratio=0.2, lag=lag)

    model = create_model(
        model_type=model_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        forecast_horizon=forecast_horizon,
    )

    if model_type == "lstm":
        learning_curve = model.fit(
            train_series,
            lag=lag,
            horizon=forecast_horizon,
            progress_callback=progress_callback,
        )
    else:
        learning_curve = None
        model.fit(train_series, lag=lag)

    max_i = len(test_series) - lag - forecast_horizon + 1
    if max_i <= 0:
        raise ValueError("ข้อมูล test ไม่พอสำหรับประเมินผล กรุณาลด lag หรือ forecast_horizon")

    preds = []
    trues = []

    for i in range(max_i):
        x = test_series[i : i + lag]
        y_true = test_series[i + lag : i + lag + forecast_horizon]

        y_pred = model.forecast(x, steps=forecast_horizon)

        preds.append(np.asarray(y_pred).flatten())
        trues.append(np.asarray(y_true).flatten())

    test_pred = np.array(preds)
    test_true = np.array(trues)

    y_true_flat = test_true.flatten()
    y_pred_flat = test_pred.flatten()

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    artifact = {
        "model": model,
        "config": {
            "model_type": model_type,
            "lag": lag,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "forecast_horizon": forecast_horizon,
        },
        "metrics": {
            "mse": mse,
            "r2": r2,
        },
        "test_true": test_true,
        "test_pred": test_pred,
        "learning_curve": learning_curve,
    }

    return artifact


def forecast_future(model, series, lag, steps):
    series = np.asarray(series, dtype=float)

    if len(series) < lag:
        raise ValueError("ข้อมูลมีน้อยกว่าค่า lag")

    last_window = series[-lag:]
    return model.forecast(last_window, steps=steps)
