# ==========================================================
# Model I/O Utilities (Production-grade)
# ==========================================================

import os
import pickle

import torch

from src.models.lstm import StableLSTM

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(artifact, model_name):
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    model = artifact["model"]
    config = dict(artifact.get("config", {}))

    artifact_to_save = {
        "config": config,
        "metrics": artifact.get("metrics"),
        "test_true": artifact.get("test_true"),
        "test_pred": artifact.get("test_pred"),
        "learning_curve": artifact.get("learning_curve"),
    }

    # ==================================================
    # CASE 1: LSTM (PyTorch wrapper)
    # ==================================================
    if hasattr(model, "model") and hasattr(model.model, "state_dict"):
        artifact_to_save["model_type"] = "pytorch"
        artifact_to_save["model_state_dict"] = {
            k: v.detach().cpu()
            for k, v in model.model.state_dict().items()
        }

        # Persist model attributes for reliable reconstruction
        artifact_to_save["lstm_params"] = {
            "hidden_size": getattr(model, "hidden_size", config.get("hidden_size", 320)),
            "num_layers": getattr(model, "num_layers", config.get("num_layers", 2)),
            "dropout": getattr(model, "dropout", config.get("dropout", 0.25)),
            "lr": getattr(model, "lr", config.get("lr", 2e-4)),
            "batch_size": getattr(model, "batch_size", config.get("batch_size", 32)),
            "epochs": getattr(model, "epochs", config.get("epochs", 350)),
            "patience": getattr(model, "patience", config.get("patience", 25)),
            "lag": getattr(model, "lag", config.get("lag")),
            "horizon": getattr(model, "horizon", config.get("forecast_horizon", 1)),
        }

        if hasattr(model, "scaler_X"):
            artifact_to_save["scaler_X"] = model.scaler_X
        if hasattr(model, "scaler_y"):
            artifact_to_save["scaler_y"] = model.scaler_y

    # ==================================================
    # CASE 2: sklearn model
    # ==================================================
    else:
        artifact_to_save["model_type"] = "sklearn"
        artifact_to_save["model_object"] = model

    with open(save_path, "wb") as f:
        pickle.dump(artifact_to_save, f)


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_name):
    load_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    with open(load_path, "rb") as f:
        artifact = pickle.load(f)

    # ==================================================
    # LSTM MODEL
    # ==================================================
    if artifact.get("model_type") == "pytorch":
        from src.models.factory import create_model

        config = artifact.get("config", {})
        lstm_params = artifact.get("lstm_params", {})

        hidden_size = lstm_params.get("hidden_size", config.get("hidden_size", 320))
        num_layers = lstm_params.get("num_layers", config.get("num_layers", 2))
        dropout = lstm_params.get("dropout", config.get("dropout", 0.25))
        lr = lstm_params.get("lr", config.get("lr", 2e-4))
        batch_size = lstm_params.get("batch_size", config.get("batch_size", 32))
        epochs = lstm_params.get("epochs", config.get("epochs", 350))
        patience = lstm_params.get("patience", config.get("patience", 25))
        lag = lstm_params.get("lag", config.get("lag"))
        horizon = lstm_params.get("horizon", config.get("forecast_horizon", 1))

        model = create_model(
            config["model_type"],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            forecast_horizon=horizon,
        )

        model.patience = patience
        model.lag = lag
        model.horizon = horizon
        model.hidden_size = hidden_size
        model.num_layers = num_layers
        model.dropout = dropout

        model.model = StableLSTM(
            horizon=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(model.device)

        model.model.load_state_dict(artifact["model_state_dict"], strict=True)

        scaler_X = artifact.get("scaler_X")
        scaler_y = artifact.get("scaler_y")
        if scaler_X is None or scaler_y is None:
            raise ValueError("Saved LSTM artifact is missing scaler_X or scaler_y")

        model.scaler_X = scaler_X
        model.scaler_y = scaler_y
        model.is_fitted = True
        model.best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.model.state_dict().items()
        }
        model.model.eval()

        artifact["model"] = model

    # ==================================================
    # SKLEARN MODEL
    # ==================================================
    elif artifact.get("model_type") == "sklearn":
        artifact["model"] = artifact["model_object"]

    else:
        raise ValueError(f"Unknown saved model_type: {artifact.get('model_type')}")

    return artifact


# ==========================================================
# LIST MODELS
# ==========================================================

def list_models():
    files = os.listdir(MODEL_DIR)
    return [
        f.replace(".pkl", "")
        for f in files
        if f.endswith(".pkl")
    ]

def delete_model(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {model_name}")
    os.remove(path)
