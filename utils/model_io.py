# ==========================================================
# Model I/O Utilities (Production-grade)
# ==========================================================

import os
import pickle

from src.models.lstm import StableLSTM

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(artifact, model_name):

    save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    model = artifact["model"]

    artifact_to_save = {

        "config": artifact["config"],
        "metrics": artifact.get("metrics"),
        "test_true": artifact.get("test_true"),
        "test_pred": artifact.get("test_pred"),
        "learning_curve": artifact.get("learning_curve"),

    }

    # ==================================================
    # CASE 1: LSTM (PyTorch)
    # ==================================================

    if hasattr(model, "model") and hasattr(model.model, "state_dict"):

        artifact_to_save["model_type"] = "pytorch"

        artifact_to_save["model_state_dict"] = model.model.state_dict()

        # save scalers
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

    # ==================================================
    # Save file
    # ==================================================

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

        config = artifact["config"]

        model = create_model(
            config["model_type"],
            hidden_size=config.get("hidden_size"),
            num_layers=config.get("num_layers"),
            dropout=config.get("dropout"),
            epochs=config.get("epochs"),
            forecast_horizon=config.get("forecast_horizon", 100)
        )

        # restore lag + horizon
        model.lag = config["lag"]
        model.horizon = config.get("forecast_horizon", 100)

        # recreate neural network
        model.model = StableLSTM(config.get("forecast_horizon", 100))
        model.model.to(model.device)

        # load weights
        model.model.load_state_dict(
            artifact["model_state_dict"],
            strict=True
        )

        # restore scalers
        model.scaler_X = artifact.get("scaler_X")
        model.scaler_y = artifact.get("scaler_y")

        model.is_fitted = True
        model.model.eval()

        artifact["model"] = model

    # ==================================================
    # SKLEARN MODEL
    # ==================================================

    elif artifact.get("model_type") == "sklearn":

        artifact["model"] = artifact["model_object"]

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