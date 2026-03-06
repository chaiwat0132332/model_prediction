# ==========================================================
# Model Factory (Production-grade)
# สร้าง model จากชื่อ model_type
# รองรับ:
# - linear
# - lstm
# ==========================================================

from src.models.linear_regression import LinearModel
from src.models.lstm import LSTMModel


def create_model(
    model_type,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    epochs=None,
    batch_size=None,
    lr=None,
    patience=None,
    device=None,
    shuffle=None,
    forecast_horizon=None,  # reserved for compatibility
):
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a non-empty string")

    model_type = model_type.strip().lower()

    # ======================================================
    # Linear Regression
    # ======================================================
    if model_type == "linear":
        return LinearModel()

    # ======================================================
    # LSTM
    # ======================================================
    if model_type == "lstm":
        hidden_size = 128 if hidden_size is None else int(hidden_size)
        num_layers = 2 if num_layers is None else int(num_layers)
        dropout = 0.2 if dropout is None else float(dropout)
        epochs = 100 if epochs is None else int(epochs)
        batch_size = 32 if batch_size is None else int(batch_size)
        lr = 0.001 if lr is None else float(lr)
        patience = 25 if patience is None else int(patience)
        shuffle = True if shuffle is None else bool(shuffle)

        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in range [0.0, 1.0)")
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if patience <= 0:
            raise ValueError("patience must be > 0")

        return LSTMModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
            shuffle=shuffle,
        )

    # ======================================================
    # Unknown model
    # ======================================================
    raise ValueError(f"Unknown model_type: {model_type}")
