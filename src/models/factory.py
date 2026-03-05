# ==========================================================
# Model Factory (Production-grade)
# สร้าง model จากชื่อ model_type
# รองรับ:
# - linear
# - lstm (Seq2Seq)
# ==========================================================

from src.models.linear_regression import LinearModel
from src.models.lstm import LSTMModel


# ==========================================================
# Create model
# ==========================================================

def create_model(
    model_type,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    epochs=None,
    batch_size=None,
    lr=None,
    forecast_horizon=None   # ยังรับไว้เพื่อ compatibility
):

    model_type = model_type.lower()

    # ======================================================
    # Linear Regression
    # ======================================================

    if model_type == "linear":
        return LinearModel()

    # ======================================================
    # LSTM (Seq2Seq Version)
    # ======================================================

    elif model_type == "lstm":

        # safe defaults
        hidden_size = hidden_size if hidden_size is not None else 128
        num_layers = num_layers if num_layers is not None else 2
        dropout = dropout if dropout is not None else 0.2
        epochs = epochs if epochs is not None else 100
        batch_size = batch_size if batch_size is not None else 32
        lr = lr if lr is not None else 0.001

        # ⚠️ forecast_horizon ไม่ใช้ใน constructor แล้ว
        # จะใช้ตอน fit() แทน

        return LSTMModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

    # ======================================================
    # Unknown model
    # ======================================================

    else:
        raise ValueError(f"Unknown model_type: {model_type}")