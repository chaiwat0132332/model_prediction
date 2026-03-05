# ==========================================================
# Production LSTM (Delta Modeling + Direct Multi Output)
# Compatible with pipeline.py + Streamlit
# ==========================================================

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Network
# ==========================================================

class StableLSTM(nn.Module):

    def __init__(self, horizon):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=320,
            num_layers=2,
            batch_first=True
        )

        self.norm = nn.LayerNorm(320)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, horizon)

        self.relu = nn.ReLU()

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.norm(out)

        out = self.dropout(out)

        out = self.relu(self.fc1(out))

        out = self.fc2(out)

        return out


# ==========================================================
# Wrapper
# ==========================================================

class LSTMModel:

    def __init__(
        self,
        hidden_size=320,
        num_layers=2,
        dropout=0.25,
        lr=2e-4,
        batch_size=32,
        epochs=350,
        patience=25,
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.learning_curve = {
            "train_loss": [],
            "val_loss": []
        }

        self.model = StableLSTM(1).to(self.device)
        self.is_fitted = False


    # ======================================================
    # Create sequences
    # ======================================================

    def create_sequences(self, series, lag, horizon):

        X = []
        y = []

        for i in range(len(series) - lag - horizon):

            X.append(series[i:i+lag])
            y.append(series[i+lag:i+lag+horizon])

        X = np.array(X)
        y = np.array(y)

        X = X.reshape(X.shape[0], lag, 1)

        return X, y


    # ======================================================
    # Fit
    # ======================================================

    def fit(self, series, lag, horizon, val_ratio=0.2):
    
        self.lag = lag
        self.horizon = horizon

        series = np.array(series)

        # --------------------------------------
        # Delta transform
        # --------------------------------------

        delta = np.diff(series)

        X, y = self.create_sequences(delta, lag, horizon)

        split = int(len(X) * (1 - val_ratio))

        X_train = X[:split]
        X_val = X[split:]

        y_train = y[:split]
        y_val = y[split:]

        # --------------------------------------
        # Scaling
        # --------------------------------------

        self.scaler_X.fit(X_train.reshape(-1,1))
        self.scaler_y.fit(y_train.reshape(-1,1))

        def scaleX(d):
            return self.scaler_X.transform(d.reshape(-1,1)).reshape(d.shape)

        def scaleY(d):
            return self.scaler_y.transform(d.reshape(-1,1)).reshape(d.shape)

        X_train = scaleX(X_train)
        X_val = scaleX(X_val)

        y_train = scaleY(y_train)
        y_val = scaleY(y_val)

        # --------------------------------------
        # Dataset
        # --------------------------------------

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )

        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size
        )

        # --------------------------------------
        # Model
        # --------------------------------------

        self.model = StableLSTM(horizon).to(self.device)

        criterion = nn.SmoothL1Loss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=15
        )

        best_val = np.inf
        counter = 0

        # --------------------------------------
        # Training loop
        # --------------------------------------

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = 0

            for xb, yb in train_loader:

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()

                pred = self.model(xb)

                loss = criterion(pred, yb)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # -------------------------
            # Validation
            # -------------------------

            self.model.eval()

            val_loss = 0

            with torch.no_grad():

                for xb, yb in val_loader:

                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    pred = self.model(xb)

                    val_loss += criterion(pred, yb).item()

            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            self.learning_curve["train_loss"].append(train_loss)
            self.learning_curve["val_loss"].append(val_loss)

            # Early stopping

            if val_loss < best_val:

                best_val = val_loss
                counter = 0

                self.best_state = self.model.state_dict()

            else:

                counter += 1

                if counter >= self.patience:
                    break

        self.model.load_state_dict(self.best_state)

        self.is_fitted = True

        return self.learning_curve


    # ======================================================
    # Forecast
    # ======================================================

    def forecast(self, last_window, steps):

        last_window = np.array(last_window)

        delta = np.diff(last_window)

        if len(delta) < self.lag:
            pad = np.zeros(self.lag - len(delta))
            delta = np.concatenate([pad, delta])

        current_seq = delta[-self.lag:].copy()

        pred_deltas = []

        self.model.eval()

        with torch.no_grad():

            while len(pred_deltas) < steps:

                seq_scaled = self.scaler_X.transform(
                    current_seq.reshape(-1,1)
            ).reshape(1,self.lag,1)

                seq_tensor = torch.tensor(
                    seq_scaled,
                    dtype=torch.float32
            ).to(self.device)

                pred_scaled = self.model(
                    seq_tensor
            ).cpu().numpy()[0]

                pred = self.scaler_y.inverse_transform(
                    pred_scaled.reshape(-1,1)
            ).flatten()

            steps_left = steps - len(pred_deltas)

            take = min(self.horizon, steps_left)

            pred_deltas.extend(pred[:take])

            current_seq = np.concatenate([
                current_seq[take:],
                pred[:take]
            ])

        pred_deltas = np.array(pred_deltas)

        value = last_window[-1]

        future = []

        for d in pred_deltas:
            value = value + d * 0.95
            future.append(value)

        return np.array(future)