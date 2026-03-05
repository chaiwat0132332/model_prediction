# ==========================================================
# Production-grade Seq2Seq LSTM (Level Modeling + Validation)
# ==========================================================

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Network (Encoder–Decoder)
# ==========================================================

class Seq2SeqLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x, target_len):

        _, (h, c) = self.encoder(x)

        decoder_input = x[:, -1:, :]
        outputs = []

        for _ in range(target_len):

            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)

            outputs.append(pred)
            decoder_input = pred

        return torch.cat(outputs, dim=1)


# ==========================================================
# Wrapper
# ==========================================================

class LSTMModel:

    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        lr=0.001,
        batch_size=32,
        epochs=150,
        patience=25,
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = Seq2SeqLSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        self.scaler = MinMaxScaler()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.learning_curve = {
            "train_loss": [],
            "val_loss": []
        }

        self.is_fitted = False
        self.best_state = None


    # ======================================================
    # Window creation
    # ======================================================

    def create_windows(self, series, lag, horizon):

        X, y = [], []

        for i in range(len(series) - lag - horizon):
            X.append(series[i:i+lag])
            y.append(series[i+lag:i+lag+horizon])

        return np.array(X), np.array(y)


    # ======================================================
    # Fit (Level Modeling + Validation)
    # ======================================================

    def fit(self, series, lag, horizon, val_ratio=0.2):

        series = np.array(series)

        # Scale level directly (no differencing)
        scaled = self.scaler.fit_transform(
            series.reshape(-1, 1)
        ).flatten()

        X, y = self.create_windows(scaled, lag, horizon)

        split = int(len(X) * (1 - val_ratio))

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)

        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val = torch.FloatTensor(y_val).unsqueeze(-1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=False
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):

            # --------------------
            # Train
            # --------------------
            self.model.train()
            train_loss = 0

            for xb, yb in train_loader:

                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(xb, target_len=horizon)
                loss = self.criterion(pred, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # --------------------
            # Validation
            # --------------------
            self.model.eval()
            with torch.no_grad():

                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)

                val_pred = self.model(
                    X_val_device,
                    target_len=horizon
                )

                val_loss = self.criterion(
                    val_pred, y_val_device
                ).item()

            self.learning_curve["train_loss"].append(train_loss)
            self.learning_curve["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        self.is_fitted = True

        return self.learning_curve


    # ======================================================
    # Forecast (Level Modeling)
    # ======================================================

    def forecast(self, last_window, steps):

        last_window = np.array(last_window)

        scaled = self.scaler.transform(
            last_window.reshape(-1, 1)
        ).flatten()

        x = torch.FloatTensor(scaled)\
            .unsqueeze(0)\
            .unsqueeze(-1)\
            .to(self.device)

        self.model.eval()

        with torch.no_grad():

            pred_scaled = self.model(
                x,
                target_len=steps
            ).cpu().numpy()[0]

        pred = self.scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()

        return pred