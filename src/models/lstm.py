import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class StableLSTM(nn.Module):
    def __init__(self, horizon, hidden_size=320, num_layers=2, dropout=0.25):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        fc_hidden = max(hidden_size // 2, 1)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


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
        device=None,
        shuffle=True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.shuffle = shuffle

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.learning_curve = {
            "train_loss": [],
            "val_loss": [],
        }

        self.model = StableLSTM(
            horizon=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.best_state = None
        self.is_fitted = False

    def create_sequences(self, series, lag, horizon):
        X = []
        y = []

        max_start = len(series) - lag - horizon + 1
        if max_start <= 0:
            return (
                np.empty((0, lag, 1), dtype=float),
                np.empty((0, horizon), dtype=float),
            )

        for i in range(max_start):
            X.append(series[i : i + lag])
            y.append(series[i + lag : i + lag + horizon])

        X = np.asarray(X, dtype=float).reshape(-1, lag, 1)
        y = np.asarray(y, dtype=float)
        return X, y

    def fit(self, series, lag, horizon, val_ratio=0.2, progress_callback=None):
        self.lag = lag
        self.horizon = horizon
        self.best_state = None
        self.learning_curve = {"train_loss": [], "val_loss": []}

        series = np.asarray(series, dtype=float)

        if series.ndim != 1:
            raise ValueError("series must be 1-dimensional")
        if lag <= 0:
            raise ValueError("lag must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if len(series) < lag + horizon + 2:
            raise ValueError(
                f"Not enough data: need at least {lag + horizon + 2} points, got {len(series)}"
            )

        delta = np.diff(series)
        X, y = self.create_sequences(delta, lag, horizon)

        if len(X) < 2:
            raise ValueError(
                "Not enough sequences after preprocessing; reduce lag/horizon or use more data"
            )

        split = int(len(X) * (1 - val_ratio))
        split = max(1, min(split, len(X) - 1))

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                "Empty train/validation split; adjust val_ratio or use more data"
            )

        self.scaler_X.fit(X_train.reshape(-1, 1))
        self.scaler_y.fit(y_train.reshape(-1, 1))

        def scale_x(arr):
            return self.scaler_X.transform(arr.reshape(-1, 1)).reshape(arr.shape)

        def scale_y(arr):
            return self.scaler_y.transform(arr.reshape(-1, 1)).reshape(arr.shape)

        X_train = scale_x(X_train)
        X_val = scale_x(X_val)
        y_train = scale_y(y_train)
        y_val = scale_y(y_val)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model = StableLSTM(
            horizon=horizon,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=15,
        )

        best_val = np.inf
        counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0

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

            if progress_callback is not None:
                progress_callback(
                epoch + 1,
                self.epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
                if counter >= self.patience:
                    break

        if self.best_state is None:
            raise RuntimeError("Training finished without a valid best_state")

        self.model.load_state_dict(self.best_state)
        self.is_fitted = True
        return self.learning_curve

    def forecast(self, last_window, steps):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        if steps <= 0:
            return np.array([], dtype=float)

        last_window = np.asarray(last_window, dtype=float)
        if last_window.ndim != 1:
            raise ValueError("last_window must be 1-dimensional")
        if len(last_window) < 2:
            raise ValueError("last_window must contain at least 2 points")

        delta = np.diff(last_window)

        if len(delta) < self.lag:
            pad = np.zeros(self.lag - len(delta), dtype=float)
            delta = np.concatenate([pad, delta])

        current_seq = delta[-self.lag :].copy()
        pred_deltas = []

        self.model.eval()
        with torch.no_grad():
            while len(pred_deltas) < steps:
                seq_scaled = self.scaler_X.transform(
                    current_seq.reshape(-1, 1)
                ).reshape(1, self.lag, 1)

                seq_tensor = torch.tensor(
                    seq_scaled,
                    dtype=torch.float32,
                    device=self.device,
                )

                pred_scaled = self.model(seq_tensor).cpu().numpy()[0]
                pred = self.scaler_y.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()

                steps_left = steps - len(pred_deltas)
                take = min(self.horizon, steps_left)

                pred_slice = pred[:take]
                pred_deltas.extend(pred_slice.tolist())

                current_seq = np.concatenate([current_seq[take:], pred_slice])

        value = float(last_window[-1])
        future = []

        for d in pred_deltas:
            value = value + d
            future.append(value)

        return np.asarray(future, dtype=float)
