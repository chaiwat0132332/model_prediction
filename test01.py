import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# 1 LOAD DATA
# ==============================

file_path = "/Users/chaiwatyingsunton/Documents/project_final/data/raw/LSTM set 1.xlsx"
column = "HFS1"

df = pd.read_excel(file_path)

series = df[column].values.astype(float)

print("Length:", len(series))
print("Mean:", np.mean(series))
print("Std:", np.std(series))

# ==============================
# 2 SMOOTH
# ==============================

smooth_window = 80

series_smooth = (
    pd.Series(series)
    .rolling(window=smooth_window, min_periods=1)
    .mean()
    .values
)

# ==============================
# 3 DELTA TRANSFORM
# ==============================

delta = np.diff(series_smooth)

# ==============================
# 4 BUILD SEQUENCES
# ==============================

lookback = 1000
horizon = 120

X = []
y = []

for i in range(len(delta) - lookback - horizon):

    X.append(delta[i:i+lookback])
    y.append(delta[i+lookback:i+lookback+horizon])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], lookback, 1)

# ==============================
# 5 TRAIN / VAL SPLIT
# ==============================

train_size = int(len(X)*0.8)

X_train = X[:train_size]
X_val = X[train_size:]

y_train = y[:train_size]
y_val = y[train_size:]

# ==============================
# 6 SCALING
# ==============================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X_train.reshape(-1,1))
scaler_y.fit(y_train.reshape(-1,1))

def scaleX(d):
    return scaler_X.transform(d.reshape(-1,1)).reshape(d.shape)

def scaleY(d):
    return scaler_y.transform(d.reshape(-1,1)).reshape(d.shape)

X_train = scaleX(X_train)
X_val = scaleX(X_val)

y_train = scaleY(y_train)
y_val = scaleY(y_val)

# ==============================
# 7 DATASET
# ==============================

train_ds = TensorDataset(
    torch.tensor(X_train,dtype=torch.float32),
    torch.tensor(y_train,dtype=torch.float32)
)

val_ds = TensorDataset(
    torch.tensor(X_val,dtype=torch.float32),
    torch.tensor(y_val,dtype=torch.float32)
)

train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 8 MODEL
# ==============================

class StableLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=320,
            num_layers=2,
            batch_first=True
        )

        self.norm = nn.LayerNorm(320)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(320,160)
        self.fc2 = nn.Linear(160,horizon)

        self.relu = nn.ReLU()

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.norm(out)

        out = self.dropout(out)

        out = self.relu(self.fc1(out))

        out = self.fc2(out)

        return out


model = StableLSTM().to(device)

# ==============================
# 9 TRAIN SETUP
# ==============================

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=15
)

epochs = 350

best_val = np.inf
patience = 25
counter = 0

# ==============================
# 10 TRAIN
# ==============================

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for xb,yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)

        loss = criterion(pred,yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():

        for xb,yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            val_loss += criterion(pred,yb).item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1} Train {train_loss:.6f} Val {val_loss:.6f}")

    if val_loss < best_val:

        best_val = val_loss
        counter = 0

        torch.save(model.state_dict(),"best_lstm_model01.pt")

    else:

        counter += 1

        if counter >= patience:
            print("Early stopping")
            break


# ==============================
# 11 FORECAST
# ==============================

model.load_state_dict(torch.load("best_lstm_model01.pt"))
model.eval()

future_steps = 3000

current_seq = delta[-lookback:].copy()

pred_deltas = []

with torch.no_grad():

    while len(pred_deltas) < future_steps:

        seq_scaled = scaler_X.transform(
            current_seq.reshape(-1,1)
        ).reshape(1,lookback,1)

        seq_tensor = torch.tensor(seq_scaled,dtype=torch.float32).to(device)

        pred_scaled = model(seq_tensor).cpu().numpy()[0]

        pred = scaler_y.inverse_transform(
            pred_scaled.reshape(-1,1)
        ).flatten()

        steps = min(horizon,future_steps-len(pred_deltas))

        pred_deltas.extend(pred[:steps])

        current_seq = np.concatenate([current_seq[horizon:],pred])

pred_deltas = np.array(pred_deltas)

# ==============================
# 12 RECONSTRUCT SERIES
# ==============================

last_value = series_smooth[-1]

future_series = []

value = last_value

for d in pred_deltas:

    value = value + d * 0.95
    future_series.append(value)

future_series = np.array(future_series)

# ==============================
# 13 PLOT
# ==============================

plt.figure(figsize=(12,6))

plt.plot(series_smooth,label="Smoothed Actual")

plt.plot(
    np.arange(len(series_smooth),
              len(series_smooth)+future_steps),
    future_series,
    label="Forecast"
)

plt.legend()
plt.grid()
plt.show()