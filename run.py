import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

# -----------------------
# 1. LOAD DATA
# -----------------------
file_path = "data.xlsx"     # แก้ path
column = "HFS1"

df = pd.read_excel(file_path)

series = df[column].values.astype(float)

# -----------------------
# 2. BASIC STATS
# -----------------------
print("Length:", len(series))
print("Mean:", np.mean(series))
print("Std:", np.std(series))
print("Min:", np.min(series))
print("Max:", np.max(series))

# signal to noise estimate
noise = np.std(np.diff(series))
signal = np.std(series)

print("Signal/Noise ratio:", signal / noise)

# -----------------------
# 3. SMOOTH FOR TREND VIEW
# -----------------------
smooth = pd.Series(series).rolling(50, min_periods=1).mean()

plt.figure(figsize=(10,5))
plt.plot(series, alpha=0.4, label="Raw")
plt.plot(smooth, linewidth=2, label="Smoothed")
plt.title("Trend Visualization")
plt.legend()
plt.grid()
plt.show()

# -----------------------
# 4. AUTOCORRELATION
# -----------------------
lag = 300

acf_vals = acf(series, nlags=lag)
pacf_vals = pacf(series, nlags=lag)

plt.figure(figsize=(10,4))
plt.plot(acf_vals)
plt.title("ACF")
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(pacf_vals)
plt.title("PACF")
plt.grid()
plt.show()

# -----------------------
# 5. SUGGEST LOOKBACK
# -----------------------
threshold = 0.2
lookback = np.where(np.abs(acf_vals) < threshold)[0][0]

print("Suggested Lookback:", lookback)