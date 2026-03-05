# ==========================================================
# Linear Regression Model (Baseline)
# ==========================================================

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearModel:

    def __init__(self):

        self.model = LinearRegression()

        self.lag = None
        self.is_fitted = False

    # ======================================================
    # Window creation
    # ======================================================

    def create_windows(self, series, lag):

        X = []
        y = []

        for i in range(len(series) - lag):

            X.append(series[i:i+lag])
            y.append(series[i+lag])

        return np.array(X), np.array(y)

    # ======================================================
    # Fit
    # ======================================================

    def fit(self, series, lag):

        self.lag = lag

        X, y = self.create_windows(series, lag)

        self.model.fit(X, y)
        self.is_fitted = True

        return None

    # ======================================================
    # Predict
    # ======================================================

    def predict(self, X):

        return self.model.predict(X)

    # ======================================================
    # Forecast
    # ======================================================

    def forecast(self, last_window, steps):

        window = last_window.copy()

        preds = []

        for _ in range(steps):

            pred = self.model.predict(
                window.reshape(1, -1)
            )[0]

            preds.append(pred)

            window = np.roll(window, -1)

            window[-1] = pred

        return np.array(preds)