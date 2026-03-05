# ==========================================================
# src/data/pipeline.py
# Simple Scaling Pipeline (Production Safe)
# ==========================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataPipeline:

    def __init__(self):

        self.scaler = MinMaxScaler()

        self.fitted = False


    # ======================================================
    # FIT + TRANSFORM
    # ======================================================

    def fit_transform(self, series):

        series = np.array(series).reshape(-1, 1)

        scaled = self.scaler.fit_transform(series)

        self.fitted = True

        return scaled.flatten()


    # ======================================================
    # TRANSFORM
    # ======================================================

    def transform(self, series):

        if not self.fitted:

            raise ValueError("Pipeline not fitted yet")

        series = np.array(series).reshape(-1, 1)

        scaled = self.scaler.transform(series)

        return scaled.flatten()


    # ======================================================
    # INVERSE TRANSFORM
    # ======================================================

    def inverse_transform(self, values):

        values = np.array(values).reshape(-1, 1)

        inv = self.scaler.inverse_transform(values)

        return inv.flatten()