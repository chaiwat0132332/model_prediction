# src/features/lag.py

import numpy as np


def create_lag_features(
    series: np.ndarray,
    lag: int
):
    """
    Convert series to supervised learning format

    Example:

    lag=3

    input:  [1 2 3 4 5]
    X:      [1 2 3]
            [2 3 4]

    y:      [4, 5]
    """

    X = []
    y = []

    for i in range(lag, len(series)):

        X.append(series[i-lag:i])
        y.append(series[i])

    return np.array(X), np.array(y)