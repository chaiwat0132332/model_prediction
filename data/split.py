# src/data/split.py

import numpy as np


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Chronological split for time series

    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
    """

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )