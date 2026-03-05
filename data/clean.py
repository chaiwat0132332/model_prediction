import pandas as pd
import numpy as np


# =========================
# Outlier removal (IQR)
# =========================
def remove_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Remove outliers using IQR method

    factor:
        1.5 = standard
        3.0 = conservative
    """

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    return series.clip(lower, upper)


# =========================
# Spike removal
# =========================
def remove_spikes(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    Remove sudden spikes using rolling std
    """

    rolling_mean = series.rolling(
        window=5,
        min_periods=1
    ).mean()

    rolling_std = series.rolling(
        window=5,
        min_periods=1
    ).std()

    upper = rolling_mean + threshold * rolling_std
    lower = rolling_mean - threshold * rolling_std

    return series.clip(lower, upper)


# =========================
# EMA smoothing
# =========================
def smooth_series_ema(
    series: pd.Series,
    span: int = 10
) -> pd.Series:
    
    return series.ewm(
        span=span,
        adjust=False
    ).mean()


# =========================
# Main cleaning function
# =========================
def clean_series(
    df: pd.DataFrame,
    target_col: str,
    smooth_span: int = 10
) -> pd.Series:
    """
    Production-grade time series cleaning pipeline

    Steps:
    - convert to numeric
    - remove NaN
    - reset index
    - ensure float
    - remove outliers (IQR)
    - remove spikes
    - smooth noise (EMA)

    Returns:
        clean float series
    """

    # Validate column
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found")

    # Convert to numeric
    series = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    # Remove NaN
    series = series.dropna()

    if len(series) == 0:
        raise ValueError("No valid numeric data found")

    # Reset index
    series = series.reset_index(drop=True)

    # Ensure float
    series = series.astype(float)

    # =========================
    # Remove outliers
    # =========================
    series = remove_outliers_iqr(series)

    # =========================
    # Remove spikes
    # =========================
    series = remove_spikes(series)

    # =========================
    # Smooth noise
    # =========================
    series = smooth_series_ema(
        series,
        span=smooth_span
    )

    return series


# =========================
# Validate length
# =========================
def validate_series_length(
    series: pd.Series,
    lag: int
):

    if len(series) <= lag:
        raise ValueError(
            f"Data length ({len(series)}) must be > lag ({lag})"
        )