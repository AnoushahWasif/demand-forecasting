"""Forecast errors. wMAPE instead of MAPE, which divides by each day's sales and breaks on zeros."""

import numpy as np
import pandas as pd


def _errors(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    if not actual.index.equals(forecast.index) or not actual.columns.equals(forecast.columns):
        raise ValueError("actual and forecast must cover the same dates and series")
    if forecast.isna().any().any() or not np.isfinite(forecast.to_numpy(dtype=float)).all():
        raise ValueError("forecast has missing or infinite values")
    return forecast.astype(float) - actual.astype(float)


def scores(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """MAE, RMSE and wMAPE pooled over every date and series.

    wMAPE is None when nothing sold at all in the window: any error would be infinite.
    """
    _errors(actual, forecast)
    return from_arrays(actual.to_numpy(dtype=float), forecast.to_numpy(dtype=float))


def from_arrays(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """The same three scores from flat arrays of matching shape."""
    if actual.size == 0:
        raise ValueError("nothing to score")
    e = np.asarray(forecast, dtype=float) - np.asarray(actual, dtype=float)
    sold = float(np.abs(actual).sum())
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "wmape": float(np.abs(e).sum() / sold) if sold > 0 else None,
    }


def by_horizon(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """The same scores for each step ahead (1 = the first forecast day)."""
    _errors(actual, forecast)
    rows = {h + 1: scores(actual.iloc[[h]], forecast.iloc[[h]]) for h in range(len(actual))}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon")


def skill(score: float, reference: float) -> float | None:
    """1 - score / reference: above 0 beats the reference, 0.2 means 20% less error."""
    if reference == 0:
        return None
    return 1 - score / reference
