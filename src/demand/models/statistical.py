"""Exponential smoothing per series, and SARIMA for the shop total.

Both use a weekly season of 6 steps: the shop is open six days a week. A model counting rows
would be thrown off by every bank holiday: one missing Monday and every later row sits one step
off in the cycle. So the models are fitted on a strict Sunday-to-Friday grid, with planned
closures filled from the same weekday a week earlier, and forecasts are read off that grid at
the requested dates.

A fit that fails or returns nonsense falls back to the 4-week weekday mean, and the fallback is
counted, so a model can't look good by quietly being the baseline.
"""

import warnings

import numpy as np
import pandas as pd

from demand.models.baselines import weekday_mean

SEASON = 6
MIN_HISTORY = 4 * SEASON
WEEK = "Sun Mon Tue Wed Thu Fri"


def _grid(history: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, np.ndarray, int]:
    """History on a Sunday-to-Friday grid, where each date sits on it, and the steps to forecast."""
    grid = pd.bdate_range(history.index.min(), dates.max(), freq="C", weekmask=WEEK)
    past, future = grid[grid <= history.index.max()], grid[grid > history.index.max()]
    filled = history.reindex(past)
    weekday = filled.index.day_of_week
    filled = filled.groupby(weekday).ffill().groupby(weekday).bfill().fillna(0)
    if not dates.isin(future).all():
        raise ValueError("forecast dates must fall on Sunday to Friday")
    positions = future.get_indexer(dates)
    return filled, positions, int(positions.max()) + 1


def _fallback(history: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    return weekday_mean(history.to_frame(), dates)[history.name].to_numpy()


def _ets_one(values: np.ndarray, steps: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter; bad results are caught below
        fit = ExponentialSmoothing(
            values, trend=None, seasonal="add", seasonal_periods=SEASON,
            initialization_method="estimated",
        ).fit()  # fmt: skip
    return fit.forecast(steps)


def _per_series(history, dates, fit_one, transform=None, inverse=None) -> pd.DataFrame:
    """Fit each column on its own. `attrs["fallbacks"]` counts columns that used the baseline."""
    if history.empty:
        raise ValueError("no history to forecast from")
    if len(dates) and dates.min() <= history.index.max():
        raise ValueError("forecast dates must come after the history")
    out, fallbacks = {}, 0
    filled, positions, steps = _grid(history, dates)
    for column in history.columns:
        series = history[column].astype(float)
        values = filled[column].to_numpy(dtype=float)
        if len(values) < MIN_HISTORY or np.ptp(values) == 0:
            # Too short to see a season, or constant: nothing to smooth.
            out[column], fallbacks = _fallback(series, dates), fallbacks + 1
            continue
        try:
            fc = fit_one(transform(values) if transform else values, steps)
            fc = (inverse(fc) if inverse else fc)[positions]
        except (ValueError, np.linalg.LinAlgError):
            fc = None
        if fc is None or not np.isfinite(fc).all() or fc.max() > 10 * max(values.max(), 1):
            out[column], fallbacks = _fallback(series, dates), fallbacks + 1
        else:
            out[column] = np.clip(fc, 0, None)  # demand can't be negative
    forecast = pd.DataFrame(out, index=dates, columns=history.columns)
    forecast.attrs["fallbacks"] = fallbacks
    return forecast


def ets(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Level plus additive weekly season, on units. No trend: Day 3 found no growth."""
    return _per_series(history, dates, _ets_one)


def ets_log(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The same on log(1 + units). It follows the typical day, so one huge order moves it less.

    Back-transforming gives roughly the median day, not the mean. MAE rewards the median.
    """
    return _per_series(history, dates, _ets_one, np.log1p, np.expm1)


def _sarima_one(values: np.ndarray, steps: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(values, order=(1, 0, 1), seasonal_order=(0, 1, 1, SEASON), trend="c").fit(
            disp=False
        )
    return fit.forecast(steps)


def sarima(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """SARIMA(1,0,1)(0,1,1)6. Slow, so the backtest only runs it on the shop total."""
    return _per_series(history, dates, _sarima_one)


MODELS = {"ets": ets, "ets_log": ets_log}
