"""Forecasts that need no fitting. Every real model has to beat these to count.

Each takes the history (rows = open days, columns = series) and the dates to forecast, and
returns a frame indexed by those dates. Only the history is ever read.
"""

import pandas as pd


def _check(history: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    if history.empty:
        raise ValueError("no history to forecast from")
    if len(dates) and dates.min() <= history.index.max():
        raise ValueError("forecast dates must come after the history")


def naive(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The last open day, repeated."""
    _check(history, dates)
    return pd.DataFrame([history.iloc[-1].to_numpy()] * len(dates), index=dates,
                        columns=history.columns, dtype=float)  # fmt: skip


def seasonal_naive(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The last open day with the same weekday. Thursday sells twice what Sunday does (Day 3)."""
    return weekday_mean(history, dates, weeks=1)


def weekday_mean(history: pd.DataFrame, dates: pd.DatetimeIndex, weeks: int = 4) -> pd.DataFrame:
    """Mean of the last `weeks` open days with the same weekday.

    Steadier than one week back when daily amounts swing as much as they do here. A weekday the
    history has never seen open (a rare Saturday) falls back to the naive forecast.
    """
    _check(history, dates)
    if weeks < 1:
        raise ValueError("weeks must be at least 1")
    weekday = history.index.day_of_week
    rows = []
    for d in dates:
        same = history[weekday == d.day_of_week]
        rows.append(same.iloc[-weeks:].mean() if len(same) else history.iloc[-1].astype(float))
    return pd.DataFrame(rows, index=dates, columns=history.columns, dtype=float)


BASELINES = {"naive": naive, "seasonal_naive": seasonal_naive, "weekday_mean_4": weekday_mean}
