"""Features for one model across all products, built only from data before each origin.

Every series is divided by its own recent level, so the model learns shapes ("Thursday runs
1.2 times the usual day") that carry over between a product selling 5 a day and one selling
500. Forecasts are multiplied back.
"""

import numpy as np
import pandas as pd

from demand.calendar import planned_closed

RECENT = 24  # four weeks of open days
WEEK = 6
WEEKMASK = "Sun Mon Tue Wed Thu Fri"


def origin_features(history: pd.DataFrame) -> pd.DataFrame:
    """One row per series: what is known at the end of `history`, scaled by its level."""
    if len(history) < RECENT:
        raise ValueError(f"need at least {RECENT} open days of history")
    recent = history.iloc[-RECENT:].astype(float)
    level = recent.mean().clip(lower=1.0)  # a product that barely sells still divides safely
    week = history.iloc[-WEEK:].mean()
    week_before = history.iloc[-2 * WEEK : -WEEK].mean()
    out = pd.DataFrame(
        {
            "level_log": np.log1p(level),
            "mean_6": week / level,
            "change_6": (week + 1) / (week_before + 1),
            "last": history.iloc[-1] / level,
            "std_24": recent.std() / level,
            "max_24": recent.max() / level,
            "zero_share_24": (recent == 0).mean(),
        }
    )
    weekday = recent.index.day_of_week
    for day in range(7):
        same = recent[weekday == day]
        out[f"weekday_{day}"] = same.mean() / level if len(same) else np.nan
    out["scale"] = level
    return out


def date_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar facts about the target dates, known in advance."""
    christmas = pd.to_datetime([f"{d.year}-12-25" for d in dates])
    previous = pd.DatetimeIndex(
        [pd.bdate_range(end=d - pd.Timedelta(days=1), periods=1, freq="C", weekmask=WEEKMASK)[0]
         for d in dates]
    )  # fmt: skip
    return pd.DataFrame(
        {
            "weekday": dates.day_of_week,
            "step": np.arange(1, len(dates) + 1),
            "weeks_to_christmas": np.clip((christmas - dates).days.to_numpy() / 7, 0, 52),
            "after_closure": planned_closed(previous).to_numpy().astype(int),
        },
        index=dates,
    )


def rows(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per series and target date, for forecasting `dates` from the end of `history`."""
    at_origin = origin_features(history)
    calendar = date_features(dates)
    out = []
    for date, cal in calendar.iterrows():
        part = at_origin.assign(**cal.to_dict(), date=date)
        part["same_weekday"] = part[f"weekday_{int(cal['weekday'])}"]
        out.append(part)
    frame = pd.concat(out).rename_axis("series").reset_index()
    return frame.drop(columns=[f"weekday_{d}" for d in range(7)])


def training_set(history: pd.DataFrame, horizon: int, every: int = WEEK) -> pd.DataFrame:
    """Past origins inside `history`, each with its next `horizon` open days as targets.

    A target is only used if it is inside `history`, so nothing at or after the real
    origin is ever learned from. Targets are scaled by the series level at their origin.
    """
    frames = []
    last = len(history) - horizon
    for end in range(RECENT, last + 1, every):
        past, future = history.iloc[:end], history.iloc[end : end + horizon]
        part = rows(past, future.index)
        actual = future.stack().rename_axis(["date", "series"]).rename("units").reset_index()
        part = part.merge(actual, on=["date", "series"], how="left", validate="one_to_one")
        part["target"] = part["units"] / part["scale"]
        frames.append(part.drop(columns="units"))
    if not frames:
        raise ValueError("history too short to build any training rows")
    return pd.concat(frames, ignore_index=True)


FEATURES = [
    "level_log", "mean_6", "change_6", "last", "std_24", "max_24", "zero_share_24",
    "same_weekday", "weekday", "step", "weeks_to_christmas", "after_closure",
]  # fmt: skip
