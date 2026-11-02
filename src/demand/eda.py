"""Numbers behind the EDA notebook, on data before the holdout only.

The holdout (decision 6) is scored once at the end. Looking at it now would shape choices
that it is meant to test, so every function here is called on `before_holdout(...)` data.
"""

import numpy as np
import pandas as pd

from demand.series import HOLDOUT_START, TOTAL

# Syntetos-Boylan cut-offs: a product selling less often than every 1.32 periods on average
# is intermittent; a squared coefficient of variation of its non-zero sales above 0.49 is erratic.
ADI_CUTOFF, CV2_CUTOFF = 1.32, 0.49


def before_holdout(wide: pd.DataFrame, holdout_start: pd.Timestamp = HOLDOUT_START):
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise ValueError("expected a table indexed by date")
    return wide.loc[wide.index < holdout_start]


def per_open_day(series: pd.Series, freq: str = "MS") -> pd.DataFrame:
    """Units per open day for each period, so a month with a holiday closure isn't read as a dip."""
    grouped = series.resample(freq)
    out = pd.DataFrame({"units": grouped.sum(), "open_days": grouped.count()})
    out["per_day"] = out["units"] / out["open_days"].where(out["open_days"] > 0)
    return out


def weekday_profile(series: pd.Series) -> pd.DataFrame:
    """Mean units per weekday, as a ratio of the overall mean. Monday is 0."""
    by_day = series.groupby(series.index.day_of_week)
    out = pd.DataFrame({"mean": by_day.mean(), "open_days": by_day.size()})
    out["ratio"] = out["mean"] / series.mean()
    out.index.name = "weekday"
    return out.reindex(range(7)).assign(open_days=lambda d: d["open_days"].fillna(0).astype(int))


def seasonal_lift(series: pd.Series, until: tuple[int, int] = (11, 30)) -> pd.Series:
    """Per year: units per open day from 1 Sep to `until` (month, day) over Jan-Aug.

    Set `until` to the last date the latest year has, or the years aren't compared like for like.
    """
    lift = {}
    for year, part in series.groupby(series.index.year):
        base = part[part.index.month <= 8]
        end = pd.Timestamp(year=int(year), month=until[0], day=until[1])
        peak = part[(part.index.month >= 9) & (part.index <= end)]
        if len(base) and len(peak):
            lift[int(year)] = float(peak.mean() / base.mean())
    return pd.Series(lift, name="lift", dtype=float)


def demand_classes(wide: pd.DataFrame) -> pd.DataFrame:
    """How often each product sells (ADI) and how much the sizes vary (CV²), and its class.

    ADI is open days per selling day. CV² is over the days with a sale only, so a product that
    sells the same amount whenever it sells is smooth however often it sells.
    """
    products = wide.drop(columns=TOTAL, errors="ignore")
    if (products < 0).any().any():
        raise ValueError("demand can't be negative")
    selling_days = (products > 0).sum()
    sold = products.where(products > 0)
    out = pd.DataFrame(
        {
            "selling_days": selling_days,
            "adi": len(products) / selling_days.where(selling_days > 0),
            "cv2": (sold.std() / sold.mean()) ** 2,
        }
    )
    frequent = out["adi"] < ADI_CUTOFF
    steady = out["cv2"] < CV2_CUTOFF
    out["class"] = np.select(
        [out["selling_days"] < 2, frequent & steady, frequent, steady],
        ["too few sales", "smooth", "erratic", "intermittent"],
        "lumpy",
    )
    return out
