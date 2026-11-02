import numpy as np
import pandas as pd
import pytest

from demand.eda import (
    before_holdout,
    demand_classes,
    per_open_day,
    seasonal_lift,
    weekday_profile,
)
from demand.series import TOTAL


def days(start, n, skip_saturdays=False):
    index = pd.date_range(start, periods=n, freq="D", name="date")
    return index[index.day_of_week != 5] if skip_saturdays else index


# --- holdout


def test_nothing_from_the_holdout_gets_through():
    wide = pd.DataFrame({"A": 1}, index=days("2011-10-10", 10))
    out = before_holdout(wide)
    assert out.index.max() == pd.Timestamp("2011-10-14")
    assert len(out) == 5


def test_holdout_split_needs_dates():
    with pytest.raises(ValueError, match="date"):
        before_holdout(pd.DataFrame({"A": [1, 2]}))


# --- per open day


def test_a_month_with_closed_days_is_not_read_as_a_dip():
    full = pd.Series(10, index=days("2010-11-01", 30))
    # December: same daily sales, but the shop shuts for the last 10 days.
    short = pd.Series(10, index=days("2010-12-01", 21))
    out = per_open_day(pd.concat([full, short]))
    assert out["per_day"].tolist() == [10, 10]
    assert out["open_days"].tolist() == [30, 21]


def test_a_period_with_no_open_days_has_no_rate():
    s = pd.Series([5, 5], index=pd.to_datetime(["2010-01-05", "2010-03-05"]))
    out = per_open_day(s)
    assert out.loc["2010-02-01", "open_days"] == 0
    assert np.isnan(out.loc["2010-02-01", "per_day"])


# --- weekdays


def test_weekday_ratios_and_a_closed_weekday():
    index = days("2010-03-01", 28, skip_saturdays=True)  # starts on a Monday
    s = pd.Series(np.where(index.day_of_week == 3, 30, 10), index=index)
    out = weekday_profile(s)
    assert out.loc[3, "mean"] == 30
    assert out.loc[3, "ratio"] == pytest.approx(30 / s.mean())
    assert out.loc[5, "open_days"] == 0
    assert np.isnan(out.loc[5, "mean"])
    assert out["open_days"].sum() == len(s)


# --- seasonal lift


def test_seasonal_lift_compares_autumn_with_the_rest_of_the_year():
    index = days("2010-01-01", 365)
    s = pd.Series(np.where(index.month.isin([9, 10, 11]), 15, 10), index=index)
    assert seasonal_lift(s).to_dict() == {2010: 1.5}


def test_seasonal_lift_compares_the_same_dates_each_year():
    # November is the busiest month, but the second year stops in mid-October.
    index = days("2010-01-01", 365).append(days("2011-01-01", 287))
    s = pd.Series(np.where(index.month == 11, 30, np.where(index.month >= 9, 15, 10)), index=index)
    assert seasonal_lift(s).to_dict()[2010] > 1.5
    assert seasonal_lift(s, until=(10, 14)).to_dict() == {2010: 1.5, 2011: 1.5}


def test_a_year_without_autumn_is_left_out():
    s = pd.Series(10, index=days("2011-01-01", 200))  # ends in July
    assert seasonal_lift(s).empty


# --- demand classes


def test_the_four_classes():
    n = 20
    every_day = np.ones(n)
    one_day_in_three = (np.arange(n) % 3 == 0).astype(float)
    sizes = np.where(np.arange(n) % 2, 1, 50)
    wide = pd.DataFrame(
        {
            "smooth": every_day * 5,
            "erratic": sizes,
            "intermittent": one_day_in_three * 5,
            "lumpy": one_day_in_three * sizes,
            TOTAL: 0,
        },
        index=days("2010-01-01", n),
    )
    out = demand_classes(wide)
    assert TOTAL not in out.index
    assert out["class"].to_dict() == {
        "smooth": "smooth",
        "erratic": "erratic",
        "intermittent": "intermittent",
        "lumpy": "lumpy",
    }
    assert out.loc["smooth", "adi"] == 1
    assert out.loc["intermittent", "adi"] == pytest.approx(20 / 7)


@pytest.mark.parametrize("sales", [[0, 0, 0, 0], [0, 7, 0, 0]])
def test_products_that_barely_sell_are_not_classed(sales):
    wide = pd.DataFrame({"A": sales}, index=days("2010-01-01", 4))
    assert demand_classes(wide).loc["A", "class"] == "too few sales"


def test_negative_demand_is_refused():
    wide = pd.DataFrame({"A": [1, -1, 2]}, index=days("2010-01-01", 3))
    with pytest.raises(ValueError, match="negative"):
        demand_classes(wide)
