import numpy as np
import pandas as pd
import pytest

from demand.models.baselines import BASELINES, naive, seasonal_naive, weekday_mean


def history(values, start="2011-03-07"):  # a Monday
    days = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"A": values}, index=days)


def next_days(hist, n):
    return pd.date_range(hist.index.max() + pd.Timedelta(days=1), periods=n, freq="D")


def test_naive_repeats_the_last_day():
    h = history([1, 2, 3])
    assert naive(h, next_days(h, 4))["A"].tolist() == [3, 3, 3, 3]


def test_seasonal_naive_copies_the_same_weekday():
    h = history(list(range(14)))  # two weeks, Monday 0 ... Sunday 13
    fc = seasonal_naive(h, next_days(h, 9))
    assert fc["A"].tolist() == [7, 8, 9, 10, 11, 12, 13, 7, 8]


def test_weekday_mean_averages_the_last_weeks():
    h = history([10] * 7 + [20] * 7 + [30] * 7)
    fc = weekday_mean(h, next_days(h, 7), weeks=2)
    assert fc["A"].tolist() == [25] * 7


def test_a_weekday_never_seen_open_falls_back_to_the_last_day():
    h = history([1, 2, 3, 4, 5])  # Monday to Friday only
    fc = seasonal_naive(h, pd.DatetimeIndex(["2011-03-12"]))  # a Saturday
    assert fc["A"].tolist() == [5]


@pytest.mark.parametrize("name", list(BASELINES))
def test_forecasts_only_read_the_history(name):
    h = history(list(np.arange(28) % 9))
    dates = next_days(h, 14)
    fc = BASELINES[name](h, dates)
    assert fc.index.equals(dates)
    assert list(fc.columns) == ["A"]
    assert not fc.isna().any().any()


@pytest.mark.parametrize("name", list(BASELINES))
def test_forecasting_into_the_past_is_refused(name):
    h = history([1, 2, 3, 4])
    with pytest.raises(ValueError, match="after the history"):
        BASELINES[name](h, h.index[-2:])


@pytest.mark.parametrize("name", list(BASELINES))
def test_empty_history_is_refused(name):
    with pytest.raises(ValueError, match="no history"):
        BASELINES[name](history([]), pd.DatetimeIndex(["2011-03-08"]))


def test_weekday_mean_needs_at_least_one_week():
    h = history([1] * 7)
    with pytest.raises(ValueError, match="weeks"):
        weekday_mean(h, next_days(h, 1), weeks=0)
