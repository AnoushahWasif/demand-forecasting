import numpy as np
import pandas as pd
import pytest

from demand.backtest import score_baselines, split_at
from demand.series import TOTAL


def open_days(start, n):
    days = pd.date_range(start, periods=n, freq="D", name="date")
    return days[days.day_of_week != 5]  # closed on Saturdays


def panel(start="2011-01-03", n=120):
    days = open_days(start, n)
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({"A": rng.integers(0, 20, len(days)), "B": 5}, index=days)
    return wide.assign(**{TOTAL: wide.sum(axis=1)})


def test_history_ends_before_the_origin_and_future_starts_at_it():
    wide = panel()
    origin = pd.Timestamp("2011-02-07")
    history, future = split_at(wide, origin, stop=None)
    assert history.index.max() < origin <= future.index.min()
    assert len(future) == 14
    assert history.index.append(future.index).is_unique


def test_closed_days_are_skipped_not_counted_as_horizons():
    _, future = split_at(panel(), pd.Timestamp("2011-02-07"), horizon=7, stop=None)
    assert (future.index.day_of_week != 5).all()
    assert future.index.max() == pd.Timestamp("2011-02-14")  # 7 open days, one Saturday skipped


def test_an_origin_on_a_closed_day_starts_at_the_next_open_day():
    _, future = split_at(panel(), pd.Timestamp("2011-02-12"), stop=None)  # a Saturday
    assert future.index.min() == pd.Timestamp("2011-02-13")


def test_the_future_window_never_reaches_the_holdout():
    wide = panel()
    stop = pd.Timestamp("2011-02-10")
    _, future = split_at(wide, pd.Timestamp("2011-02-07"), stop=stop)
    assert future.index.max() < stop
    assert len(future) == 3
    with pytest.raises(ValueError, match="holdout"):
        split_at(wide, stop, stop=stop)


@pytest.mark.parametrize("origin", ["2010-12-01", "2012-01-01"])
def test_no_history_or_no_future_is_refused(origin):
    with pytest.raises(ValueError, match="no history or no future"):
        split_at(panel(), pd.Timestamp(origin), stop=None)


def test_bad_input_is_refused():
    wide = panel()
    with pytest.raises(ValueError, match="horizon"):
        split_at(wide, pd.Timestamp("2011-02-07"), horizon=0, stop=None)
    with pytest.raises(ValueError, match="sorted"):
        split_at(wide.iloc[::-1], pd.Timestamp("2011-02-07"), stop=None)


def test_score_baselines_reports_every_baseline_and_skill():
    result = score_baselines(panel(), pd.Timestamp("2011-03-07"))
    assert set(result["models"]) == {"naive", "seasonal_naive", "weekday_mean_4"}
    assert result["models"]["seasonal_naive"]["products"]["mae_skill_vs_seasonal_naive"] == 0
    assert len(result["models"]["naive"]["products_wmape_by_horizon"]) == 14
    # B never changes, so every baseline gets B right; only A contributes error.
    assert result["models"]["naive"]["total"]["mae"] > 0
