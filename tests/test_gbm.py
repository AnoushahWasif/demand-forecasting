import numpy as np
import pandas as pd
import pytest

from demand import features as ft
from demand.backtest import run_backtest
from demand.calendar import planned_open_days
from demand.models.gbm import lightgbm
from demand.series import TOTAL

PATTERN = {6: 0.5, 0: 1.0, 1: 1.0, 2: 1.0, 3: 1.4, 4: 0.8}  # Sun ... Fri, Thursday busiest


def panel(start="2010-09-01", end="2011-06-30", seed=0):
    days = planned_open_days(pd.Timestamp(start), pd.Timestamp(end))
    rng = np.random.default_rng(seed)
    shape = days.day_of_week.map(PATTERN).to_numpy()
    wide = pd.DataFrame(
        {
            "big": rng.poisson(200 * shape),
            "small": rng.poisson(5 * shape),
        },
        index=days,
    )
    return wide.assign(**{TOTAL: wide.sum(axis=1)})


# --- features


def test_origin_features_are_scaled_by_each_series_level():
    f = ft.origin_features(panel())
    assert f.loc["big", "scale"] > 30 * f.loc["small", "scale"]
    # Scaled, a big and a small seller with the same shape look alike.
    assert f.loc["big", "weekday_3"] == pytest.approx(f.loc["small", "weekday_3"], rel=0.35)
    assert f.loc["big", "weekday_3"] > f.loc["big", "weekday_6"]


def test_a_series_that_never_sold_divides_safely():
    wide = panel().assign(dead=0)
    f = ft.origin_features(wide)
    assert f.loc["dead", "scale"] == 1.0
    assert np.isfinite(f.drop(columns=[c for c in f if c.startswith("weekday_")])).all().all()


def test_origin_features_need_four_weeks():
    with pytest.raises(ValueError, match="at least"):
        ft.origin_features(panel().iloc[:10])


def test_date_features_know_closures_and_christmas():
    dates = pd.DatetimeIndex(["2011-04-21", "2011-04-26", "2011-12-23"])
    f = ft.date_features(dates)
    assert f["after_closure"].tolist() == [0, 1, 0]  # 26 Apr follows Easter Monday
    assert f["step"].tolist() == [1, 2, 3]
    assert f.loc["2011-12-23", "weeks_to_christmas"] == pytest.approx(2 / 7)
    assert f.loc["2011-04-21", "weekday"] == 3


# --- training rows never look past their own origin


def test_training_targets_stay_inside_the_history():
    history = panel()
    train = ft.training_set(history, horizon=14)
    assert train["date"].max() <= history.index.max()
    assert len(train) == train[["date", "series", "step"]].drop_duplicates().shape[0]


def test_training_features_only_use_data_before_their_origin():
    history = panel()
    train = ft.training_set(history, horizon=14)
    # The first training origin is after RECENT rows; its features must equal features
    # computed on exactly those rows, whatever comes after them.
    first = train[train["date"] == history.index[ft.RECENT]]
    poisoned = history.copy()
    poisoned.iloc[ft.RECENT :] *= 1000
    expected = ft.origin_features(poisoned.iloc[: ft.RECENT])
    for _, row in first.iterrows():
        assert row["mean_6"] == pytest.approx(expected.loc[row["series"], "mean_6"])
        assert row["scale"] == pytest.approx(expected.loc[row["series"], "scale"])


def test_targets_are_units_over_scale():
    history = panel()
    train = ft.training_set(history, horizon=3)
    row = train.iloc[0]
    assert row["target"] * row["scale"] == pytest.approx(history.loc[row["date"], row["series"]])


def test_too_short_for_any_training_row():
    with pytest.raises(ValueError, match="too short"):
        ft.training_set(panel().iloc[: ft.RECENT + 5], horizon=14)


# --- the model


def next_dates(history, n):
    return planned_open_days(
        history.index.max() + pd.Timedelta(days=1), pd.Timestamp("2011-12-20")
    )[:n]


def test_lightgbm_learns_the_weekly_shape_for_big_and_small_sellers():
    history = panel()
    dates = next_dates(history, 12)
    fc = lightgbm(history, dates)
    assert fc.index.equals(dates)
    assert list(fc.columns) == list(history.columns)
    assert (fc.to_numpy() >= 0).all()
    shape = dates.day_of_week.map(PATTERN).to_numpy()
    for column, level in (("big", 200), ("small", 5)):
        error = np.abs(fc[column].to_numpy() - level * shape).mean() / level
        assert error < 0.25, column


def test_lightgbm_is_repeatable():
    history = panel()
    dates = next_dates(history, 6)
    pd.testing.assert_frame_equal(lightgbm(history, dates), lightgbm(history, dates))


def test_lightgbm_refuses_bad_input():
    history = panel()
    with pytest.raises(ValueError, match="after the history"):
        lightgbm(history, history.index[-2:])
    with pytest.raises(ValueError, match="not enough history"):
        lightgbm(history.iloc[:30], next_dates(history.iloc[:30], 3))


def test_lightgbm_never_sees_the_future():
    clean = panel(end="2011-07-31")
    origin, stop = pd.Timestamp("2011-06-04"), pd.Timestamp("2011-07-15")
    poisoned = clean.copy()
    poisoned.loc[poisoned.index >= origin] *= 1000
    a = run_backtest(clean, {"lightgbm": lightgbm}, [origin], stop=stop)
    b = run_backtest(poisoned, {"lightgbm": lightgbm}, [origin], stop=stop)
    assert np.allclose(a["forecast"], b["forecast"])


def test_variants_refuse_unknown_or_no_features():
    from demand.models.gbm import make_lightgbm

    with pytest.raises(ValueError, match="unknown"):
        make_lightgbm(["level_log", "tomorrow_sales"])
    with pytest.raises(ValueError):
        make_lightgbm([])


def test_products_only_variant_still_forecasts_the_total():
    from demand.models.gbm import VARIANTS

    history = panel()
    fc = VARIANTS["lightgbm_products_only"](history, next_dates(history, 3))
    assert fc[TOTAL].notna().all() and (fc[TOTAL] > 0).all()
