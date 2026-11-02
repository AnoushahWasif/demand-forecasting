import numpy as np
import pandas as pd
import pytest

from demand.backtest import run_backtest
from demand.calendar import planned_open_days
from demand.models import statistical as st
from demand.models.baselines import weekday_mean

PATTERN = np.array([10, 30, 30, 30, 25, 15], dtype=float)  # six open days a week


def weekly(weeks=20, noise=1.0, seed=0, pattern=PATTERN):
    days = pd.bdate_range(
        "2011-01-02", periods=weeks * 6, freq="C", weekmask="Sun Mon Tue Wed Thu Fri"
    )
    rng = np.random.default_rng(seed)
    values = np.tile(pattern, weeks) + rng.normal(0, noise, weeks * 6)
    return pd.DataFrame({"A": np.clip(values, 0, None)}, index=days)


def after(history, n):
    return pd.bdate_range(history.index.max() + pd.Timedelta(days=1), periods=n, freq="C",
                          weekmask="Sun Mon Tue Wed Thu Fri")  # fmt: skip


@pytest.mark.parametrize("model", [st.ets, st.ets_log, st.sarima])
def test_a_clean_weekly_pattern_is_learned(model):
    h = weekly()
    fc = model(h, after(h, 12))
    assert fc.attrs["fallbacks"] == 0
    assert np.abs(fc["A"].to_numpy() - np.tile(PATTERN, 2)).mean() < 2.5


def test_forecasts_are_never_negative():
    h = weekly(pattern=np.array([0, 0, 40, 0, 0, 0], dtype=float), noise=3)
    assert (st.ets(h, after(h, 12)).to_numpy() >= 0).all()


def test_log_model_is_pulled_less_by_one_huge_order():
    h = weekly()
    spiked = h.copy()
    spiked.iloc[-3, 0] = 3_000
    dates = after(h, 6)
    shift = {
        m.__name__: (m(spiked, dates) - m(h, dates)).abs().mean().item()
        for m in (st.ets, st.ets_log)
    }
    assert shift["ets_log"] < shift["ets"]


@pytest.mark.parametrize(
    "history",
    [weekly(weeks=3), pd.DataFrame({"A": 7.0}, index=weekly().index)],
    ids=["too short", "constant"],
)
def test_short_or_constant_history_falls_back_to_the_weekday_mean(history):
    dates = after(history, 6)
    fc = st.ets(history, dates)
    assert fc.attrs["fallbacks"] == 1
    pd.testing.assert_frame_equal(fc, weekday_mean(history, dates), check_dtype=False)


def test_a_failed_fit_falls_back_and_is_counted(monkeypatch):
    def broken(values, steps):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(st, "_ets_one", broken)
    h = weekly().assign(B=weekly(seed=1)["A"])
    fc = st._per_series(h, after(h, 6), st._ets_one)
    assert fc.attrs["fallbacks"] == 2
    assert np.isfinite(fc.to_numpy()).all()


@pytest.mark.parametrize("bad", [np.inf, 1e9])
def test_an_absurd_forecast_falls_back(bad):
    h = weekly()
    fc = st._per_series(h, after(h, 6), lambda values, steps: np.full(steps, bad))
    assert fc.attrs["fallbacks"] == 1
    assert fc["A"].max() < 100


def test_bad_inputs_are_refused():
    h = weekly()
    with pytest.raises(ValueError, match="after the history"):
        st.ets(h, h.index[-3:])
    with pytest.raises(ValueError, match="no history"):
        st.ets(h.iloc[:0], after(h, 3))


@pytest.mark.parametrize("name", list(st.MODELS))
def test_statistical_models_never_see_the_future(name):
    days = planned_open_days(pd.Timestamp("2011-01-04"), pd.Timestamp("2011-07-31"))
    rng = np.random.default_rng(2)
    clean = pd.DataFrame({"A": rng.integers(5, 50, len(days)).astype(float)}, index=days)
    origin = pd.Timestamp("2011-06-04")
    poisoned = clean.copy()
    poisoned.loc[poisoned.index >= origin] *= 1000
    stop = pd.Timestamp("2011-07-15")
    a = run_backtest(clean, {name: st.MODELS[name]}, [origin], stop=stop)
    b = run_backtest(poisoned, {name: st.MODELS[name]}, [origin], stop=stop)
    assert np.array_equal(a["forecast"], b["forecast"])


@pytest.mark.parametrize("model", [st.ets, st.sarima])
def test_a_closed_weekday_does_not_shift_the_weekly_pattern(model):
    h = weekly(weeks=20, noise=0.5)
    closed = h.index[(h.index.day_of_week == 0)][-1]  # a bank-holiday Monday last week
    gappy = h.drop(closed)
    fc = model(gappy, after(h, 6))
    assert fc.attrs["fallbacks"] == 0
    assert np.abs(fc["A"].to_numpy() - PATTERN).mean() < 2.5


def test_forecast_dates_with_a_closure_in_between_are_read_off_the_grid():
    h = weekly(weeks=20, noise=0.5)
    dates = after(h, 8).delete(3)  # the shop will be shut on the 4th day
    fc = st.ets(h, dates)
    expected = np.tile(PATTERN, 2)[[0, 1, 2, 4, 5, 6, 7]]
    assert fc.index.equals(dates)
    assert np.abs(fc["A"].to_numpy() - expected).mean() < 2.5


def test_a_saturday_forecast_is_refused():
    h = weekly()
    saturday = h.index.max() + pd.Timedelta(days=(5 - h.index.max().day_of_week) % 7 or 7)
    with pytest.raises(ValueError, match="Sunday to Friday"):
        st.ets(h, pd.DatetimeIndex([saturday]))
