import numpy as np
import pandas as pd
import pytest

from demand import intervals as iv
from demand.backtest import origins


def test_staggered_origins_cycle_through_the_trading_weekdays():
    at = origins(pd.Timestamp("2011-04-16"), n=6, staggered=True)
    assert [d.day_name() for d in at] == [
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    ]  # fmt: skip
    assert origins(pd.Timestamp("2011-04-16"), n=2) == [
        pd.Timestamp("2011-04-16"),
        pd.Timestamp("2011-04-23"),
    ]


def wide():
    days = pd.date_range("2011-01-01", periods=60, freq="D")
    return pd.DataFrame({"A": 10.0, "B": 0.0}, index=days)


def test_levels_use_only_days_before_the_origin_and_never_go_below_one():
    w = wide()
    w.loc["2011-02-10":, "A"] = 1000.0  # after the origin; must not count
    lv = iv.levels(w, [pd.Timestamp("2011-02-10")])
    assert lv.set_index("series")["level"].to_dict() == {"A": 10.0, "B": 1.0}
    with pytest.raises(ValueError, match="no history"):
        iv.levels(w, [pd.Timestamp("2010-01-01")])


def results(n=1000, seed=0, spread=0.25, origin="2011-02-10"):
    rng = np.random.default_rng(seed)
    forecast = np.full(n, 10.0)
    return pd.DataFrame(
        {
            "origin": pd.Timestamp(origin),
            "series": "A",
            "horizon": np.tile([1, 2], n // 2),
            "forecast": forecast,
            "actual": forecast + rng.normal(0, spread * 10, n),
        }
    )


LEVEL = pd.DataFrame({"origin": [pd.Timestamp("2011-02-10")], "series": ["A"], "level": [10.0]})


def test_intervals_cover_what_they_promise_on_new_data_from_the_same_process():
    q = iv.fit(results(seed=0), LEVEL, coverage=0.8)
    checked = iv.coverage(iv.apply(results(seed=1), LEVEL, q))
    assert checked["coverage"].item() == pytest.approx(0.8, abs=0.04)


def test_intervals_undercover_when_the_errors_grow():
    q = iv.fit(results(seed=0), LEVEL)
    checked = iv.coverage(iv.apply(results(seed=1, spread=0.75), LEVEL, q))
    assert checked["coverage"].item() < 0.5


def test_intervals_never_go_below_zero_and_contain_the_forecast():
    q = iv.fit(results(seed=0, spread=2.0), LEVEL)
    out = iv.apply(results(seed=1), LEVEL, q)
    assert (out["lower"] >= 0).all()
    assert ((out["lower"] <= out["forecast"]) & (out["forecast"] <= out["upper"])).all()


def test_bad_requests_are_refused():
    with pytest.raises(ValueError, match="coverage"):
        iv.fit(results(), LEVEL, coverage=1.0)
    q = iv.fit(results(), LEVEL)
    with pytest.raises(ValueError, match="horizons"):
        iv.apply(results().assign(horizon=3), LEVEL, q)
    with pytest.raises(ValueError, match="no level"):
        iv.apply(results(origin="2011-03-01"), LEVEL, q)


def test_coverage_by_group():
    out = iv.apply(results(seed=1), LEVEL, iv.fit(results(seed=0), LEVEL))
    by = iv.coverage(out, by="horizon")
    assert list(by.index) == [1, 2]
    assert by["rows"].sum() == len(out)
