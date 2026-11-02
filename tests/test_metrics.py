import numpy as np
import pandas as pd
import pytest

from demand.metrics import by_horizon, scores, skill

DAYS = pd.date_range("2011-03-07", periods=3, freq="D")


def frame(a, b):
    return pd.DataFrame({"A": a, "B": b}, index=DAYS)


def test_scores_by_hand():
    actual = frame([10, 0, 10], [0, 0, 0])
    forecast = frame([12, 2, 10], [0, 0, 4])
    s = scores(actual, forecast)
    assert s["mae"] == pytest.approx(8 / 6)
    assert s["rmse"] == pytest.approx(np.sqrt(24 / 6))
    assert s["wmape"] == pytest.approx(8 / 20)


def test_wmape_survives_zero_sales_days_but_not_a_window_with_no_sales():
    zero = frame([0, 0, 0], [0, 0, 0])
    assert scores(zero, frame([1, 0, 0], [0, 0, 0]))["wmape"] is None
    assert scores(frame([0, 5, 0], [0, 0, 0]), frame([1, 5, 0], [0, 0, 0]))["wmape"] == 0.2


def test_by_horizon_scores_each_step():
    actual = frame([10, 10, 10], [10, 10, 10])
    forecast = frame([10, 12, 16], [10, 8, 4])
    h = by_horizon(actual, forecast)
    assert h.index.tolist() == [1, 2, 3]
    assert h["mae"].tolist() == [0, 2, 6]


@pytest.mark.parametrize(
    "forecast",
    [
        frame([1, 2, np.nan], [0, 0, 0]),
        frame([1, 2, np.inf], [0, 0, 0]),
        frame([1, 2, 3], [0, 0, 0]).iloc[:2],
        frame([1, 2, 3], [0, 0, 0]).rename(columns={"B": "C"}),
    ],
)
def test_mismatched_or_broken_forecasts_are_refused(forecast):
    with pytest.raises(ValueError):
        scores(frame([1, 2, 3], [0, 0, 0]), forecast)


def test_skill():
    assert skill(8, 10) == pytest.approx(0.2)
    assert skill(12, 10) == pytest.approx(-0.2)
    assert skill(1, 0) is None


def test_from_arrays_refuses_nothing():
    from demand.metrics import from_arrays

    with pytest.raises(ValueError, match="nothing"):
        from_arrays(np.array([]), np.array([]))


def test_bias_shows_under_and_over_forecasting():
    actual = frame([10, 10, 0], [0, 0, 0])
    assert scores(actual, frame([5, 5, 0], [0, 0, 0]))["bias"] == pytest.approx(-0.5)
    assert scores(actual, frame([10, 10, 5], [0, 0, 0]))["bias"] == pytest.approx(0.25)
    assert scores(frame([0, 0, 0], [0, 0, 0]), frame([1, 0, 0], [0, 0, 0]))["bias"] is None
