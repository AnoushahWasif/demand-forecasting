import numpy as np
import pandas as pd
import pytest

from demand.compare import combine, error_on_flagged_days, large_order_days, wmape_by_series

ORIGIN = pd.Timestamp("2011-05-07")
D1, D2 = pd.Timestamp("2011-05-08"), pd.Timestamp("2011-05-09")


def results(rows):
    """(model, date, series, forecast, actual) tuples -> backtest result rows."""
    df = pd.DataFrame(rows, columns=["model", "date", "series", "forecast", "actual"])
    df["origin"] = ORIGIN
    df["horizon"] = df["date"].map({D1: 1, D2: 2})
    return df[["date", "series", "forecast", "actual", "horizon", "origin", "model"]]


BASE = results(
    [
        ("a", D1, "X", 10, 12),
        ("a", D2, "X", 10, 8),
        ("b", D1, "X", 14, 12),
        ("b", D2, "X", 6, 8),
        ("a", D1, "Y", 0, 0),
        ("b", D1, "Y", 2, 0),
    ]
)


def test_combine_averages_forecasts_and_keeps_actuals():
    out = combine(BASE, ["a", "b"], "avg")
    x = out[out["series"] == "X"].sort_values("date")
    assert x["forecast"].tolist() == [12, 8]
    assert x["actual"].tolist() == [12, 8]
    assert set(out["model"]) == {"avg"}
    assert list(out.columns) == list(BASE.columns)


def test_combine_drops_rows_one_model_lacks():
    partial = BASE[~((BASE["model"] == "b") & (BASE["date"] == D2))]
    out = combine(partial, ["a", "b"], "avg")
    assert D2 not in set(out["date"])


@pytest.mark.parametrize(
    "models, name, match",
    [(["a"], "avg", "two"), (["a", "zzz"], "avg", "no results"), (["a", "b"], "a", "exists")],
)
def test_combine_refuses_bad_requests(models, name, match):
    with pytest.raises(ValueError, match=match):
        combine(BASE, models, name)


def test_wmape_by_series_and_a_series_that_never_sold():
    w = wmape_by_series(BASE)
    assert w.loc["X", "a"] == pytest.approx(4 / 20)
    assert w.loc["X", "b"] == pytest.approx(4 / 20)
    assert np.isnan(w.loc["Y", "a"])


def test_large_order_days_are_unique_dates_and_codes():
    sales = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(
                ["2011-05-08 09:00", "2011-05-08 15:00", "2011-05-09 10:00"]
            ),
            "StockCode": ["X", "X", "Y"],
            "large_order": [True, True, False],
        }
    )
    days = large_order_days(sales)
    assert days.to_dict("records") == [{"date": D1, "series": "X"}]


def test_error_on_flagged_days():
    days = pd.DataFrame({"date": [D1], "series": ["X"]})
    out = error_on_flagged_days(BASE, days)
    # Model a: errors 2 (D1, X, flagged), 2 (D2, X), 0 (D1, Y).
    assert out.loc["a", "flagged_share_of_error"] == pytest.approx(0.5)
    assert out.loc["a", "flagged_share_of_days"] == pytest.approx(1 / 3)
    assert out.loc["a", "flagged_share_of_units"] == pytest.approx(12 / 20)
