import pandas as pd
import pytest

from demand.series import TOTAL, closed_days, daily_units, descriptions, open_days, pick_products


def sales(*rows):
    """(code, qty, "YYYY-MM-DD", description) tuples."""
    df = pd.DataFrame(rows, columns=["StockCode", "Quantity", "InvoiceDate", "Description"])
    return df.assign(InvoiceDate=pd.to_datetime(df["InvoiceDate"]))


def test_open_days_come_from_any_invoice_including_cancellations():
    raw = sales(("A", 1, "2010-03-01", "x"), ("A", -1, "2010-03-03", "x"))
    assert list(open_days(raw).strftime("%d")) == ["01", "03"]


def test_open_days_ignore_the_time_of_day():
    raw = pd.DataFrame({"InvoiceDate": pd.to_datetime(["2010-03-01 08:00", "2010-03-01 17:30"])})
    assert len(open_days(raw)) == 1


def test_closed_days_are_the_gaps():
    days = pd.DatetimeIndex(pd.to_datetime(["2010-03-01", "2010-03-02", "2010-03-05"]))
    assert list(closed_days(days).strftime("%d")) == ["03", "04"]
    assert closed_days(days[:0]).empty


def test_closed_days_are_left_out_and_open_days_with_no_sales_are_zero():
    days = pd.DatetimeIndex(pd.to_datetime(["2010-03-01", "2010-03-02", "2010-03-04"]))
    wide = daily_units(
        sales(
            ("A", 3, "2010-03-01", "x"), ("A", 2, "2010-03-01", "x"), ("B", 1, "2010-03-04", "y")
        ),
        days,
    )
    assert list(wide.index.strftime("%d")) == ["01", "02", "04"]  # the 3rd was closed
    assert wide["A"].tolist() == [5, 0, 0]
    assert wide["B"].tolist() == [0, 0, 1]
    assert wide[TOTAL].tolist() == [5, 0, 1]
    assert (wide.dtypes == "int64").all()


def test_sales_on_a_day_not_marked_open_are_an_error():
    days = pd.DatetimeIndex(pd.to_datetime(["2010-03-01"]))
    with pytest.raises(ValueError, match="not marked open"):
        daily_units(sales(("A", 1, "2010-03-02", "x")), days)


def wide_from(counts: dict[str, list[int]], start="2010-03-01") -> pd.DataFrame:
    n = len(next(iter(counts.values())))
    return pd.DataFrame(counts, index=pd.date_range(start, periods=n, freq="D"))


def test_products_are_picked_by_days_sold_then_units_then_code():
    wide = wide_from(
        {
            "often": [1, 1, 1, 0],
            "bulk": [0, 0, 90, 0],
            "tie_b": [1, 1, 0, 0],
            "tie_a": [1, 1, 0, 0],
            "never": [0, 0, 0, 0],
        }
    )
    wide[TOTAL] = wide.sum(axis=1)
    picked = pick_products(wide, pd.Timestamp("2010-03-10"), n=4)
    assert picked == ["often", "tie_a", "tie_b", "bulk"]


def test_products_are_picked_only_from_data_before_the_cut():
    wide = wide_from({"early": [1, 1, 0, 0], "late": [0, 0, 1, 1]})
    cut = pd.Timestamp("2010-03-03")
    assert pick_products(wide, cut, n=1) == ["early"]
    # Changing everything after the cut doesn't change the choice (no look-ahead).
    wide.loc[wide.index >= cut, ["early", "late"]] = [0, 1000]
    assert pick_products(wide, cut, n=1) == ["early"]


def test_picking_never_returns_the_total_or_products_with_no_sales():
    wide = wide_from({"a": [1, 0], "b": [0, 0]})
    wide[TOTAL] = wide.sum(axis=1)
    with pytest.raises(ValueError, match="only 1 products"):
        pick_products(wide, pd.Timestamp("2010-03-10"), n=2)


def test_picking_with_no_history_is_an_error():
    wide = wide_from({"a": [1, 1]})
    with pytest.raises(ValueError, match="no data before"):
        pick_products(wide, pd.Timestamp("2010-01-01"), n=1)


def test_descriptions_use_the_most_common_spelling():
    s = sales(
        ("A", 1, "2010-03-01", "RED MUG "),
        ("A", 1, "2010-03-02", "RED MUG"),
        ("A", 1, "2010-03-03", "red mug??"),
        ("B", 1, "2010-03-03", None),
    )
    assert descriptions(s, ["A", "B"]) == {"A": "RED MUG", "B": ""}
