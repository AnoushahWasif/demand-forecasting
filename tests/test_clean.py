import pandas as pd
import pytest

from demand.clean import clean, flag_large_orders, match_cancellations


def lines(*rows):
    """(invoice, code, qty, "YYYY-MM-DD HH:MM", price, customer) tuples -> raw frame."""
    df = pd.DataFrame(
        rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]
    )
    return df.assign(
        InvoiceDate=pd.to_datetime(df["InvoiceDate"]),
        Description="THING",
        Country="United Kingdom",
        **{"Customer ID": df["Customer ID"].astype("Float64")},
    )


SALE = ("500001", "85123A", 10, "2010-03-01 10:00", 2.5, 1.0)


# --- cancellations


def test_cancellation_removes_the_sale_it_cancels():
    out, report = clean(lines(SALE, ("C500002", "85123A", -10, "2010-03-01 11:00", 2.5, 1.0)))
    assert out.empty
    assert report["removed"]["fully_cancelled_sales"] == 1
    assert report["cancellations"]["units_removed"] == 10


def test_partial_cancellation_leaves_the_rest():
    out, _ = clean(lines(SALE, ("C500002", "85123A", -4, "2010-03-02 09:00", 2.5, 1.0)))
    assert out["Quantity"].tolist() == [6]


def test_cancellation_takes_from_the_newest_earlier_sale_first():
    out, _ = clean(
        lines(
            ("500001", "22423", 5, "2010-03-01 10:00", 12.0, 7.0),
            ("500002", "22423", 5, "2010-03-05 10:00", 12.0, 7.0),
            ("C500003", "22423", -7, "2010-03-06 10:00", 12.0, 7.0),
        )
    )
    assert out.sort_values("InvoiceDate")["Quantity"].tolist() == [3]
    assert out["Invoice"].tolist() == ["500001"]


def test_cancellation_never_reaches_a_later_sale_or_another_customer():
    out, report = clean(
        lines(
            ("C500001", "22423", -5, "2010-03-01 10:00", 12.0, 7.0),
            ("500002", "22423", 5, "2010-03-02 10:00", 12.0, 7.0),  # after the cancellation
            ("500003", "22423", 5, "2010-02-01 10:00", 12.0, 8.0),  # another customer
        )
    )
    assert sorted(out["Quantity"].tolist()) == [5, 5]
    assert report["cancellations"]["no_earlier_sale"] == 1
    assert report["cancellations"]["units_unmatched"] == 5


def test_cancelling_more_than_was_bought_stops_at_zero():
    out, report = clean(lines(SALE, ("C500002", "85123A", -25, "2010-03-02 09:00", 2.5, 1.0)))
    assert out.empty
    assert report["cancellations"]["units_removed"] == 10
    assert report["cancellations"]["units_unmatched"] == 15


def test_cancellation_without_a_customer_is_counted_not_guessed():
    out, report = clean(lines(SALE, ("C500002", "85123A", -10, "2010-03-01 11:00", 2.5, None)))
    assert out["Quantity"].tolist() == [10]
    assert report["cancellations"]["no_customer"] == 1


def test_sale_without_a_customer_is_kept_but_never_matched():
    remaining, stats = match_cancellations(
        lines(("500001", "22423", 5, "2010-03-01 10:00", 1.0, None)),
        lines(("C500002", "22423", -5, "2010-03-02 10:00", 1.0, 3.0)),
    )
    assert remaining.tolist() == [5]
    assert stats["no_earlier_sale"] == 1


# --- other rules


@pytest.mark.parametrize(
    "code", ["POST", "DOT", "M", "m", "C2", "BANK CHARGES", "AMAZONFEE", "gift_0001_20",
             "DCGS0058", "TEST001", "123456", "1234", "85123ABC"]
)  # fmt: skip
def test_non_product_codes_are_dropped(code):
    out, report = clean(lines(SALE, ("500003", code, 1, "2010-03-01 10:00", 18.0, 1.0)))
    assert out["StockCode"].tolist() == ["85123A"]
    assert report["removed"]["non_product_codes"]["by_code"] == {code: 1}


def test_codes_with_spaces_around_them_still_count_as_products():
    out, _ = clean(lines(("500001", " 85123A ", 3, "2010-03-01 10:00", 2.5, 1.0)))
    assert out["StockCode"].tolist() == ["85123A"]


def test_write_offs_and_zero_prices_are_dropped():
    out, report = clean(
        lines(
            SALE,
            ("500003", "22423", -12, "2010-03-01 10:00", 0.0, None),  # stock write-off
            ("500004", "22424", 3, "2010-03-01 10:00", 0.0, 1.0),  # free sample
            ("A500005", "22425", 1, "2010-03-01 10:00", -1000.0, None),  # bad-debt adjustment
        )
    )
    assert out["Invoice"].tolist() == ["500001"]
    assert report["removed"]["write_offs_and_zero_prices"] == 2
    assert report["removed"]["other_invoice_types"] == 1


def test_lines_entered_twice_are_dropped_once():
    out, report = clean(lines(SALE, SALE, SALE))
    assert out["Quantity"].tolist() == [10]
    assert report["removed"]["exact_repeats"] == {"rows": 2, "units": 20}


def test_large_orders_are_flagged_not_removed():
    rows = [("50000" + str(i), "22423", 2, f"2010-03-0{i} 10:00", 1.0, 1.0) for i in range(1, 6)]
    rows.append(("500009", "22423", 500, "2010-03-09 10:00", 1.0, 2.0))
    out, report = clean(lines(*rows))
    assert len(out) == 6
    assert out.loc[out["Quantity"] == 500, "large_order"].item()
    assert report["large_order_units"] == 500


def test_large_order_needs_a_minimum_size_too():
    # 50 is 50x the median of 1, but under the 100-unit minimum.
    rows = [("50000" + str(i), "22423", 1, "2010-03-01", 1.0, 1.0) for i in range(9)]
    df = lines(*rows, ("500099", "22423", 50, "2010-03-01", 1.0, 1.0))
    assert not flag_large_orders(df).any()


# --- output shape and bad input


def test_customer_id_never_leaves_cleaning():
    out, _ = clean(lines(SALE))
    assert "Customer ID" not in out.columns


def test_counts_add_up():
    out, report = clean(
        lines(
            SALE,
            SALE,
            ("500003", "POST", 1, "2010-03-01 10:00", 18.0, 1.0),
            ("C500004", "85123A", -10, "2010-03-02 10:00", 2.5, 1.0),
        )
    )
    removed = report["removed"]
    dropped = (
        removed["exact_repeats"]["rows"]
        + removed["other_invoice_types"]
        + removed["cancellation_lines"]
        + removed["non_product_codes"]["rows"]
        + removed["write_offs_and_zero_prices"]
        + removed["fully_cancelled_sales"]
    )
    assert report["rows_in"] - dropped == report["rows_out"] == len(out) == 0


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        clean(lines(SALE).drop(columns="Price"))


def test_rejects_text_dates():
    with pytest.raises(ValueError, match="datetime"):
        clean(lines(SALE).assign(InvoiceDate="2010-03-01"))


def test_empty_input_gives_an_empty_result():
    out, report = clean(lines(SALE).iloc[:0])
    assert out.empty
    assert report["rows_out"] == 0


def test_input_frame_is_not_changed():
    raw = lines(SALE, ("C500002", "85123A", -4, "2010-03-02 09:00", 2.5, 1.0))
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)
