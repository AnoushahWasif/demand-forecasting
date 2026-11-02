"""Turn raw invoice lines into completed product sales, counting what each step removes.

Demand here means gross units sold (ARCHITECTURE.md decision 1). A cancellation takes its
units off the same customer's most recent earlier sales of that product (decision 2), because
most cancellations are partial: only 36% cancel a sale of exactly the same quantity.
"""

from collections import defaultdict

import pandas as pd

from demand.data import COLUMNS, check_columns

# Real products are five digits, sometimes with a letter or two for a colour or size
# ("85123A"). Everything else is postage, fees, manual adjustments, samples or vouchers.
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"

# A line counts as a large order when it is this many times the product's median line
# quantity (and at least LARGE_MIN units). Flagged, not removed (decision 5).
LARGE_FACTOR, LARGE_MIN = 20, 100


def match_cancellations(sales: pd.DataFrame, cancels: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Quantity left on each sale after cancellations are taken off it.

    For each cancellation, oldest first, units come off the same customer's earlier sales of
    the same product, newest sale first, never below zero. Cancellations with no customer id
    or no earlier sale can't be matched; they are counted and otherwise ignored.
    """
    remaining = sales["Quantity"].astype("int64").copy()
    by_key: dict[tuple, list] = defaultdict(list)
    known = sales[sales["Customer ID"].notna()].sort_values("InvoiceDate", kind="stable")
    for idx, cust, code, when in zip(
        known.index, known["Customer ID"], known["StockCode"], known["InvoiceDate"], strict=True
    ):
        by_key[(cust, code)].append((when, idx))

    stats = {"no_customer": 0, "no_earlier_sale": 0, "units_removed": 0, "units_unmatched": 0}
    for cust, code, when, qty in cancels.sort_values("InvoiceDate", kind="stable")[
        ["Customer ID", "StockCode", "InvoiceDate", "Quantity"]
    ].itertuples(index=False, name=None):
        to_remove = -int(qty)
        if pd.isna(cust):
            stats["no_customer"] += 1
            stats["units_unmatched"] += to_remove
            continue
        earlier = [idx for sold, idx in by_key.get((cust, code), []) if sold <= when]
        if not earlier:
            stats["no_earlier_sale"] += 1
            stats["units_unmatched"] += to_remove
            continue
        for idx in reversed(earlier):
            take = min(to_remove, int(remaining.at[idx]))
            remaining.at[idx] -= take
            to_remove -= take
            stats["units_removed"] += take
            if to_remove == 0:
                break
        stats["units_unmatched"] += to_remove
    return remaining, stats


def flag_large_orders(df: pd.DataFrame) -> pd.Series:
    # The median spans all dates, holdout included. Fine for the with/without comparison;
    # it must never become a model feature, or it leaks the future (decision 4).
    median = df.groupby("StockCode")["Quantity"].transform("median")
    return (df["Quantity"] >= LARGE_FACTOR * median) & (df["Quantity"] >= LARGE_MIN)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    check_columns(raw, "transactions")
    if not pd.api.types.is_datetime64_any_dtype(raw["InvoiceDate"]):
        raise ValueError("InvoiceDate must be a datetime column")
    df = raw[COLUMNS].copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    report: dict = {"rows_in": len(df), "removed": {}}
    removed = report["removed"]

    # Same invoice, product, quantity, price and minute: almost certainly entered twice.
    dup = df.duplicated()
    removed["exact_repeats"] = {"rows": int(dup.sum()), "units": int(df.loc[dup, "Quantity"].sum())}
    df = df[~dup]

    is_cancel = df["Invoice"].str.startswith("C")
    is_sale = df["Invoice"].str.fullmatch(r"\d+")
    other = ~is_cancel & ~is_sale
    removed["other_invoice_types"] = int(other.sum())  # "A..." bad-debt adjustments
    cancels = df[is_cancel & (df["Quantity"] < 0)]
    removed["cancellation_lines"] = int(is_cancel.sum())
    df = df[is_sale]

    not_product = ~df["StockCode"].str.fullmatch(PRODUCT_CODE)
    removed["non_product_codes"] = {
        "rows": int(not_product.sum()),
        "by_code": df.loc[not_product, "StockCode"].value_counts().head(20).to_dict(),
    }
    df = df[~not_product]
    cancels = cancels[cancels["StockCode"].str.fullmatch(PRODUCT_CODE)]

    bad = (df["Quantity"] <= 0) | (df["Price"] <= 0)
    removed["write_offs_and_zero_prices"] = int(bad.sum())
    df = df[~bad]

    remaining, matched = match_cancellations(df, cancels)
    report["cancellations"] = {"lines": len(cancels), **matched}
    df = df.assign(Quantity=remaining)
    fully_cancelled = df["Quantity"] == 0
    removed["fully_cancelled_sales"] = int(fully_cancelled.sum())
    df = df[~fully_cancelled]

    df = df.assign(large_order=flag_large_orders(df))
    # Only needed to match cancellations; it never leaves this function (decision 3).
    df = df.drop(columns="Customer ID")

    report["rows_out"] = len(df)
    report["units_out"] = int(df["Quantity"].sum())
    report["products"] = int(df["StockCode"].nunique())
    report["large_order_lines"] = int(df["large_order"].sum())
    report["large_order_units"] = int(df.loc[df["large_order"], "Quantity"].sum())
    return df.reset_index(drop=True), report
