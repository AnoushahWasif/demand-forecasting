"""Daily units per product on the days the shop traded, and the products to forecast.

A day with no invoices at all (not even a cancellation) is closed and left out of every
series. On an open day a product with no sales gets 0: the shop was open and nobody bought it.
"""

import pandas as pd

TOTAL = "TOTAL"
N_PRODUCTS = 100
HOLDOUT_START = pd.Timestamp("2011-10-15")
# 26 weekly backtest origins before the holdout. Products are picked on data before the first
# one, so the backtest can't favour products that were still selling months later.
FIRST_ORIGIN = HOLDOUT_START - pd.Timedelta(weeks=26)


def open_days(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Every date with at least one invoice line of any kind, taken from the uncleaned data."""
    return pd.DatetimeIndex(sorted(raw["InvoiceDate"].dt.normalize().unique()), name="date")


def closed_days(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if days.empty:
        return days
    calendar = pd.date_range(days.min(), days.max(), freq="D", name="date")
    return calendar.difference(days)


def daily_units(sales: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide table: one row per open day, one column per product, plus the all-products total."""
    dates = sales["InvoiceDate"].dt.normalize()
    stray = ~dates.isin(days)
    if stray.any():
        raise ValueError(f"{int(stray.sum())} sales fall on days not marked open")
    wide = (
        sales.assign(date=dates)
        .pivot_table(index="date", columns="StockCode", values="Quantity", aggfunc="sum")
        .reindex(days)
        .fillna(0)
        .astype("int64")
    )
    wide.columns.name = None
    wide[TOTAL] = wide.sum(axis=1)
    return wide


def pick_products(wide: pd.DataFrame, before: pd.Timestamp = FIRST_ORIGIN, n: int = N_PRODUCTS):
    """The n products that sold on the most open days before `before`.

    Ties go to more units, then to the stock code, so the choice is the same on every run.
    """
    history = wide.loc[wide.index < before].drop(columns=TOTAL, errors="ignore")
    if history.empty:
        raise ValueError(f"no data before {before.date()} to pick products from")
    ranking = pd.DataFrame({"days": (history > 0).sum(), "units": history.sum()})
    ranking = ranking[ranking["days"] > 0]
    ranking = ranking.sort_index().sort_values(["days", "units"], ascending=False, kind="stable")
    if len(ranking) < n:
        raise ValueError(f"only {len(ranking)} products sold before {before.date()}, need {n}")
    return list(ranking.index[:n])


def descriptions(sales: pd.DataFrame, codes: list[str]) -> dict[str, str]:
    """Most common description per product (the same code is sometimes typed differently)."""
    subset = sales[sales["StockCode"].isin(codes)].dropna(subset=["Description"])
    common = subset.groupby("StockCode")["Description"].agg(lambda s: s.str.strip().mode().iloc[0])
    return {code: common.get(code, "") for code in codes}
