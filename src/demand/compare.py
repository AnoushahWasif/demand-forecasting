"""Comparing backtest results: averaging models, and where the error comes from."""

import numpy as np
import pandas as pd

KEY = ["origin", "date", "series"]


def combine(results: pd.DataFrame, models: list[str], name: str) -> pd.DataFrame:
    """The plain average of several models' forecasts, as one more model's rows.

    Only rows every model forecast are kept, so a model that skipped something can't
    quietly drop out of the average for it.
    """
    if len(models) < 2:
        raise ValueError("need at least two models to average")
    missing = set(models) - set(results["model"])
    if missing:
        raise ValueError(f"no results for {sorted(missing)}")
    if name in set(results["model"]):
        raise ValueError(f"{name} already exists")
    parts = results[results["model"].isin(models)]
    counts = parts.groupby(KEY)["model"].nunique()
    complete = counts[counts == len(models)].index
    parts = parts.set_index(KEY).loc[complete].reset_index()
    out = parts.groupby(KEY, as_index=False).agg(
        forecast=("forecast", "mean"), actual=("actual", "first"), horizon=("horizon", "first")
    )
    return out.assign(model=name)[results.columns]


def wmape_by_series(results: pd.DataFrame) -> pd.DataFrame:
    """wMAPE per model (columns) and series (rows). None-safe: a series that never sold is NaN."""
    abs_error = (results["forecast"] - results["actual"]).abs()
    grouped = results.assign(abs_error=abs_error).groupby(["series", "model"])
    sold = grouped["actual"].sum()
    return (grouped["abs_error"].sum() / sold.where(sold > 0)).unstack("model")


def large_order_days(sales: pd.DataFrame) -> pd.DataFrame:
    """(date, series) pairs that include at least one flagged large-order line."""
    flagged = sales[sales["large_order"]]
    return (
        flagged.assign(date=flagged["InvoiceDate"].dt.normalize(), series=flagged["StockCode"])[
            ["date", "series"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def error_on_flagged_days(results: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Per model: share of product-days that are flagged, and share of the error on them."""
    marked = results.merge(days.assign(flagged=True), on=["date", "series"], how="left")
    marked["flagged"] = marked["flagged"].astype("boolean").fillna(False).astype(bool)
    marked["abs_error"] = (marked["forecast"] - marked["actual"]).abs()
    rows = {}
    for model, m in marked.groupby("model"):
        total_error = m["abs_error"].sum()
        rows[model] = {
            "flagged_share_of_days": float(m["flagged"].mean()),
            "flagged_share_of_units": float(
                m.loc[m["flagged"], "actual"].sum() / m["actual"].sum()
            ),
            "flagged_share_of_error": float(m.loc[m["flagged"], "abs_error"].sum() / total_error)
            if total_error > 0
            else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")
