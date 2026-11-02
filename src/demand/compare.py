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


def paired_difference(results: pd.DataFrame, a: str, b: str, rounds: int = 2_000, seed: int = 0):
    """wMAPE of model a minus model b, origin by origin, with a bootstrap interval over origins.

    Origins are resampled whole, because days inside one origin's window share a lot.
    """
    abs_error = (results["forecast"] - results["actual"]).abs()
    per = (
        results.assign(abs_error=abs_error)
        .groupby(["origin", "model"])[["abs_error", "actual"]]
        .sum()
        .unstack("model")
    )
    for m in (a, b):
        if m not in per["abs_error"]:
            raise ValueError(f"no results for {m}")
    err, sold = per["abs_error"], per["actual"][a]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per), size=(rounds, len(per)))
    diffs = (err[a].to_numpy()[picks].sum(1) - err[b].to_numpy()[picks].sum(1)) / sold.to_numpy()[
        picks
    ].sum(1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": float((err[a].sum() - err[b].sum()) / sold.sum()),
        "low": float(low),
        "high": float(high),
        "origins_a_better": int((err[a] < err[b]).sum()),
        "origins": len(per),
    }
