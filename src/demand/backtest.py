"""Forecast origins, the split at each one, and the rolling backtest over all of them."""

from collections.abc import Callable

import pandas as pd

from demand.series import HOLDOUT_START

HORIZON = 14  # open days ahead


def split_at(
    wide: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int = HORIZON,
    stop: pd.Timestamp | None = HOLDOUT_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History strictly before `origin`, and the next `horizon` open days from it.

    The future window is cut at `stop`, so no backtest window ever reaches into the holdout:
    the last origins get fewer horizons instead.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not wide.index.is_monotonic_increasing or not wide.index.is_unique:
        raise ValueError("dates must be sorted and unique")
    history = wide.loc[wide.index < origin]
    future = wide.loc[wide.index >= origin]
    if stop is not None:
        if origin >= stop:
            raise ValueError(f"origin {origin.date()} is inside the holdout")
        future = future.loc[future.index < stop]
    future = future.iloc[:horizon]
    if history.empty or future.empty:
        raise ValueError(f"no history or no future around {origin.date()}")
    return history, future


def score_baselines(wide: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """Every baseline at one origin, for the 100 products pooled and for the daily total."""
    from demand.metrics import by_horizon, scores, skill
    from demand.models.baselines import BASELINES
    from demand.series import TOTAL

    history, future = split_at(wide, origin)
    products = [c for c in wide.columns if c != TOTAL]
    out = {
        "origin": str(origin.date()),
        "history_days": len(history),
        "forecast_days": [str(d.date()) for d in future.index],
        "models": {},
    }
    for name, forecast in BASELINES.items():
        fc = forecast(history, future.index)
        out["models"][name] = {
            "products": scores(future[products], fc[products]),
            "total": scores(future[[TOTAL]], fc[[TOTAL]]),
            "products_wmape_by_horizon": [
                round(v, 3) for v in by_horizon(future[products], fc[products])["wmape"]
            ],
        }
    reference = out["models"]["seasonal_naive"]["products"]["mae"]
    for result in out["models"].values():
        result["products"]["mae_skill_vs_seasonal_naive"] = skill(
            result["products"]["mae"], reference
        )
    return out


# --- the rolling backtest

N_ORIGINS = 26


def origins(first: pd.Timestamp | None = None, n: int = N_ORIGINS) -> list[pd.Timestamp]:
    from demand.series import FIRST_ORIGIN

    first = FIRST_ORIGIN if first is None else first
    return [first + pd.Timedelta(weeks=k) for k in range(n)]


def forecast_dates(
    origin: pd.Timestamp, horizon: int = HORIZON, stop: pd.Timestamp | None = HOLDOUT_START
) -> pd.DatetimeIndex:
    """The next `horizon` planned open days, from the calendar, not from the data."""
    from demand.calendar import planned_open_days

    dates = planned_open_days(origin, origin + pd.Timedelta(days=3 * horizon + 14))
    if stop is not None:
        dates = dates[dates < stop]
    return dates[:horizon]


Forecaster = Callable[[pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]


def run_backtest(
    wide: pd.DataFrame,
    forecasters: dict[str, Forecaster],
    at: list[pd.Timestamp],
    horizon: int = HORIZON,
    stop: pd.Timestamp | None = HOLDOUT_START,
) -> pd.DataFrame:
    """Every forecaster at every origin. One row per origin, model, date and series."""
    rows = []
    for origin in at:
        history, future = split_at(wide, origin, horizon, stop)
        dates = forecast_dates(origin, horizon, stop)
        if not dates.equals(future.index):
            raise ValueError(f"calendar and data disagree on open days after {origin.date()}")
        step = {d: i + 1 for i, d in enumerate(dates)}
        for name, forecast in forecasters.items():
            # A copy, so a forecaster that changes its input can't affect the next one.
            fc = forecast(history.copy(), dates)
            if not fc.index.equals(dates) or not fc.columns.equals(history.columns):
                raise ValueError(f"{name} returned the wrong dates or series at {origin.date()}")
            long = (
                fc.rename_axis(index="date", columns="series")
                .stack()
                .rename("forecast")
                .reset_index()
                .assign(
                    actual=future.stack().to_numpy(),
                    origin=origin,
                    model=name,
                )
            )
            long["horizon"] = long["date"].map(step)
            rows.append(long)
    return pd.concat(rows, ignore_index=True)


def summarise(results: pd.DataFrame) -> dict:
    """Pooled, per horizon (with how many origins reach it), per origin, and per product."""
    from demand.metrics import from_arrays, skill
    from demand.series import TOTAL

    def pooled(d):
        return from_arrays(d["actual"].to_numpy(), d["forecast"].to_numpy())

    out = {}
    for model, m in results.groupby("model", sort=False):
        products, total = m[m["series"] != TOTAL], m[m["series"] == TOTAL]
        per_product = products.groupby("series").apply(lambda d: pooled(d)["wmape"])
        per_origin = products.groupby("origin").apply(lambda d: pooled(d)["wmape"])
        out[model] = {
            "products": {
                **pooled(products),
                "median_product_wmape": float(per_product.median()),
            },
            "total": pooled(total),
            "by_horizon": [
                {
                    "horizon": int(h),
                    "origins": int(d["origin"].nunique()),
                    "products_wmape": pooled(d[d["series"] != TOTAL])["wmape"],
                    "total_wmape": pooled(d[d["series"] == TOTAL])["wmape"],
                }
                for h, d in m.groupby("horizon")
            ],  # fmt: skip
            "by_origin": {str(o.date()): w for o, w in per_origin.items()},
        }
    if "seasonal_naive" in out:
        reference = out["seasonal_naive"]
        for result in out.values():
            for level in ("products", "total"):
                result[level]["mae_skill_vs_seasonal_naive"] = skill(
                    result[level]["mae"], reference[level]["mae"]
                )
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path

    from demand import data
    from demand.models.baselines import BASELINES

    wide = pd.read_parquet(data.data_dir() / "processed" / "daily_units.parquet")
    results = run_backtest(wide, BASELINES, origins())
    summary = summarise(results)
    Path("reports/backtest.json").write_text(json.dumps(summary, indent=2))
    from demand.plots import plot_backtest

    plot_backtest(summary, origins()[0], Path("reports/figures/backtest_baselines.png"))
    results.to_parquet(data.data_dir() / "processed" / "backtest_baselines.parquet")
    for name, r in summary.items():
        p, t = r["products"], r["total"]
        print(f"{name:16} products wMAPE {p['wmape']:.3f} median {p['median_product_wmape']:.3f}"
              f" | total wMAPE {t['wmape']:.3f}")  # fmt: skip
