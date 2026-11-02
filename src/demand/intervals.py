"""Prediction intervals from past forecast errors, checked on origins they weren't set on.

For each horizon, the errors at earlier origins are divided by each series' recent level, so a
product selling 5 a day and one selling 500 share one set of quantiles. An interval is the
forecast plus the level times those quantiles, never below zero. Whether "80%" really covers
80% is then measured on later origins only.
"""

import numpy as np
import pandas as pd

from demand.features import RECENT


def levels(wide: pd.DataFrame, at: list[pd.Timestamp]) -> pd.DataFrame:
    """Each series' mean over the RECENT open days before each origin (at least 1)."""
    rows = []
    for origin in at:
        history = wide.loc[wide.index < origin].iloc[-RECENT:]
        if history.empty:
            raise ValueError(f"no history before {origin.date()}")
        level = history.mean().clip(lower=1.0)
        rows.append(level.rename_axis("series").rename("level").reset_index().assign(origin=origin))
    return pd.concat(rows, ignore_index=True)


def fit(results: pd.DataFrame, level: pd.DataFrame, coverage: float = 0.8) -> pd.DataFrame:
    """Lower and upper scaled-error quantiles per horizon, from these results only."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    scaled = _scaled(results, level)
    low, high = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    return (
        scaled.groupby("horizon")["scaled_error"]
        .quantile([low, high])
        .unstack()
        .set_axis(["low", "high"], axis=1)
    )


def apply(results: pd.DataFrame, level: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    """Add `lower` and `upper` to each forecast. Horizons the quantiles don't cover are refused."""
    missing = set(results["horizon"]) - set(quantiles.index)
    if missing:
        raise ValueError(f"no interval for horizons {sorted(missing)}")
    out = _scaled(results, level).join(quantiles, on="horizon")
    out["lower"] = np.clip(out["forecast"] + out["level"] * out["low"], 0, None)
    out["upper"] = np.clip(out["forecast"] + out["level"] * out["high"], 0, None)
    return out.drop(columns=["low", "high", "scaled_error"])


def coverage(with_intervals: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Share of actuals inside their interval, and the mean width relative to the level."""
    d = with_intervals.assign(
        inside=lambda x: (x["actual"] >= x["lower"]) & (x["actual"] <= x["upper"]),
        width=lambda x: (x["upper"] - x["lower"]) / x["level"],
    )
    if by is None:
        return pd.DataFrame(
            {"coverage": [d["inside"].mean()], "width": [d["width"].mean()], "rows": [len(d)]}
        )
    return d.groupby(by).agg(
        coverage=("inside", "mean"), width=("width", "mean"), rows=("inside", "size")
    )


def _scaled(results: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    out = results.merge(level, on=["origin", "series"], how="left", validate="many_to_one")
    if out["level"].isna().any():
        raise ValueError("a forecast has no level for its origin and series")
    out["scaled_error"] = (out["actual"] - out["forecast"]) / out["level"]
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path

    from demand import data
    from demand.backtest import origins, run_backtest
    from demand.metrics import from_arrays
    from demand.models.baselines import BASELINES
    from demand.models.gbm import VARIANTS
    from demand.plots import plot_horizon_and_coverage
    from demand.series import TOTAL

    out = data.data_dir() / "processed"
    saved = out / "backtest_staggered.parquet"
    wide = pd.read_parquet(out / "daily_units.parquet")
    at = origins(staggered=True)
    if not saved.is_file():
        models = {
            "weekday_mean_4": BASELINES["weekday_mean_4"],
            "lightgbm_no_christmas": VARIANTS["lightgbm_no_christmas"],
        }
        run_backtest(wide, models, at).to_parquet(saved)
    results = pd.read_parquet(saved)
    products = results[results["series"] != TOTAL]

    report = {"origins": [str(d.date()) for d in at], "by_horizon": {}, "intervals": {}}
    for model, m in products.groupby("model"):
        report["by_horizon"][model] = {
            int(h): {
                "wmape": from_arrays(d["actual"].to_numpy(), d["forecast"].to_numpy())["wmape"],
                "origins": int(d["origin"].nunique()),
            }
            for h, d in m.groupby("horizon")
        }

    # Set the intervals on the first half of the origins, check them on the second half.
    level = levels(wide, at)
    early, late = at[: len(at) // 2], at[len(at) // 2 :]
    checked = {}
    for model, m in products.groupby("model"):
        q = fit(m[m["origin"].isin(early)], level)
        later = apply(m[m["origin"].isin(late)], level, q)
        later["month"] = later["origin"].dt.month
        checked[model] = later
        report["intervals"][model] = {
            "nominal": 0.8,
            "set_on": [str(early[0].date()), str(early[-1].date())],
            "checked_on": [str(late[0].date()), str(late[-1].date())],
            "overall": coverage(later).iloc[0].to_dict(),
            "by_month": coverage(later, "month").round(3).to_dict(orient="index"),
            "by_horizon": coverage(later, "horizon")["coverage"].round(3).to_dict(),
        }
    Path("reports/intervals.json").write_text(json.dumps(report, indent=2, default=float))
    plot_horizon_and_coverage(report, checked, Path("reports/figures/horizon_intervals.png"))
    print(json.dumps({m: r["overall"] for m, r in report["intervals"].items()}, indent=2))
