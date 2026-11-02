"""Forecast origins, the split at each one, and the rolling backtest over all of them."""

from collections.abc import Callable

import numpy as np
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

    end = origin + pd.Timedelta(days=3 * horizon + 14)
    if stop is not None:
        end = min(end, stop)
    dates = planned_open_days(origin, end)
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
            if not np.isfinite(fc.to_numpy(dtype=float)).all():
                raise ValueError(f"{name} returned missing or infinite values at {origin.date()}")
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


def summarise(
    results: pd.DataFrame, references: tuple[str, ...] = ("seasonal_naive", "weekday_mean_4")
) -> dict:
    """Pooled, per horizon (with how many origins reach it), per origin, and per product."""
    from demand.metrics import from_arrays, skill
    from demand.series import TOTAL

    def pooled(d):
        return from_arrays(d["actual"].to_numpy(), d["forecast"].to_numpy())

    def _wmape(d):
        return pooled(d)["wmape"] if len(d) else None

    out = {}
    for model, m in results.groupby("model", sort=False):
        products, total = m[m["series"] != TOTAL], m[m["series"] == TOTAL]
        level = products if len(products) else total
        per_origin = level.groupby("origin").apply(lambda d: pooled(d)["wmape"])
        out[model] = {"total": pooled(total)} if len(total) else {}
        if len(products):
            per_product = products.groupby("series").apply(lambda d: pooled(d)["wmape"])
            out[model]["products"] = {
                **pooled(products),
                "median_product_wmape": float(per_product.median()),
            }
        out[model] |= {
            "by_horizon": [
                {
                    "horizon": int(h),
                    "origins": int(d["origin"].nunique()),
                    "products_wmape": _wmape(d[d["series"] != TOTAL]),
                    "total_wmape": _wmape(d[d["series"] == TOTAL]),
                }
                for h, d in m.groupby("horizon")
            ],  # fmt: skip
            "by_origin": {str(o.date()): w for o, w in per_origin.items()},
        }
    for ref in references:
        if ref not in out:
            continue
        for result in out.values():
            for level in ("products", "total"):
                if level in result and level in out[ref]:
                    result[level][f"mae_skill_vs_{ref}"] = skill(
                        result[level]["mae"], out[ref][level]["mae"]
                    )
    return out


def counting(forecast: Forecaster, counts: dict, name: str) -> Forecaster:
    """Wrap a forecaster to add up the fallbacks it reports in `attrs`."""

    def wrapped(history, dates):
        fc = forecast(history, dates)
        counts[name] = counts.get(name, 0) + fc.attrs.get("fallbacks", 0)
        return fc

    return wrapped


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from demand import data
    from demand.models.baselines import BASELINES
    from demand.series import TOTAL

    parser = argparse.ArgumentParser()
    parser.add_argument("--statistical", action="store_true", help="add ETS and SARIMA (slow)")
    parser.add_argument("--rescore", action="store_true", help="re-summarise the saved results")
    parser.add_argument("--add", nargs="+", help="run only these models, merged into saved results")
    args = parser.parse_args()

    out = data.data_dir() / "processed"
    saved = {"all": out / "backtest_results.parquet", "total": out / "backtest_total.parquet"}
    counts_path = out / "backtest_fallbacks.json"
    if args.add:
        from demand.models.gbm import VARIANTS
        from demand.models.statistical import MODELS

        registry = dict(BASELINES) | MODELS | VARIANTS
        unknown = set(args.add) - set(registry)
        if unknown:
            raise SystemExit(f"unknown models: {sorted(unknown)}")
        wide = pd.read_parquet(out / "daily_units.parquet")
        counts = json.loads(counts_path.read_text()) if counts_path.is_file() else {}
        for name in args.add:
            counts[name] = 0
        new = run_backtest(wide, {n: counting(registry[n], counts, n) for n in args.add}, origins())
        old = pd.read_parquet(saved["all"])
        pd.concat([old[~old["model"].isin(args.add)], new]).to_parquet(saved["all"])
        counts_path.write_text(json.dumps(counts))
    elif not args.rescore:
        wide = pd.read_parquet(out / "daily_units.parquet")
        forecasters, counts = dict(BASELINES), {}
        if args.statistical:
            from demand.models.statistical import MODELS, sarima

            forecasters |= {name: counting(f, counts, name) for name, f in MODELS.items()}
        run_backtest(wide, forecasters, origins()).to_parquet(saved["all"])
        if args.statistical:
            # SARIMA takes seconds per fit, so it only runs on the shop total.
            sarima_only = {"sarima": counting(sarima, counts, "sarima")}
            run_backtest(wide[[TOTAL]], sarima_only, origins()).to_parquet(saved["total"])
        counts_path.write_text(json.dumps(counts))

    results = pd.read_parquet(saved["all"])
    summary = summarise(results)
    counts = json.loads(counts_path.read_text()) if counts_path.is_file() else {}
    n_series = results["series"].nunique()
    for name, result in summary.items():
        if name in counts:
            result["fallbacks"] = {"forecasts": counts[name], "of": len(origins()) * n_series}
    Path("reports/backtest.json").write_text(json.dumps(summary, indent=2))
    from demand.plots import LABELS, plot_backtest

    def product(m, key):
        return summary[m]["products"][key]

    # Decision 23: a model more than 10% off on units can't win on error alone.
    honest = [m for m in summary if abs(product(m, "bias")) < 0.1]
    best = min(honest, key=lambda m: product(m, "wmape"))
    title = f"{LABELS.get(best, best)}: lowest product error within 10% bias"
    title += f" (wMAPE {product(best, 'wmape'):.2f})"
    lower = [m for m in summary if product(m, "wmape") < product(best, "wmape")]
    if lower:
        trap = min(lower, key=lambda m: product(m, "wmape"))
        title += f"\n{LABELS.get(trap, trap)} is lower but {product(trap, 'bias'):+.0%} on units"
    plot_backtest(summary, origins()[0], title, Path("reports/figures/backtest.png"))
    for name, r in summary.items():
        p, t = r["products"], r["total"]
        print(f"{name:16} products wMAPE {p['wmape']:.3f} bias {p['bias']:+.2f}"
              f" | total wMAPE {t['wmape']:.3f} bias {t['bias']:+.2f}")  # fmt: skip

    if saved["total"].is_file() and "sarima" in counts:
        total = pd.concat([results[results["series"] == TOTAL], pd.read_parquet(saved["total"])])
        total_summary = summarise(total)
        total_summary["sarima"]["fallbacks"] = {"forecasts": counts["sarima"], "of": len(origins())}
        Path("reports/backtest_total.json").write_text(json.dumps(total_summary, indent=2))
        t = total_summary["sarima"]["total"]
        print(f"{'sarima':16} total wMAPE {t['wmape']:.3f} bias {t['bias']:+.2f}")
