"""demand data | backtest | forecast | holdout."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from demand import data
from demand.backtest import run_backtest, summarise
from demand.calendar import COVERS, planned_open_days
from demand.models.baselines import BASELINES
from demand.models.gbm import VARIANTS
from demand.models.statistical import MODELS
from demand.series import HOLDOUT_START, TOTAL

PICK = "lightgbm_no_christmas"  # decision 31
REFERENCES = ("ets", "weekday_mean_4")
REGISTRY = dict(BASELINES) | MODELS | VARIANTS
HOLDOUT_REPORT = Path("reports/holdout.json")
# A cell starting with one of these is read as a formula by Excel and LibreOffice.
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: str) -> str:
    """Neutralise spreadsheet formula injection in text read from the data."""
    value = str(value)
    return "'" + value if value.startswith(FORMULA_START) else value


def next_open_days(after: pd.Timestamp, days: int) -> pd.DatetimeIndex:
    """The next `days` planned open days after `after`, as far as the calendar reaches."""
    if days < 1:
        raise ValueError("days must be at least 1")
    start = after + pd.Timedelta(days=1)
    end = min(start + pd.Timedelta(days=3 * days + 14), COVERS[1])
    if start > end:
        raise ValueError(f"the closure calendar ends on {COVERS[1].date()}; extend it first")
    return planned_open_days(start, end)[:days]


def forecast(
    wide: pd.DataFrame,
    model: str,
    days: int,
    quantiles: dict[str, pd.DataFrame] | None = None,
    descriptions: dict | None = None,
) -> pd.DataFrame:
    """Forecast the next `days` open days from everything in `wide`. Long format, one row per
    date and series, with an 80% interval when backtest quantiles are given. Products and the
    shop total get their own quantiles: relative to its level, the total swings far less."""
    from demand.intervals import levels

    if model not in REGISTRY:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(REGISTRY)}")
    dates = next_open_days(wide.index.max(), days)
    fc = REGISTRY[model](wide, dates)
    out = (
        fc.rename_axis(index="date", columns="series")
        .stack()
        .rename("forecast")
        .reset_index()
        .assign(horizon=lambda d: d["date"].map({x: i + 1 for i, x in enumerate(dates)}))
    )
    if quantiles is not None:
        level = levels(wide, [dates[0]]).drop(columns="origin")
        out = out.merge(level, on="series")
        parts = []
        for group, part in out.groupby(out["series"] == TOTAL):
            q = quantiles["total" if group else "products"]
            parts.append(part.join(q, on="horizon"))
        out = pd.concat(parts).sort_values(["date", "series"], kind="stable")
        out["lower"] = (out["forecast"] + out["level"] * out["low"]).clip(lower=0)
        out["upper"] = (out["forecast"] + out["level"] * out["high"]).clip(lower=0)
        out = out.drop(columns=["level", "low", "high"]).reset_index(drop=True)
    descriptions = descriptions or {}
    out["description"] = out["series"].map(
        lambda s: "all products" if s == TOTAL else safe_cell(descriptions.get(s, ""))
    )
    out["series"] = out["series"].map(safe_cell)
    return out.round({"forecast": 1, "lower": 1, "upper": 1})


def holdout_origins(wide: pd.DataFrame) -> list[pd.Timestamp]:
    """Weekly origins from the holdout start while there is data to score."""
    last = wide.index.max()
    at, origin = [], HOLDOUT_START
    while origin <= last:
        at.append(origin)
        origin += pd.Timedelta(weeks=1)
    return at


def score_holdout(wide: pd.DataFrame, report: Path = HOLDOUT_REPORT) -> dict:
    """Score the pick and its references on the holdout, once. A second run is refused, so
    nobody can tune against the holdout and quietly score it again (decision 6)."""
    if report.exists():
        raise RuntimeError(f"{report} exists: the holdout has been scored. Delete it in a commit "
                           "that says why, if it really must be scored again.")  # fmt: skip
    at = holdout_origins(wide)
    if not at:
        raise ValueError("no data in the holdout")
    models = {name: REGISTRY[name] for name in (PICK, *REFERENCES)}
    results = run_backtest(wide, models, at, stop=None)
    summary = {
        "origins": [str(d.date()) for d in at],
        "last_day": str(wide.index.max().date()),
        "models": summarise(results, references=("weekday_mean_4", "ets")),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    results.to_parquet(data.data_dir() / "processed" / "holdout_results.parquet")
    return summary


def _quantiles(model: str) -> dict[str, pd.DataFrame] | None:
    """Interval quantiles from the staggered backtest, separately for products and the total."""
    from demand.backtest import origins
    from demand.intervals import fit, levels

    path = data.data_dir() / "processed" / "backtest_staggered.parquet"
    if not path.is_file():
        return None
    results = pd.read_parquet(path)
    results = results[results["model"] == model]
    if results.empty:
        return None
    wide = pd.read_parquet(data.data_dir() / "processed" / "daily_units.parquet")
    level = levels(wide, origins(staggered=True))
    is_total = results["series"] == TOTAL
    return {
        "products": fit(results[~is_total], level),
        "total": fit(results[is_total], level),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demand", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="download, check and clean the data; build the daily series")
    bt = sub.add_parser("backtest", help="rolling-origin backtest of every model")
    bt.add_argument("--statistical", action="store_true", help="include ETS and SARIMA (slow)")
    fc = sub.add_parser("forecast", help="forecast the next open days to a CSV")
    fc.add_argument("--days", type=int, default=14)
    fc.add_argument("--model", default=PICK, choices=sorted(REGISTRY))
    fc.add_argument("--out", type=Path, required=True)
    sub.add_parser("holdout", help="score the chosen model on the holdout, once")
    args = parser.parse_args(argv)

    processed = data.data_dir() / "processed" / "daily_units.parquet"
    if args.command == "data":
        from demand.build import main as build_main

        build_main()
        return 0
    if args.command == "backtest":
        from demand.backtest import main as backtest_main

        backtest_main(["--statistical"] if args.statistical else [])
        return 0
    if not processed.is_file():
        print(f"{processed} not found; run `demand data` first", file=sys.stderr)
        return 1
    wide = pd.read_parquet(processed)
    if args.command == "forecast":
        if args.out.suffix.lower() != ".csv":
            print("--out must be a .csv file", file=sys.stderr)
            return 2
        cleaning = Path("reports/cleaning.json")
        names = (
            json.loads(cleaning.read_text())["products"]["descriptions"]
            if cleaning.is_file()
            else {}
        )
        quantiles = _quantiles(args.model)
        if quantiles is None:
            print("no staggered backtest for this model: no intervals (run demand.intervals)")
        result = forecast(wide, args.model, args.days, quantiles, names)
        if result["date"].nunique() < args.days:
            print(f"only {result['date'].nunique()} planned open days before the calendar ends "
                  f"or the shop closes; forecasting those")  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"wrote {len(result)} rows for {result['date'].nunique()} days to {args.out}")
        return 0
    if args.command == "holdout":
        summary = score_holdout(wide)
        for name, r in summary["models"].items():
            p, t = r["products"], r["total"]
            print(f"{name:24} products wMAPE {p['wmape']:.3f} bias {p['bias']:+.2f}"
                  f" | total wMAPE {t['wmape']:.3f} bias {t['bias']:+.2f}")  # fmt: skip
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
