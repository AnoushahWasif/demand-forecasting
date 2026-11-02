"""Splitting a series at a forecast origin. The rolling backtest over many origins is Day 5."""

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


if __name__ == "__main__":
    import json
    from pathlib import Path

    from demand import data
    from demand.series import FIRST_ORIGIN

    wide = pd.read_parquet(data.data_dir() / "processed" / "daily_units.parquet")
    result = score_baselines(wide, FIRST_ORIGIN)
    Path("reports/baselines.json").write_text(json.dumps(result, indent=2))
    for name, r in result["models"].items():
        p, t = r["products"], r["total"]
        print(f"{name:16} products MAE {p['mae']:.1f} wMAPE {p['wmape']:.3f} | "
              f"total MAE {t['mae']:.0f} wMAPE {t['wmape']:.3f}")  # fmt: skip
