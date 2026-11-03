"""Raw transactions -> cleaned sales -> daily series for the chosen products.

Writes data/processed/daily_units.parquet and reports/cleaning.json.
"""

import json
from pathlib import Path

import pandas as pd

from demand import data
from demand.clean import clean
from demand.series import (
    FIRST_ORIGIN,
    TOTAL,
    closed_days,
    daily_units,
    descriptions,
    open_days,
    pick_products,
)


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sales, report = clean(raw)
    days = open_days(raw)
    wide = daily_units(sales, days)
    products = pick_products(wide)
    closed = closed_days(days)
    report["days"] = {
        "open": len(days),
        "closed": len(closed),
        "closed_by_weekday": closed.day_name().value_counts().to_dict(),
        "first": str(days.min().date()),
        "last": str(days.max().date()),
    }
    picked = wide[products]
    share = picked.sum().sum() / wide[TOTAL].sum()
    report["products"] = {
        "picked_before": str(FIRST_ORIGIN.date()),
        "count": len(products),
        "share_of_units": round(float(share), 3),
        "zero_days_share": round(float((picked == 0).mean().mean()), 3),
        "descriptions": descriptions(sales, products),
    }
    return wide[[*products, TOTAL]], report


def main() -> None:
    series, report = build(data.load())
    out = data.data_dir() / "processed"
    out.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out / "daily_units.parquet")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/cleaning.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "products"}, indent=2))
    print({k: v for k, v in report["products"].items() if k != "descriptions"})


if __name__ == "__main__":
    main()
