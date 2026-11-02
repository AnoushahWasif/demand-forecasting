from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {
    "weekday_mean_4": "#2a78d6",
    "seasonal_naive": "#eb6834",
    "naive": "#a3a29e",
    "ets": "#1baf7a",
    "ets_log": "#8a5cd0",
    "lightgbm_no_christmas": "#0b0b0b",
}
LABELS = {
    "weekday_mean_4": "mean of last 4 same weekdays",
    "seasonal_naive": "seasonal naive",
    "naive": "naive",
    "ets": "ETS",
    "ets_log": "ETS on log units",
    "lightgbm": "LightGBM",
    "lightgbm_no_christmas": "LightGBM, no Christmas feature",
    "lightgbm_products_only": "LightGBM trained on products only",
}


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_backtest(summary: dict, first_origin: pd.Timestamp, title: str, path: Path) -> None:
    """Products wMAPE at each origin, and by horizon labelled with the weekday it falls on."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), dpi=150, width_ratios=[3, 2])
    # Only the models with a colour; the ablations are compared in notebook 02.
    for model, result in summary.items():
        if model not in COLORS:
            continue
        color, label = COLORS[model], LABELS.get(model, model)
        by_origin = pd.Series(result["by_origin"])
        left.plot(pd.to_datetime(by_origin.index), by_origin.to_numpy(), color=color, lw=1.8,
                  marker="o", ms=3, label=label)  # fmt: skip
        horizons = result["by_horizon"]
        right.plot([h["horizon"] for h in horizons], [h["products_wmape"] for h in horizons],
                   color=color, lw=1.8, marker="o", ms=3)  # fmt: skip
    left.set_ylabel("wMAPE, 100 products pooled", color=MUTED)
    left.set_xlabel("forecast origin (each forecasts the next 14 open days)", color=MUTED)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    left.set_ylim(0, None)
    # Every origin is the same weekday, so each horizon always lands on the same weekday.
    first = summary[next(iter(summary))]["by_horizon"]
    dates = pd.bdate_range(
        first_origin, periods=len(first) + 4, freq="C", weekmask="Sun Mon Tue Wed Thu Fri"
    )
    ticks = [h["horizon"] for h in first]
    right.set_xticks(ticks, [f"{h}\n{dates[h - 1]:%a}" for h in ticks], fontsize=7)
    right.set_xlabel("open days ahead (and the weekday it usually is)", color=MUTED)
    right.set_ylim(0, None)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    for ax in (left, right):
        _style(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_horizon_and_coverage(report: dict, checked: dict, path: Path) -> None:
    """Left: error by horizon with staggered origins. Right: 80% interval coverage by month."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for model, by_h in report["by_horizon"].items():
        hs = sorted(by_h, key=int)
        color, label = COLORS.get(model, MUTED), LABELS.get(model, model)
        wmape = [by_h[h]["wmape"] for h in hs]
        left.plot([int(h) for h in hs], wmape, marker="o", ms=3, lw=1.8, color=color, label=label)
    left.set_xlabel("open days ahead (origins spread over all weekdays)", color=MUTED)
    left.set_ylabel("wMAPE, 100 products pooled", color=MUTED)
    left.set_ylim(0, None)
    left.legend(frameon=False, fontsize=8, loc="lower right")
    months = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
    width = 0.38
    for i, (model, later) in enumerate(checked.items()):
        by = later.groupby("month")
        cov = by.apply(lambda d: ((d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"])).mean())
        x = np.arange(len(cov)) + (i - 0.5) * width
        right.bar(x, cov.to_numpy(), width=width * 0.9, color=COLORS.get(model, MUTED))
    right.axhline(0.8, color=INK, lw=1, ls="--")
    right.set_xticks(np.arange(len(cov)), [months[m] for m in cov.index])
    right.set_ylim(0, 1)
    right.set_ylabel("actuals inside the 80% interval", color=MUTED)
    right.set_xlabel("month of the origin (origins the intervals were not set on)", color=MUTED)
    model = (
        "lightgbm_no_christmas"
        if "lightgbm_no_christmas" in report["intervals"]
        else next(iter(report["intervals"]))
    )
    overall = report["intervals"][model]["overall"]["coverage"]
    fig.suptitle(
        f"Error barely grows with horizon; the 80% interval covers {overall:.0%} on later origins",
        x=0.01, ha="left", color=INK, fontsize=11,
    )  # fmt: skip
    for ax in (left, right):
        _style(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
