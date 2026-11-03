import json

import numpy as np
import pandas as pd
import pytest

from demand import cli
from demand.calendar import planned_open_days
from demand.models.baselines import BASELINES
from demand.series import TOTAL


@pytest.mark.parametrize(
    "raw, safe",
    [
        ('=HYPERLINK("http://evil")', '\'=HYPERLINK("http://evil")'),
        ("+1+1", "'+1+1"),
        ("-2", "'-2"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tTAB", "'\tTAB"),
        ("WHITE HANGING HEART", "WHITE HANGING HEART"),
        ("85123A", "85123A"),
    ],
)
def test_cells_that_a_spreadsheet_would_run_are_neutralised(raw, safe):
    assert cli.safe_cell(raw) == safe


def test_next_open_days_skip_closures_and_stop_at_the_calendar_end():
    days = cli.next_open_days(pd.Timestamp("2011-12-09"), 14)
    assert days[0] == pd.Timestamp("2011-12-11")  # Saturday 10th is closed
    assert days[-1] == pd.Timestamp("2011-12-23")  # shut from the 24th
    assert len(days) == 12
    with pytest.raises(ValueError, match="calendar ends"):
        cli.next_open_days(pd.Timestamp("2011-12-31"), 3)
    with pytest.raises(ValueError, match="at least 1"):
        cli.next_open_days(pd.Timestamp("2011-06-01"), 0)


def panel(end="2011-11-05"):
    days = planned_open_days(pd.Timestamp("2011-01-04"), pd.Timestamp(end))
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({"85123A": rng.integers(5, 50, len(days)), "=EVIL": 3}, index=days)
    return wide.assign(**{TOTAL: wide.sum(axis=1)})


QUANTILES = {
    "products": pd.DataFrame({"low": -0.5, "high": 0.5}, index=pd.Index(range(1, 15), name="h")),
    "total": pd.DataFrame({"low": -0.1, "high": 0.1}, index=pd.Index(range(1, 15), name="h")),
}


def test_forecast_rows_intervals_and_safe_text():
    out = cli.forecast(panel(), "weekday_mean_4", 5, QUANTILES, {"85123A": "=cmd|calc"})
    assert out["date"].nunique() == 5
    assert set(out["series"]) == {"85123A", "'=EVIL", TOTAL}
    assert set(out["description"]) == {"'=cmd|calc", "", "all products"}
    assert (out["lower"] <= out["forecast"]).all() and (out["forecast"] <= out["upper"]).all()
    assert (out["lower"] >= 0).all()
    total = out[out["series"] == TOTAL]
    product = out[out["series"] == "85123A"]
    # The total uses its own, narrower quantiles.
    assert ((total["upper"] - total["lower"]) / total["forecast"]).max() < (
        (product["upper"] - product["lower"]) / product["forecast"]
    ).min()


def test_forecast_without_intervals_and_unknown_model():
    out = cli.forecast(panel(), "naive", 3)
    assert "lower" not in out.columns and len(out) == 9
    with pytest.raises(ValueError, match="unknown model"):
        cli.forecast(panel(), "crystal_ball", 3)


@pytest.fixture
def fast_registry(monkeypatch):
    fast = {cli.PICK: BASELINES["naive"], "ets": BASELINES["naive"]}
    monkeypatch.setattr(cli, "REGISTRY", BASELINES | fast)


def test_holdout_is_scored_once(tmp_path, monkeypatch, fast_registry):
    monkeypatch.setenv("DEMAND_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "processed").mkdir(parents=True)
    report = tmp_path / "holdout.json"
    summary = cli.score_holdout(panel(), report)
    assert summary["origins"] == ["2011-10-15", "2011-10-22", "2011-10-29"]  # data ends 4 Nov
    assert set(summary["models"]) == {cli.PICK, "ets", "weekday_mean_4"}
    assert json.loads(report.read_text())["last_day"] == "2011-11-04"
    with pytest.raises(RuntimeError, match="has been scored"):
        cli.score_holdout(panel(), report)


def test_holdout_needs_holdout_data(tmp_path, fast_registry):
    with pytest.raises(ValueError, match="no data in the holdout"):
        cli.score_holdout(panel(end="2011-10-01"), tmp_path / "holdout.json")


def test_main_explains_missing_data_and_a_bad_output_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DEMAND_DATA_DIR", str(tmp_path))
    assert cli.main(["forecast", "--out", str(tmp_path / "f.csv")]) == 1
    assert "demand data" in capsys.readouterr().err
    (tmp_path / "processed").mkdir()
    panel().to_parquet(tmp_path / "processed" / "daily_units.parquet")
    assert cli.main(["forecast", "--out", str(tmp_path / "f.xlsx")]) == 2
    monkeypatch.chdir(tmp_path)
    assert cli.main(["forecast", "--days", "3", "--model", "naive", "--out", "out/f.csv"]) == 0
    written = pd.read_csv(tmp_path / "out" / "f.csv")
    assert len(written) == 9
    with pytest.raises(SystemExit):
        cli.main(["forecast", "--model", "crystal_ball", "--out", "f.csv"])
