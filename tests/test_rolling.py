"""The rolling backtest, and that no forecaster can see past its origin."""

import numpy as np
import pandas as pd
import pytest

from demand.backtest import forecast_dates, origins, run_backtest, summarise
from demand.calendar import mismatches, planned_closed, planned_open_days
from demand.models.baselines import BASELINES
from demand.series import TOTAL

START, END = pd.Timestamp("2011-01-04"), pd.Timestamp("2011-07-31")
STOP = pd.Timestamp("2011-07-15")


def panel():
    days = planned_open_days(START, END)
    rng = np.random.default_rng(1)
    wide = pd.DataFrame(
        {"A": rng.integers(0, 50, len(days)), "B": rng.integers(0, 5, len(days))}, index=days
    )
    return wide.assign(**{TOTAL: wide.sum(axis=1)})


AT = origins(pd.Timestamp("2011-05-07"), n=10)  # Saturdays; the last ones run into STOP


# --- the calendar


def test_calendar_knows_the_planned_closures():
    days = pd.DatetimeIndex(["2011-04-22", "2011-04-24", "2011-04-29", "2010-12-27",
                             "2011-01-03", "2011-03-05", "2011-03-07"])  # fmt: skip
    assert planned_closed(days).tolist() == [True] * 6 + [False]


def test_mismatches_lists_both_kinds():
    days = planned_open_days(pd.Timestamp("2011-03-01"), pd.Timestamp("2011-03-14"))
    actual = days.drop(pd.Timestamp("2011-03-09")).append(pd.DatetimeIndex(["2011-03-12"]))
    found = mismatches(actual.sort_values())
    assert found == {
        "open_but_planned_closed": ["2011-03-12"],
        "closed_but_planned_open": ["2011-03-09"],
    }


# --- origins and dates


def test_origins_are_weekly():
    at = origins(pd.Timestamp("2011-04-16"), n=3)
    assert [d.date().isoformat() for d in at] == ["2011-04-16", "2011-04-23", "2011-04-30"]


def test_forecast_dates_skip_planned_closures_and_stop_at_the_holdout():
    dates = forecast_dates(pd.Timestamp("2011-04-21"), horizon=6, stop=None)
    assert [d.day for d in dates] == [21, 26, 27, 28, 1, 3]  # Easter, royal wedding, May Day
    assert forecast_dates(pd.Timestamp("2011-07-11"), horizon=14, stop=STOP).max() < STOP


# --- no leaks


def test_every_forecaster_only_sees_data_before_its_origin():
    seen = []

    def spy(history, dates):
        seen.append((history.index.max(), dates.min(), dates.max()))
        return BASELINES["naive"](history, dates)

    run_backtest(panel(), {"spy": spy}, AT, stop=STOP)
    assert len(seen) == len(AT)
    for origin, (last_seen, first, last) in zip(AT, seen, strict=True):
        assert last_seen < origin <= first
        assert last < STOP


@pytest.mark.parametrize("name", list(BASELINES))
def test_changing_the_future_never_changes_a_forecast(name):
    clean = panel()
    poisoned = clean.copy()
    first_origin = AT[0]
    poisoned.loc[poisoned.index >= first_origin] *= 1000  # every value any origin could peek at
    a = run_backtest(clean, {name: BASELINES[name]}, AT[:1], stop=STOP)
    b = run_backtest(poisoned, {name: BASELINES[name]}, AT[:1], stop=STOP)
    assert np.array_equal(a["forecast"], b["forecast"])
    assert not np.array_equal(a["actual"], b["actual"])


def test_a_forecaster_changing_its_input_cannot_affect_the_next():
    def vandal(history, dates):
        history.iloc[:] = 0
        return BASELINES["naive"](history, dates)

    both = run_backtest(panel(), {"vandal": vandal, "naive": BASELINES["naive"]}, AT[:1], stop=STOP)
    alone = run_backtest(panel(), {"naive": BASELINES["naive"]}, AT[:1], stop=STOP)
    assert np.array_equal(both.loc[both["model"] == "naive", "forecast"], alone["forecast"])


def test_a_forecast_for_the_wrong_dates_is_refused():
    def shifted(history, dates):
        return BASELINES["naive"](history, dates + pd.Timedelta(days=1))

    with pytest.raises(ValueError, match="wrong dates"):
        run_backtest(panel(), {"bad": shifted}, AT[:1], stop=STOP)


def test_an_unplanned_closure_is_refused():
    wide = panel().drop(pd.Timestamp("2011-05-10"))
    with pytest.raises(ValueError, match="disagree"):
        run_backtest(wide, {"naive": BASELINES["naive"]}, AT[:1], stop=STOP)


# --- results and summary


def test_results_have_one_row_per_origin_model_date_and_series():
    results = run_backtest(panel(), BASELINES, AT, stop=STOP)
    assert not results.duplicated(["origin", "model", "date", "series"]).any()
    assert set(results["model"]) == set(BASELINES)
    assert results["horizon"].between(1, 14).all()
    assert (results["date"] >= results["origin"]).all()


def test_summary_counts_origins_per_horizon_and_scores_skill():
    summary = summarise(run_backtest(panel(), BASELINES, AT, stop=STOP))
    horizons = summary["naive"]["by_horizon"]
    assert horizons[0]["origins"] == len(AT)
    assert horizons[-1]["origins"] < len(AT)  # the last origins are cut at STOP
    assert summary["seasonal_naive"]["products"]["mae_skill_vs_seasonal_naive"] == 0
    assert len(summary["naive"]["by_origin"]) == len(AT)
    assert summary["naive"]["products"]["median_product_wmape"] > 0


def test_horizon_scores_leave_the_total_out_of_the_product_numbers():
    results = run_backtest(panel(), {"naive": BASELINES["naive"]}, AT, stop=STOP)
    summary = summarise(results)["naive"]
    by_h = [h["products_wmape"] for h in summary["by_horizon"]]
    # Pooled wMAPE is a sales-weighted mean of the horizons, so it must lie between them.
    assert min(by_h) <= summary["products"]["wmape"] <= max(by_h)
    h1 = results[(results["horizon"] == 1) & (results["series"] != TOTAL)]
    expected = (h1["forecast"] - h1["actual"]).abs().sum() / h1["actual"].sum()
    assert summary["by_horizon"][0]["products_wmape"] == pytest.approx(expected)


def test_the_calendar_refuses_years_it_does_not_cover():
    with pytest.raises(ValueError, match="only covers"):
        planned_closed(pd.DatetimeIndex(["2012-04-06"]))  # Good Friday 2012


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_forecast_with_missing_values_is_refused(bad):
    def broken(history, dates):
        fc = BASELINES["naive"](history, dates)
        fc.iloc[0, 0] = bad
        return fc

    with pytest.raises(ValueError, match="missing or infinite"):
        run_backtest(panel(), {"broken": broken}, AT[:1], stop=STOP)
