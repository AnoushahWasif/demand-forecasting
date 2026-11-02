"""When the shop is planned to be shut, known in advance. Forecast dates come from here.

Every closure in the data turned out to be planned: Saturdays, a Christmas break from 24 Dec
to 3 Jan, Easter (Good Friday to Monday) and England's bank holidays, including the 29 Apr
2011 royal wedding, announced in Nov 2010. `mismatches` checks that against the data.
"""

import pandas as pd

BANK_HOLIDAYS = pd.DatetimeIndex(
    [
        "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30",
        "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29",
    ]
)  # fmt: skip
EASTER_SUNDAYS = pd.DatetimeIndex(["2010-04-04", "2011-04-24"])
# The holiday lists above only cover these years. Past them, Easter would quietly count as open.
COVERS = (pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-31"))


def planned_closed(dates: pd.DatetimeIndex) -> pd.Series:
    """True where the shop is planned to be shut."""
    dates = pd.DatetimeIndex(dates).normalize()
    if len(dates) and (dates.min() < COVERS[0] or dates.max() > COVERS[1]):
        raise ValueError(
            f"the closure calendar only covers {COVERS[0].date()} to {COVERS[1].date()}"
        )
    christmas = ((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 3))
    closed = (
        (dates.day_of_week == 5)
        | christmas
        | dates.isin(BANK_HOLIDAYS)
        | dates.isin(EASTER_SUNDAYS)
    )
    return pd.Series(closed, index=dates)


def planned_open_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    days = pd.date_range(start, end, freq="D", name="date")
    return days[~planned_closed(days).to_numpy()]


def mismatches(actual_open: pd.DatetimeIndex) -> dict:
    """Days the calendar gets wrong: opened when planned shut, or shut when planned open."""
    planned = planned_open_days(actual_open.min(), actual_open.max())
    return {
        "open_but_planned_closed": [str(d.date()) for d in actual_open.difference(planned)],
        "closed_but_planned_open": [str(d.date()) for d in planned.difference(actual_open)],
    }
