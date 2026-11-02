"""LightGBM across all products, retrained at every origin on that origin's history only."""

import numpy as np
import pandas as pd

from demand.features import FEATURES, RECENT, rows, training_set

HORIZON = 14
PARAMS = {
    # Tweedie suits counts with many small days and a few huge ones, and aims at the mean.
    # An L1 loss would aim at the median and repeat ETS-on-logs' -48% bias (decision 23).
    "objective": "tweedie",
    "tweedie_variance_power": 1.3,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "seed": 42,
    "deterministic": True,
    "num_threads": 2,
    "verbose": -1,
}
ROUNDS = 300


def lightgbm(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    import lightgbm as lgb

    if len(dates) and dates.min() <= history.index.max():
        raise ValueError("forecast dates must come after the history")
    if len(history) < RECENT + HORIZON:
        raise ValueError("not enough history to train on")
    train = training_set(history, HORIZON)
    model = lgb.train(PARAMS, lgb.Dataset(train[FEATURES], train["target"]), ROUNDS)
    pred = rows(history, dates)
    pred["forecast"] = np.clip(model.predict(pred[FEATURES]), 0, None) * pred["scale"]
    wide = pred.pivot(index="date", columns="series", values="forecast")
    forecast = wide.reindex(index=dates, columns=history.columns)
    forecast.attrs["fallbacks"] = 0
    return forecast
