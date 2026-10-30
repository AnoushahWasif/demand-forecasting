# demand-forecasting

Forecasting how many units of each product a UK online gift retailer will sell per day, two weeks ahead, using two years of real transactions (UCI Online Retail II).

The point is to forecast honestly rather than chase a low error on one lucky split. Every model is compared against naive baselines, scored with rolling-origin backtests over many cut-off dates, and checked by tests that make sure no feature can see the future.

**Status:** in progress (day 1 of 11). Results will go here once there are any.

## Planned approach

1. Clean the transactions: cancellations, returns, non-product rows (postage, fees) and data-entry outliers
2. Build a daily series per product, treating days the shop was closed as missing, not as zero demand
3. Baselines: naive and seasonal-naive (same weekday last week)
4. Models: exponential smoothing / SARIMA, then LightGBM on lag, rolling and calendar features
5. Score with MAE, RMSE and wMAPE (not MAPE, which breaks on zero-sales days), by forecast horizon

## Stack

Python 3.12, uv, pandas, statsmodels, LightGBM, matplotlib, pytest, ruff. CI: GitHub Actions with ruff, pytest, pip-audit and Gitleaks.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) by Daqing Chen, UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). DOI: 10.24432/C5CG6D. It isn't committed: a script downloads it and checks its hash.

## Quick start

```bash
uv sync
make lint test
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
