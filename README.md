# demand-forecasting

Forecasting how many units of each product a UK online gift retailer will sell per day, two weeks ahead, using two years of real transactions (UCI Online Retail II).

The point is to forecast honestly rather than chase a low error on one lucky split. Every model is compared against naive baselines, scored with rolling-origin backtests over many cut-off dates, and checked by tests that make sure no feature can see the future.

**Status:** in progress (day 3 of 11).

## So far

**Cleaning** ([clean.py](src/demand/clean.py)). 1,044,848 invoice lines go in (after cutting the 22,523 rows the two sheets share) and 997,186 completed product sales come out. Each step is counted in `reports/cleaning.json`:

| Step | Rows removed |
|------|-------------:|
| Lines entered twice | 11,812 |
| Cancellation lines (their units come off the sales they cancel, below) | 19,104 |
| Postage, fees, manual adjustments, vouchers (anything not a 5-digit product code) | 4,784 |
| Stock write-offs and zero prices | 5,928 |
| Sales cancelled in full | 6,028 |
| Bad-debt adjustments | 6 |

Cancellations were the interesting part. Only 36% cancel a sale of exactly the same quantity, so each cancellation instead takes its units off the same customer's earlier sales of that product, newest first. That removes 450,259 cancelled units. 17,480 units couldn't be matched (no customer id, or the sale was before the data starts) and are reported, not guessed. Two orders of 80,995 and 74,215 units, each cancelled within minutes, are gone.

Large orders stay in but are flagged: 7,335 lines (0.7%) carry 2.4 million units, 22% of all demand. That's wholesale lumpiness the models will have to live with. Day 8 compares results with and without them.

**Daily series** ([series.py](src/demand/series.py)). The shop traded on 604 days and was closed on 135: 104 Saturdays plus Christmas and bank holidays. Closed days are left out instead of counted as zero sales. The 100 products that sold on the most days before 16 Apr 2011 (picked before any backtest data, so no look-ahead) make up 20.5% of units and still have no sales on 14% of open days.

**EDA** ([notebook](notebooks/01_eda.ipynb), [eda.py](src/demand/eda.py)). Only data before the 15 Oct 2011 holdout, so the holdout stays unseen.

![Units per open day by month](reports/figures/eda_season.png)

- **The peak is November, not December.** From September, sales per open day run 1.54 times the January to August level in 2010 and 1.55 times in 2011. The customers are mostly wholesalers stocking up for Christmas. December is 35% below November even per open day, so the drop isn't only the shop closing from 24 December.
- **No growth to model.** January to August 2011 is slightly below 2010.
- **Thursday sells 1.23 times an average open day, Sunday 0.57.** Saturday is closed. The seasonal-naive baseline should copy the same weekday last week, not yesterday.
- **Every product sells on most open days, but the amounts swing.** All 100 are "erratic" or "lumpy" by the usual cut-offs. Zeros aren't the problem, so Croston-style methods aren't needed. Without large orders, the median variation (CV²) halves from 2.2 to 1.1, but 99 of 100 are still erratic or lumpy.

![Demand classes with and without large orders](reports/figures/eda_demand_classes.png)

What this changes for the next days: the backtest's 26 weekly origins run from April to October, and only the last 8 forecast into September. The holdout is all peak season. So backtest errors will look better than holdout errors, and a model that can't follow the season will look better than it is until the holdout. MAE and wMAPE are the main metrics, because RMSE gets pulled around by a few huge days. 11 tests.

## Planned approach

1. Clean the transactions: cancellations, returns, non-product rows (postage, fees) and data-entry outliers
2. Build a daily series per product, treating days the shop was closed as missing, not as zero demand
3. Baselines: naive and seasonal-naive (same weekday last week)
4. Models: exponential smoothing / SARIMA, then LightGBM on lag, rolling and calendar features
5. Score with MAE, RMSE and wMAPE (not MAPE, which breaks on zero-sales days), by forecast horizon

## Planned stack

Python 3.12, uv, pandas, statsmodels, LightGBM, matplotlib, pytest, ruff. CI: GitHub Actions with ruff, pytest, pip-audit and Gitleaks.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) by Daqing Chen, UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). DOI: 10.24432/C5CG6D. It isn't committed: a script downloads it and checks its hash.

## Quick start

```bash
uv sync
make lint test
make data   # downloads 46 MB, checks its SHA-256, builds the daily series
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
