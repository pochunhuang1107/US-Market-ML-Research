# US Market ML Research

Research workspace for ML-driven ETF strategy experiments on curated US market data.

This repository is intentionally separate from `US-Market-Data`. The data repository is responsible for ingestion, cleaning, normalization, and publishing curated ETF daily parquet datasets to S3. This ML repository starts downstream of that work and focuses on research workflows: feature generation, label generation, walk-forward validation, model training, signal generation, and signal backtesting.

## First Milestone: ETF Rotation MLP v1

The first research direction is an ETF rotation system:

- Read curated ETF OHLCV parquet data produced by `US-Market-Data`.
- Generate leakage-safe historical features.
- Generate forward-return labels.
- Train baseline models before neural networks.
- Evaluate an MLP v1 model once the baseline pipeline is stable.
- Backtest top-K ETF rotation signals against SPY, QQQ, and equal-weight ETF baselines.

## Current Scope

This bootstrap includes only the project foundation:

- Python package layout under `src/us_market_ml`.
- Configuration placeholders for data, features, models, and experiments.
- Packaging metadata for Python 3.11+.
- Basic development tooling configuration.
- A minimal import test.
- A baseline supervised regression experiment runner for already-loaded OHLCV
  DataFrames and explicit walk-forward folds.
- Deterministic top-K equal-weight signal generation from model prediction
  scores, preserving experiment output rows for future backtesting.
- A simple deterministic signal backtester that lags target weights by symbol
  and converts ETF close prices into daily portfolio returns and an equity
  curve for research use.
- A first end-to-end baseline ETF rotation workflow that composes model
  prediction, top-K target weights, and lagged signal backtesting across
  walk-forward folds.
- Local CSV/parquet readers for curated ETF daily OHLCV research exports.
- Optional S3 parquet readers for curated ETF daily price partitions.
- An optional PyTorch tabular MLP regressor that fits supervised ETF feature
  datasets, optionally standardizes targets using train-only statistics, and
  preserves row alignment during prediction.
- Static benchmark report utilities that compare MLP test-period strategy
  equity against SPY, QQQ, and equal-weight universe benchmarks.
- A walk-forward benchmark report that stitches multiple MLP test-period
  backtests into one longer out-of-sample strategy-vs-benchmark chart.
- A local candidate comparison report that overlays multiple selected MLP
  candidate OOS curves against SPY, QQQ, and equal-weight benchmarks.

## Not Included Yet

The following are intentionally not implemented in this task:

- S3 writing or artifact publishing.
- Sequence, convolutional, transformer, registry, and artifact-writing model
  infrastructure beyond the first tabular MLP.
- Transaction-cost, slippage, and order-execution simulation.
- Live trading or brokerage integration.
- Large notebooks or exploratory artifacts.

## Local Data Loading

Local research exports can be loaded from a single CSV/parquet file or a
directory of files:

```python
from us_market_ml.data import read_local_market_data

ohlcv = read_local_market_data("data/curated/etf_daily_ohlcv")
```

The local reader validates required OHLCV columns, parses `date`, sorts by
`symbol` and `date`, and preserves extra curated columns such as `year`,
`month`, and `transactions`. It does not read S3, build features or labels,
write artifacts, or mutate source files.

## S3 Curated ETF Data Loading

S3 loading is optional and uses the curated ETF parquet layout produced by
`US-Market-Data`:

```text
curated/etf_daily_prices/symbol=SYMBOL/year=YYYY/data.parquet
```

Install AWS reader dependencies only when you need them. The S3 reader uses
`boto3` to fetch parquet object bytes directly:

```bash
python -m pip install -e ".[dev,aws]"
```

```python
from us_market_ml.data import S3MarketDataConfig, read_s3_curated_etf_data

config = S3MarketDataConfig(
    bucket="us-market-data-casper",
    region_name="us-west-2",
)

ohlcv = read_s3_curated_etf_data(
    symbols=["SPY", "QQQ", "IWM"],
    years=[2022, 2023, 2024],
    config=config,
)
```

The S3 reader validates required OHLCV columns, parses `date`, sorts by
`symbol` and `date`, and preserves extra curated columns. Missing symbol/year
partitions are skipped unless all requested partitions are missing. Feature
generation, labels, modeling, and artifact writing remain downstream concerns.

## Baseline Smoke CLI

Install the package in editable mode, then run the baseline end-to-end smoke
test against either a local curated CSV/parquet file or directory:

```bash
python -m pip install -e ".[dev]"

us-market-ml-baseline-smoke \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --train-start 2020-01-01 \
  --train-end 2020-12-31 \
  --validation-start 2021-01-01 \
  --validation-end 2021-03-31 \
  --test-start 2021-04-01 \
  --test-end 2021-06-30 \
  --top-k 3
```

To run the same smoke workflow directly from curated ETF parquet partitions in
S3, install the optional AWS dependencies and provide a bucket, symbols, and
years:

```bash
python -m pip install -e ".[dev,aws]"

us-market-ml-baseline-smoke \
  --s3-bucket us-market-data-casper \
  --s3-curated-prefix curated/etf_daily_prices \
  --symbols SPY,QQQ,TLT \
  --years 2023,2024 \
  --region-name us-west-2 \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --validation-start 2024-01-01 \
  --validation-end 2024-03-31 \
  --test-start 2024-04-01 \
  --test-end 2024-06-30 \
  --top-k 3
```

This command is a deterministic real-data smoke test for the current research
workflow. It reads ETF OHLCV data from one selected input source, builds
features and labels, fits the baseline model, generates top-K signals, and
runs the simple lagged signal backtest. It is not production trading
infrastructure and does not model transaction costs, slippage, or order
execution.

## MLP Smoke CLI

Install the optional neural network dependencies to run the MLP end-to-end
smoke workflow from local curated market data:

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-mlp-smoke \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --train-start 2020-01-01 \
  --train-end 2020-12-31 \
  --validation-start 2021-01-01 \
  --validation-end 2021-03-31 \
  --test-start 2021-04-01 \
  --test-end 2021-06-30 \
  --hidden-units 128,64 \
  --max-epochs 100 \
  --patience 10 \
  --top-k 3
```

S3 mode requires both AWS and neural network optional dependencies:

```bash
python -m pip install -e ".[dev,aws,nn]"

us-market-ml-mlp-smoke \
  --s3-bucket us-market-data-casper \
  --s3-curated-prefix curated/etf_daily_prices \
  --symbols SPY,QQQ,TLT \
  --years 2023,2024 \
  --region-name us-west-2 \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --validation-start 2024-01-01 \
  --validation-end 2024-03-31 \
  --test-start 2024-04-01 \
  --test-end 2024-06-30 \
  --hidden-units 128,64 \
  --top-k 3
```

The MLP smoke command reads one selected market data source, builds the
supervised feature and label dataset, fits the tabular PyTorch MLP, generates
top-K ETF rotation signals, and runs the same simple lagged signal backtest. It
does not write artifacts or model transaction costs, slippage, or order
execution. MLP target scaling is optional through `MLPRegressorConfig`; when
enabled, training uses train-only standardized targets while predictions and
fit metrics stay in original return units.

## Benchmark Report

The benchmark report CLI reruns one selected local MLP E2E configuration,
uses the test split equity curve as the strategy curve, and exports a static
comparison against SPY, QQQ, and an equal-weight universe benchmark. It reads
local CSV/parquet cache files only in this first version.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-benchmark-report \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/benchmark_reports \
  --report-name mlp_v1_fold_1 \
  --train-start 2020-01-01 \
  --train-end 2020-12-31 \
  --validation-start 2021-01-01 \
  --validation-end 2021-03-31 \
  --test-start 2021-04-01 \
  --test-end 2021-06-30 \
  --benchmark-symbols SPY,QQQ \
  --universe-symbols SPY,QQQ,TLT,IWM \
  --hidden-units 256,128 \
  --learning-rate 0.0005 \
  --dropout 0.2 \
  --scale-target \
  --top-k 3
```

The command writes `{report_name}_combined_equity.csv`,
`{report_name}_metrics.csv`, and `{report_name}_equity.png` under the selected
output directory. Benchmark curves are calculated over the same test-period
date range as the strategy backtest, so the PNG and CSVs show whether the
selected MLP strategy outperformed SPY, QQQ, and the equal-weight universe
during that held-out interval. It does not read S3, write model artifacts,
model transaction costs or slippage, or create interactive plots.

## Walk-forward Benchmark Report

The walk-forward benchmark report is the preferred long-period out-of-sample
visualization for MLP strategy research. It runs one MLP E2E experiment across
multiple explicit walk-forward folds, takes only each fold's test split
backtest, stitches those non-overlapping test periods chronologically, and
plots the stitched OOS strategy equity against SPY, QQQ, and equal-weight
benchmarks.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-walk-forward-benchmark-report \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/benchmark_reports \
  --report-name mlp_v1_walk_forward \
  --fold fold_1:2018-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
  --fold fold_2:2018-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
  --fold fold_3:2018-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
  --benchmark-symbols SPY,QQQ \
  --universe-symbols SPY,QQQ,TLT,IWM \
  --hidden-units 256,128 \
  --learning-rate 0.0005 \
  --dropout 0.2 \
  --scale-target \
  --top-k 3
```

The command writes `{report_name}_combined_equity.csv`,
`{report_name}_metrics.csv`, `{report_name}_equity.png`, and
`{report_name}_fold_summary.csv`. It also writes holdings and attribution CSVs:
`{report_name}_selected_holdings.csv`,
`{report_name}_position_contributions.csv`,
`{report_name}_holdings_summary.csv`,
`{report_name}_contribution_summary.csv`, and
`{report_name}_daily_selection_summary.csv`. The report also exports static
holdings visualization PNGs: `{report_name}_selection_frequency.png` for
selected days by symbol, `{report_name}_contribution_by_symbol.png` for total
position return by symbol, and `{report_name}_monthly_selection_heatmap.png`
for monthly selection patterns. These files show what the MLP strategy selected
during test folds and which lagged positions drove returns, helping identify
whether outperformance came from a few ETFs or from broad rotation across the
universe.

Training history such as 2018 can be used as feature warmup and model fitting
history, but it is not counted as plotted OOS performance unless it is inside a
test fold. In the example above, the plotted out-of-sample period starts on
2022-01-01, the first test fold start date. This first version reads local
cache files only and does not model transaction costs or slippage.

## Candidate Comparison Report

The candidate comparison report is for comparing selected MLP sweep winners
side by side. It reruns each named candidate across the same explicit
walk-forward folds, stitches only the test split equity curves, verifies that
candidate OOS dates align, and overlays all candidate strategy curves against
SPY, QQQ, and equal-weight benchmarks over the shared OOS date range. This is
local-only in v1 and does not write model artifacts, model transaction costs or
slippage, compare baseline models, or create interactive charts.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-candidate-comparison-report \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/benchmark_reports \
  --report-name mlp_v1_candidate_comparison \
  --fold fold_1:2018-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
  --fold fold_2:2018-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
  --fold fold_3:2018-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
  --candidate best_sharpe_top3:3:256,128:0.0005:0.2:42 \
  --candidate best_return_top3:3:128,64:0.001:0.1:43 \
  --candidate conservative_top5:5:128:0.0005:0.0:42 \
  --benchmark-symbols SPY,QQQ \
  --universe-symbols SPY,QQQ,TLT,IWM \
  --scale-target \
  --max-epochs 120 \
  --patience 12 \
  --batch-size 128
```

Each `--candidate` value uses
`name:top_k:hidden_units:learning_rate:dropout:random_state`. Candidate names
must be unique. The command writes `{report_name}_combined_equity.csv` with all
candidate and benchmark equity series, `{report_name}_metrics.csv` with metrics
for every plotted series, `{report_name}_candidate_rank.csv` with candidate-only
total return, Sharpe-like, drawdown, final equity, and metric ranks,
`{report_name}_fold_summary.csv` with per-candidate test-fold summaries, and
`{report_name}_equity.png` with the combined static equity chart.

## Structured Sweep Runner

The structured sweep runner replaces shell loops and log scraping for longer
baseline and MLP E2E comparisons. It reads local CSV/parquet cache files only,
loads each universe once per sweep process, writes structured CSV results, and
keeps completed/failed checkpoint text files so interrupted runs can resume.
It does not read S3 directly; create local cache files first when using S3
curated data.

Baseline-only example:

```bash
python -m pip install -e ".[dev]"

us-market-ml-sweep \
  --run-name baseline_grid_v1 \
  --output-dir artifacts/sweeps \
  --universe core:data/curated/etf_daily_ohlcv.parquet \
  --fold fold_1:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-06-30 \
  --model-family baseline \
  --baseline-models ridge,linear_regression \
  --top-k 1,2,3,5 \
  --label-horizon-days 20
```

MLP example:

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-sweep \
  --run-name mlp_grid_v1 \
  --output-dir artifacts/sweeps \
  --universe core:data/curated/etf_daily_ohlcv.parquet \
  --fold fold_1:2020-01-01:2020-12-31:2021-01-01:2021-03-31:2021-04-01:2021-06-30 \
  --model-family mlp \
  --hidden-units 128,64 \
  --learning-rates 0.001,0.0005 \
  --dropouts 0.0,0.1 \
  --seeds 42,43 \
  --max-epochs 100 \
  --patience 10 \
  --batch-size 64 \
  --scale-target \
  --top-k 1,2,3 \
  --label-horizon-days 20
```

For MLP sweeps, `--scale-target` standardizes the target using the training
split's target mean and standard deviation, then inverse-transforms predictions
back into original return units before evaluation and backtesting.

Each run writes `{run_name}_results.csv`,
`{run_name}_completed_keys.txt`, and `{run_name}_failed_keys.txt` under the
selected output directory. Resume is enabled by default; pass `--no-resume` to
rerun all requested configurations.

Purged walk-forward splits are optional but recommended when using
forward-looking labels. A target such as `excess_return_vs_spy_20d` uses future
prices, so rows near the end of a train or validation window can have labels
that cross into the next split. Passing `--label-horizon-days 20` removes those
calendar-day boundary rows from train and validation splits while leaving test
rows unchanged by default.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Optional neural network dependencies are kept out of the default install:

```bash
python -m pip install -e ".[dev,nn]"
```

## Expected Future Workflow

1. Load curated ETF daily parquet/CSV data from local research exports or curated S3 partitions.
2. Build leakage-safe historical feature tables.
3. Generate forward-return labels for one or more horizons.
4. Create walk-forward train, validation, and test splits.
5. Run supervised baseline regression experiments with explicit walk-forward folds.
6. Run the end-to-end baseline workflow to generate top-K ETF rotation signals
   and backtest lagged target weights.
7. Compare neural network candidates such as LSTM/GRU, 1D CNN, and TCN against
   the first tabular MLP.
8. Backtest signals against SPY, QQQ, and equal-weight ETF baselines.
9. Store reproducible artifacts under `artifacts/`.
