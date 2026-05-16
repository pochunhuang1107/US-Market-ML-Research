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
- A local candidate comparison report that overlays selected MLP and
  baseline/tree candidate OOS curves against SPY, QQQ, and equal-weight
  benchmarks.
- Candidate deep-dive holding-period, position-switching, and turnover
  diagnostics for judging whether a shortlist strategy is realistically
  tradable or passive enough for further research.
- A local cost sensitivity report that reruns selected candidates over a grid
  of turnover-based transaction-cost assumptions.
- A local allocation study report that ranks completed allocation sweep
  artifacts and compares candidates against the current top-1 champion.
- A research-only paper-tracking export that writes daily target holdings,
  switch logs, and tracking-window metrics for one selected local candidate.
- A model signal artifact export that writes point-in-time runtime-compatible
  signal files for the downstream simulation platform.
- A local daily model signal export, healthcheck CLI, and operator runbook for
  validating runtime-safe signal artifacts before platform ingestion.

## Not Included Yet

The following are intentionally not implemented in this task:

- S3 writing or artifact publishing.
- Sequence, convolutional, transformer, registry, and artifact-writing model
  infrastructure beyond the first tabular MLP.
- Spread, slippage, tax, and order-execution simulation.
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

## Feature Sets

Feature engineering is versioned so sweeps can compare research datasets while
keeping the default baseline unchanged:

- `default_features_v1`: original/current leakage-safe OHLCV feature set.
- `technical_features_v2`: expanded OHLCV-only technical feature set with
  additional momentum, volatility, trend, liquidity, cross-sectional rank, and
  benchmark-relative features.

No external data is required for either feature set. Feature engineering remains
research-only and is still subject to walk-forward validation before any
candidate is promoted.

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
infrastructure or a full execution simulator.

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
does not write artifacts or simulate order execution. MLP target scaling is
optional through `MLPRegressorConfig`; when enabled, training uses train-only
standardized targets while predictions and fit metrics stay in original return
units.

Signal backtests support an optional fixed transaction-cost assumption through
`transaction_cost_bps`, which defaults to `0.0`. Daily turnover is calculated
from target weight changes by date and symbol, with the first date set to zero;
cost is `turnover × bps / 10000`. Strategy equity and summary metrics use net
`portfolio_return`, while `gross_portfolio_return`, `transaction_cost`, and
`turnover` are also written on strategy equity curves. This is a first-order
turnover-based approximation, not a bid/ask spread, slippage, intraday fill,
tax, or full execution simulator.

Signal generation can optionally apply research-only smoothing before the
lagged signal backtest. `min_holding_days` keeps selected ETFs for at least a
minimum number of strategy dates, which can reduce one-day holding churn.
`switch_margin` requires the best replacement candidate's prediction to clear
the current holding by a stronger score margin before switching. This smoothing
logic is portfolio-construction research logic only; it is not an execution
model and does not change transaction-cost math, fills, slippage, or model
training.

### Portfolio allocation modes

Portfolio allocation research asks whether model scores can keep a meaningful
share of the return while reducing concentration, drawdown, and turnover versus
a concentrated top-1 ETF rotation. These methods are offline
portfolio-construction experiments only; they are not execution logic, trading
advice, or live allocation instructions.

`equal_weight` is the default and preserves the previous behavior: selected
top-K symbols receive equal target weights.

`score_weighted` weights selected symbols in proportion to positive prediction
scores, shifting negative selected scores before normalization.

`softmax_score_weighted` applies a numerically stabilized softmax to selected
scores using `score_temperature`.

`volatility_adjusted_score_weighted` weights positive shifted scores by recent
volatility, using `volatility_20d` when present and falling back safely to
score-weighted allocation when volatility is unavailable or invalid.

`capped_score_weighted` starts from score-weighted allocation, caps each symbol
at `max_weight`, and redistributes excess across uncapped selected symbols.

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
  --top-k 3 \
  --transaction-cost-bps 5
```

The command writes `{report_name}_combined_equity.csv`,
`{report_name}_metrics.csv`, and `{report_name}_equity.png` under the selected
output directory. Benchmark curves are calculated over the same test-period
date range as the strategy backtest, so the PNG and CSVs show whether the
selected MLP strategy outperformed SPY, QQQ, and the equal-weight universe
during that held-out interval. It does not read S3, write model artifacts,
simulate advanced execution or slippage, or create interactive plots.

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
  --top-k 3 \
  --transaction-cost-bps 5
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
cache files only and uses only the optional fixed-bps turnover cost model, not
full execution or slippage simulation.

## Candidate Comparison Report

The candidate comparison report is for comparing selected MLP and
baseline/tree sweep winners side by side. It reruns each named candidate across
the same explicit walk-forward folds, stitches only the test split equity
curves, verifies that candidate OOS dates align, and overlays all candidate
strategy curves against SPY, QQQ, and equal-weight benchmarks over the shared
OOS date range. This is local-only in v1 and does not write model artifacts,
simulate advanced execution or slippage, or create interactive charts.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-candidate-comparison-report \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/benchmark_reports \
  --report-name mlp_v1_candidate_comparison \
  --fold fold_1:2018-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
  --fold fold_2:2018-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
  --fold fold_3:2018-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
  --candidate baseline:linear_unsmoothed_top1:linear_regression:1:42 \
  --candidate baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002 \
  --candidate mlp:mlp_broad_top5_safe:5:256,128:0.001:0.3:42 \
  --benchmark-symbols SPY,QQQ \
  --universe-symbols SPY,QQQ,TLT,IWM \
  --scale-target \
  --max-epochs 120 \
  --patience 12 \
  --batch-size 128 \
  --transaction-cost-bps 5
```

MLP candidates use
`mlp:name:top_k:hidden_units:learning_rate:dropout:random_state`. Baseline and
tree candidates use `baseline:name:model_type:top_k:random_state`, where
`model_type` can be `linear_regression`, `ridge`, `random_forest`,
`gradient_boosting`, or `extra_trees`. The legacy MLP-only format
`name:top_k:hidden_units:learning_rate:dropout:random_state` is still accepted
for backward compatibility.

Candidate specs can optionally append per-candidate smoothing settings:
`baseline:name:model_type:top_k:random_state:min_holding_days:switch_margin`
or
`mlp:name:top_k:hidden_units:learning_rate:dropout:random_state:min_holding_days:switch_margin`.
When present, these override the global `--min-holding-days` and
`--switch-margin` for that candidate only. This lets one report compare an
unsmoothed candidate such as
`baseline:linear_unsmoothed_top1:linear_regression:1:42` against a smoothed
version such as
`baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002`.
Candidate names must be unique. The command writes
`{report_name}_combined_equity.csv` with all candidate and benchmark equity
series, `{report_name}_metrics.csv` with metrics for every plotted series,
`{report_name}_candidate_rank.csv` with candidate family, model type, total
return, Sharpe-like, drawdown, final equity, optional cost/turnover metrics,
effective smoothing settings, and metric ranks,
`{report_name}_fold_summary.csv` with per-candidate test-fold summaries and
effective smoothing settings, and
`{report_name}_equity.png` with the combined static equity chart.

## Candidate Deep-dive Report

After a sweep and candidate comparison identify a shortlist, the candidate
deep-dive report reruns those selected MLP and baseline/tree candidates and
writes a full diagnostic pack for each one. It uses only stitched OOS test
curves for performance comparison, verifies candidate test dates align,
computes benchmarks over the shared OOS date range, and adds holdings,
contribution, daily selection, and static visualization outputs for every
candidate. This is local-only in v1 and does not write model artifacts,
simulate advanced execution or slippage, or create a web UI.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-candidate-deep-dive-report \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/candidate_deep_dives \
  --report-name mixed_shortlist_deep_dive \
  --fold fold_1:2018-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
  --fold fold_2:2018-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
  --fold fold_3:2018-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
  --candidate baseline:linear_unsmoothed_top1:linear_regression:1:42 \
  --candidate baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002 \
  --candidate mlp:mlp_broad_top5_safe:5:256,128:0.001:0.3:42 \
  --benchmark-symbols SPY,QQQ \
  --universe-symbols SPY,QQQ,TLT,IWM \
  --scale-target \
  --max-epochs 120 \
  --patience 12 \
  --batch-size 128 \
  --transaction-cost-bps 5
```

MLP candidates use
`mlp:name:top_k:hidden_units:learning_rate:dropout:random_state`. Baseline and
tree candidates use `baseline:name:model_type:top_k:random_state`, with the
same baseline model types as the comparison report. The legacy MLP-only format
is still accepted for backward compatibility. The same optional per-candidate
smoothing suffixes from the comparison report are supported, so a deep-dive can
include both
`baseline:linear_unsmoothed_top1:linear_regression:1:42` and
`baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002`.

The top-level output directory receives `{report_name}_candidate_equity.csv`,
`{report_name}_candidate_metrics.csv`, `{report_name}_candidate_rank.csv`,
`{report_name}_concentration_summary.csv`,
`{report_name}_holding_period_summary.csv`, `{report_name}_fold_summary.csv`,
and `{report_name}_candidate_equity.png` for candidate-only comparison. Each
candidate also gets its own subdirectory with combined candidate-and-benchmark
equity, metrics, fold summary, selected holdings, position contributions,
holdings summary, contribution summary, concentration report, daily selection
summary, holding periods, holding-period summary, holding-period by-symbol
summary, turnover summary, top turnover days, equity PNG, selection frequency
PNG, contribution by symbol PNG, and monthly selection heatmap PNG.

The concentration diagnostics quantify whether a high-return candidate is
fragile because it is dominated by one symbol or a small set of ETFs. Selection
metrics report top-symbol selection share, top-three selection share, HHI,
effective selected symbols, and maximum average target weight. Contribution
metrics report how much of positive return came from the top symbol and top
three contributors, plus HHI and effective contributing symbols using absolute
position returns. Warning flags are raised when selection or contribution is
especially concentrated, which helps identify cases where strong OOS return may
mostly reflect exposure to one ETF rather than broad rotation skill.

Concentration reports also include `selection_concentration_risk`,
`contribution_concentration_risk`, `overall_concentration_risk`, and
`risk_score`. Risk labels are `low`, `medium`, or `high`; `risk_score` maps
those labels to 1, 2, and 3 using the higher of the selection and contribution
risks. These grades are heuristic research-triage signals only. They are meant
to make candidate review easier before deeper analysis or possible future
real-money paper tracking, not to certify that a strategy is safe to trade.

Holding-period diagnostics measure how long selected symbols stay in the
portfolio before disappearing from the strategy's selected-date sequence. They
write per-candidate CSVs for every holding streak, one-row holding summaries,
symbol-level holding summaries, turnover distribution summaries, and the
highest-turnover OOS dates. Consecutive holding days use the strategy's
available dates, so weekends, holidays, and missing calendar days do not break a
streak by themselves.

Short holding periods and frequent position switching are important real-money
feasibility warnings. A candidate that looks strong before costs may still be
hard to implement passively if it often replaces positions after one or two
strategy dates. High average turnover or clustered high-turnover days can make
fixed-bps cost assumptions dominate performance and can also point to practical
issues that this research backtester does not model, including spread,
slippage, tax, and execution timing.

## Paper-tracking Export

The paper-tracking export reruns one selected candidate over a single local
tracking window and writes daily research CSVs showing model predictions,
selected symbols, target weights, switch days, and backtest metrics. This
export is for paper tracking and research validation only, not live trading
advice. It does not connect to a broker, place orders, schedule jobs, ingest
live data, or model execution beyond the existing lagged close-to-close
research backtest.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-paper-tracking-export \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/paper_tracking \
  --export-name linear_smooth10_margin002_tracking \
  --candidate baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002 \
  --train-start 2018-01-01 \
  --train-end 2022-12-31 \
  --validation-start 2023-01-01 \
  --validation-end 2023-12-31 \
  --tracking-start 2024-01-01 \
  --tracking-end 2024-12-31 \
  --benchmark-symbols SPY,QQQ \
  --transaction-cost-bps 10
```

The command writes `{export_name}_daily_targets.csv`,
`{export_name}_daily_summary.csv`, `{export_name}_switch_log.csv`, and
`{export_name}_metrics.csv`. Candidate specs use the same unified formats as
the comparison and deep-dive reports, including optional per-candidate
`min_holding_days:switch_margin` smoothing suffixes.

## Model Signal Artifact Export

The model signal artifact export is the bridge from this ML research/training
repo to the downstream `US-Market-Simulation-Platform`. It exports
point-in-time model signal rows that the platform can validate, load, and use
as an input signal layer while the platform continues to execute deterministic
Strategy V2 configs. The platform should not train models or run live
inference.

The export writes runtime signal data only: `date`, `symbol`, `model_version`,
`score`, and optional metadata such as `predicted_return`, `rank`,
`signal_bucket`, `feature_set_version`, `label_horizon_days`, and
`created_at`. It does not export training labels, target columns, future
returns, or realized returns.

Preferred layout:

```text
curated/model_signals/model_version=<VERSION>/signals.parquet
```

Example for the current paper-tracking candidate:

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-model-signal-export \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-path curated/model_signals/model_version=linear_smooth10_margin002_v1/signals.parquet \
  --candidate baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002 \
  --model-version linear_smooth10_margin002_v1 \
  --feature-set-version default_features_v1 \
  --train-start 2018-01-01 \
  --train-end 2022-12-31 \
  --validation-start 2023-01-01 \
  --validation-end 2023-12-31 \
  --export-start 2024-01-01 \
  --export-end 2024-12-31 \
  --label-horizon-days 20
```

For smoothed candidates, the artifact is built from post-smoothing
`test_signals`, so `linear_smooth10_margin002_v1` exports the actual
point-in-time strategy signal. In v1, `score` and `predicted_return` are both
the model `prediction`, `rank` is `signal_rank`, and `signal_bucket` is `top`
for selected rows and `other` for non-selected rows. This is research-only
point-in-time signal data, not investment advice or an order/execution system.

The simulation platform can validate an exported artifact with:

```bash
python scripts/validate_model_signal_export.py \
  --path <signals.parquet> \
  --model-version <MODEL_VERSION>
```

## Candidate Model Card

The candidate model card turns a candidate deep-dive output folder into a
single Markdown research note. It summarizes configuration, OOS performance,
SPY comparison, fold behavior, optional transaction-cost sensitivity, top
holdings, top contributors, concentration diagnostics, holding-period
diagnostics, strengths, weaknesses, and a deterministic research-triage verdict.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-model-card \
  --candidate-dir artifacts/candidate_deep_dives/linear_smooth10_margin002 \
  --candidate-name linear_smooth10_margin002 \
  --candidate-family baseline \
  --model-type linear_regression \
  --output-path artifacts/model_cards/linear_smooth10_margin002_model_card.md \
  --config-json '{"top_k": 1, "random_state": 42, "min_holding_days": 10, "switch_margin": 0.002}' \
  --cost-sensitivity-path artifacts/cost_sensitivity/mixed_shortlist_cost_sensitivity.csv
```

The command reads the per-candidate CSVs produced by the deep-dive report:
`{candidate_name}_metrics.csv`, `{candidate_name}_fold_summary.csv`,
`{candidate_name}_concentration_report.csv`,
`{candidate_name}_holding_period_summary.csv`,
`{candidate_name}_holdings_summary.csv`, and
`{candidate_name}_contribution_summary.csv`. If `--cost-sensitivity-path` is
provided, matching rows from the optional cost sensitivity CSV are included as a
Markdown table with tested bps levels, total return, Sharpe-like, drawdown,
final equity, transaction cost, and turnover when those columns are available.
The card also adds a deterministic cost robustness summary, including the
highest tested cost with positive total return, the first non-positive tested
cost when observed, selected 10/20/50 bps returns when present, and a
high/medium/low/fragile label based on the tested grid. The model card does not
rerun training or backtests, does not use S3, and does not create PDFs.

The verdict is a research-triage label only:
`promising_for_further_research`, `needs_more_validation`, or
`not_currently_promising`. It is based on simple SPY-relative performance,
drawdown, and concentration checks. Model cards are documentation for research
review and possible paper-trading preparation, not financial or trading advice.

## Cost Sensitivity Report

The cost sensitivity report reruns selected MLP and baseline/tree candidates
over multiple `transaction_cost_bps` assumptions, stitches the out-of-sample
test equity curves, and summarizes how candidate performance degrades as costs
rise. It focuses on candidate strategies only; SPY/QQQ and equal-weight
benchmarks are not rerun or included in the v1 sensitivity chart.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-cost-sensitivity-report \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/cost_sensitivity \
  --report-name mixed_shortlist_cost_sensitivity \
  --fold fold_1:2018-01-01:2020-12-31:2021-01-01:2021-12-31:2022-01-01:2022-12-31 \
  --fold fold_2:2018-01-01:2021-12-31:2022-01-01:2022-12-31:2023-01-01:2023-12-31 \
  --fold fold_3:2018-01-01:2022-12-31:2023-01-01:2023-12-31:2024-01-01:2024-12-31 \
  --candidate baseline:linear_full_top1:linear_regression:1:42 \
  --candidate mlp:mlp_broad_top5_safe:5:256,128:0.001:0.3:42 \
  --cost-bps 0,1,2,5,10,20,50 \
  --scale-target \
  --max-epochs 120 \
  --patience 12 \
  --batch-size 128
```

The command writes `{report_name}_cost_sensitivity.csv`,
`{report_name}_cost_sensitivity_pivot.csv`, and
`{report_name}_cost_sensitivity.png`. The long CSV includes return, Sharpe-like,
drawdown, final equity, total transaction cost, and turnover metrics for every
candidate/cost pair. When `0` bps is in the grid, it also reports return and
final-equity decay versus the no-cost baseline. The pivot CSV creates dynamic
columns such as `total_return_0bps`, `total_return_10bps`,
`final_equity_10bps`, and `sharpe_like_10bps`.

The cost model is the same first-order approximation used by the signal
backtester:

```text
net return = gross return - turnover x bps / 10000
```

This is a research approximation, not an execution simulation. It does not
model bid/ask spreads, intraday fills, market impact, taxes, or broker-specific
fees.

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
  --baseline-models ridge,linear_regression,gradient_boosting,extra_trees,random_forest \
  --top-k 1,2,3,5 \
  --min-holding-days 1,3,5 \
  --switch-margins 0,0.005,0.01 \
  --transaction-cost-bps 0,5,10 \
  --label-horizon-days 20
```

Baseline model types currently supported by the shared supervised baseline
factory are `ridge`, `linear_regression`, `random_forest`,
`gradient_boosting`, and `extra_trees`.

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
  --min-holding-days 1,3 \
  --switch-margins 0,0.005 \
  --transaction-cost-bps 0,5,10 \
  --label-horizon-days 20
```

For MLP sweeps, `--scale-target` standardizes the target using the training
split's target mean and standard deviation, then inverse-transforms predictions
back into original return units before evaluation and backtesting.

Sweep smoothing arguments expand the signal grid across every
`top_k × min_holding_days × switch_margin` combination. Defaults are
`--min-holding-days 1` and `--switch-margins 0`, which preserve the original
unsmoothed top-K behavior.

Sweep allocation arguments add `--allocation-modes`, `--score-temperatures`,
and `--max-weights` grids. Defaults are `equal_weight`, `1.0`, and `1.0`,
which preserve the previous equal-weight top-K behavior. In v1, temperature is
only used by `softmax_score_weighted`, and max weight is only used by
`capped_score_weighted`, though the values are still recorded in run keys and
result rows for deterministic comparison. Capped allocation sweep points where
`top_k × max_weight < 1` are skipped because they cannot produce a fully
allocated portfolio.

Transaction-cost sweeps use `--transaction-cost-bps`, also as a comma-separated
grid. The default is `0`, and nonzero values are passed through to the existing
turnover-based signal backtester.

Each run writes `{run_name}_results.csv`,
`{run_name}_completed_keys.txt`, and `{run_name}_failed_keys.txt` under the
selected output directory. Resume is enabled by default; pass `--no-resume` to
rerun all requested configurations.

## Allocation Study Report

After the focused allocation sweeps complete, the allocation study report reads
their existing result CSVs and checkpoint files, validates that no failed keys
were recorded, aggregates `split == "test"` rows across folds, and compares
allocation candidates against the current top-1 champion. It does not rerun
sweeps or redesign the experiment runner.

```bash
us-market-ml-allocation-study-report \
  --sweep-dir artifacts/sweeps \
  --sweep allocation_core_modes_study \
  --sweep allocation_capped_study \
  --sweep allocation_softmax_study \
  --output-dir artifacts/reports/allocation_study_comparison
```

The command writes `allocation_candidate_metrics.csv`,
`allocation_top_candidates.csv`, `allocation_champion_comparison.csv`,
`allocation_champion_family_comparison.csv`, and `allocation_study_report.md`.
The markdown report includes sweep completion status, champion metrics,
top-return and balanced rankings, a champion-family allocation comparison,
drawdown tradeoffs, and a research-stage recommendation.

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

## Daily Model Signal Export

The daily model signal export is the local paper-tracking artifact path for
platform ingestion. It reruns one selected research candidate through the
existing feature, model, ranking, and smoothing workflow, then writes a compact
runtime-safe signal artifact plus manifest for the latest available signal
date. This remains research-only output: it is not live trading advice, not a
broker integration, not scheduling, and not an order/execution system.

```bash
python -m pip install -e ".[dev,nn]"

us-market-ml-daily-signal-export \
  --data-path data/curated/etf_daily_ohlcv.parquet \
  --output-dir artifacts/model_signals_daily \
  --candidate baseline:linear_smooth10_margin002:linear_regression:1:42:10:0.002 \
  --model-version linear_smooth10_margin002_v1 \
  --feature-set-version default_features_v1 \
  --train-start 2018-01-01 \
  --train-end 2022-12-31 \
  --validation-start 2023-01-01 \
  --validation-end 2023-12-31 \
  --label-horizon-days 20 \
  --format parquet \
  --merge-existing
```

The command writes:

```text
artifacts/model_signals_daily/model_version=<VERSION>/signals.parquet
artifacts/model_signals_daily/model_version=<VERSION>/model_signal_manifest.json
```

Run the local healthcheck before passing the artifact to the simulation
platform:

```bash
us-market-ml-daily-signal-healthcheck \
  --artifact-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/signals.parquet \
  --manifest-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/model_signal_manifest.json \
  --model-version linear_smooth10_margin002_v1 \
  --min-symbol-count 2 \
  --max-signal-age-days 3 \
  --output-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/healthcheck.csv \
  --warnings-output-path artifacts/model_signals_daily/model_version=linear_smooth10_margin002_v1/healthcheck_warnings.txt
```

The healthcheck validates required runtime columns, model version consistency,
numeric scores, duplicate date/symbol/model rows, forbidden lookahead columns,
manifest consistency, and optional freshness via `--max-signal-age-days`.
Freshness compares `latest_signal_date` with the current UTC date by default,
or with `--reference-date` when provided. It returns `0` for `ready` or
`warning`, and `1` for `failed`; pass `--stale-as-failure` to make a stale
latest signal date fail the check. The daily operator checklist lives in
[`docs/DAILY_MODEL_SIGNAL_RUNBOOK.md`](docs/DAILY_MODEL_SIGNAL_RUNBOOK.md).
