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

## Not Included Yet

The following are intentionally not implemented in this task:

- S3 data loading.
- Neural network implementations.
- Portfolio backtesting engine.
- Transaction-cost, return, and equity-curve simulation.
- Live trading or brokerage integration.
- Large notebooks or exploratory artifacts.

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

1. Load curated ETF daily parquet data from the upstream data repository's S3 outputs.
2. Build leakage-safe historical feature tables.
3. Generate forward-return labels for one or more horizons.
4. Create walk-forward train, validation, and test splits.
5. Run supervised baseline regression experiments with explicit walk-forward folds.
6. Compare neural network candidates such as MLP, LSTM/GRU, 1D CNN, and TCN.
7. Generate top-K ETF rotation signals from prediction scores.
8. Backtest signals against SPY, QQQ, and equal-weight ETF baselines.
9. Store reproducible artifacts under `artifacts/`.
