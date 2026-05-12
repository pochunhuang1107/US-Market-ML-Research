"""End-to-end MLP ETF rotation research experiment workflow.

This module composes the optional PyTorch MLP supervised runner, deterministic
top-K signal generation, and simple signal backtester into one research
workflow. It is not live trading infrastructure: MLP predictions become top-K
target weights, and those weights are backtested with one-row-per-symbol
lagged application to avoid same-day lookahead.

Transaction costs, slippage, artifact writing, live execution, and additional
neural architectures are intentionally future tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from us_market_ml.backtest import (
    SignalBacktestConfig,
    SignalBacktestResult,
    TopKSignalConfig,
    generate_top_k_signals,
    run_signal_backtest,
)
from us_market_ml.experiments.mlp_experiment import (
    MLPExperimentConfig,
    MLPExperimentResult,
    run_mlp_experiment,
)
from us_market_ml.splits import WalkForwardFold


@dataclass(frozen=True)
class MLPE2EExperimentConfig:
    """Configuration for an end-to-end MLP research experiment.

    This config wires together supervised MLP prediction, top-K equal-weight
    target generation, and lagged signal backtesting. It is for offline
    research only; transaction costs, slippage, artifact writing, and live
    trading concerns are intentionally outside this workflow.
    """

    mlp_config: MLPExperimentConfig = field(default_factory=MLPExperimentConfig)
    signal_config: TopKSignalConfig = field(default_factory=TopKSignalConfig)
    backtest_config: SignalBacktestConfig = field(default_factory=SignalBacktestConfig)


@dataclass(frozen=True)
class MLPE2EFoldResult:
    """Signals and backtests for one MLP walk-forward fold."""

    fold_name: str
    train_signals: pd.DataFrame
    validation_signals: pd.DataFrame
    test_signals: pd.DataFrame
    train_backtest: SignalBacktestResult
    validation_backtest: SignalBacktestResult
    test_backtest: SignalBacktestResult


@dataclass(frozen=True)
class MLPE2EExperimentResult:
    """Complete end-to-end MLP ETF rotation research output."""

    config: MLPE2EExperimentConfig
    mlp_result: MLPExperimentResult
    fold_results: list[MLPE2EFoldResult]


def run_mlp_e2e_experiment(
    raw_df: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: MLPE2EExperimentConfig | None = None,
) -> MLPE2EExperimentResult:
    """Run MLP prediction, top-K signal generation, and signal backtesting.

    The supervised MLP runner trains only on each fold's training rows and
    returns train, validation, and test prediction frames. Each prediction
    frame is converted into top-K target weights, then backtested with lagged
    weight application inherited from ``run_signal_backtest`` so a target
    generated on date T cannot affect the return already realized on date T.

    This is an end-to-end research experiment, not live trading. It does not
    write artifacts, read from S3/local files, or model transaction costs,
    slippage, order execution, or additional neural architectures.
    """

    resolved_config = config if config is not None else MLPE2EExperimentConfig()
    _validate_prediction_column_alignment(resolved_config)

    mlp_result = run_mlp_experiment(
        raw_df=raw_df,
        folds=folds,
        config=resolved_config.mlp_config,
    )

    fold_results: list[MLPE2EFoldResult] = []
    for mlp_fold_result in mlp_result.fold_results:
        train_signals = generate_top_k_signals(
            mlp_fold_result.train_predictions,
            resolved_config.signal_config,
        )
        validation_signals = generate_top_k_signals(
            mlp_fold_result.validation_predictions,
            resolved_config.signal_config,
        )
        test_signals = generate_top_k_signals(
            mlp_fold_result.test_predictions,
            resolved_config.signal_config,
        )

        fold_results.append(
            MLPE2EFoldResult(
                fold_name=mlp_fold_result.fold_name,
                train_signals=train_signals,
                validation_signals=validation_signals,
                test_signals=test_signals,
                train_backtest=run_signal_backtest(
                    train_signals,
                    resolved_config.backtest_config,
                ),
                validation_backtest=run_signal_backtest(
                    validation_signals,
                    resolved_config.backtest_config,
                ),
                test_backtest=run_signal_backtest(
                    test_signals,
                    resolved_config.backtest_config,
                ),
            )
        )

    return MLPE2EExperimentResult(
        config=resolved_config,
        mlp_result=mlp_result,
        fold_results=fold_results,
    )


def summarize_mlp_e2e_experiment(
    result: MLPE2EExperimentResult,
) -> pd.DataFrame:
    """Return one row per fold and split with signal backtest metrics.

    Metrics summarize the lagged-weight research backtest produced from top-K
    target weights. Transaction costs, when configured, are simple turnover
    bps approximations rather than execution simulation.
    """

    rows: list[dict[str, object]] = []
    for mlp_fold_result, e2e_fold_result in zip(
        result.mlp_result.fold_results,
        result.fold_results,
        strict=True,
    ):
        fit_result = mlp_fold_result.fit_result
        split_backtests = (
            ("train", e2e_fold_result.train_backtest),
            ("validation", e2e_fold_result.validation_backtest),
            ("test", e2e_fold_result.test_backtest),
        )
        for split, backtest in split_backtests:
            metrics = backtest.summary_metrics
            rows.append(
                {
                    "fold_name": e2e_fold_result.fold_name,
                    "split": split,
                    "model_type": "mlp",
                    "target_col": fit_result.target_col,
                    "selected_top_k": result.config.signal_config.top_k,
                    "epochs_trained": fit_result.epochs_trained,
                    "total_return": metrics["total_return"],
                    "mean_daily_return": metrics["mean_daily_return"],
                    "daily_volatility": metrics["daily_volatility"],
                    "sharpe_like": metrics["sharpe_like"],
                    "max_drawdown": metrics["max_drawdown"],
                    "final_equity": metrics["final_equity"],
                    "total_transaction_cost": metrics.get("total_transaction_cost"),
                    "mean_daily_turnover": metrics.get("mean_daily_turnover"),
                    "max_daily_turnover": metrics.get("max_daily_turnover"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "fold_name",
            "split",
            "model_type",
            "target_col",
            "selected_top_k",
            "epochs_trained",
            "total_return",
            "mean_daily_return",
            "daily_volatility",
            "sharpe_like",
            "max_drawdown",
            "final_equity",
            "total_transaction_cost",
            "mean_daily_turnover",
            "max_daily_turnover",
        ],
    )


def _validate_prediction_column_alignment(config: MLPE2EExperimentConfig) -> None:
    mlp_prediction_col = config.mlp_config.prediction_col
    signal_prediction_col = config.signal_config.prediction_col
    if mlp_prediction_col != signal_prediction_col:
        raise ValueError(
            "mlp_config.prediction_col must match signal_config.prediction_col; "
            f"got {mlp_prediction_col!r} and {signal_prediction_col!r}."
        )


__all__ = [
    "MLPE2EExperimentConfig",
    "MLPE2EExperimentResult",
    "MLPE2EFoldResult",
    "run_mlp_e2e_experiment",
    "summarize_mlp_e2e_experiment",
]
