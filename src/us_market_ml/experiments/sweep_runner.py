"""Structured resumable sweeps for E2E ETF rotation experiments.

This module orchestrates local-cache-only baseline and MLP E2E runs across
universes, walk-forward folds, model configs, and top-K signal configs. It
loads each local data source once per universe, writes structured CSV rows
after each successful run, and maintains simple text checkpoints for resume.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from us_market_ml.backtest import SignalBacktestConfig, SignalSmoothingConfig, TopKSignalConfig
from us_market_ml.data import (
    DEFAULT_FEATURE_SET_VERSION,
    read_local_market_data,
    validate_feature_set_version,
)
from us_market_ml.experiments.baseline_e2e import (
    BaselineE2EExperimentConfig,
    run_baseline_e2e_experiment,
    summarize_baseline_e2e_experiment,
)
from us_market_ml.experiments.baseline_experiment import BaselineExperimentConfig
from us_market_ml.experiments.mlp_e2e import (
    MLPE2EExperimentConfig,
    run_mlp_e2e_experiment,
    summarize_mlp_e2e_experiment,
)
from us_market_ml.experiments.mlp_experiment import MLPExperimentConfig
from us_market_ml.models import BaselineModelConfig, MLPRegressorConfig
from us_market_ml.splits import PurgeConfig, WalkForwardFold

RESULT_COLUMNS = [
    "run_key",
    "universe",
    "data_path",
    "fold",
    "split",
    "model_family",
    "model_type",
    "feature_set_version",
    "feature_count",
    "hidden_units",
    "dropout",
    "scale_target",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "random_state",
    "top_k",
    "min_holding_days",
    "switch_margin",
    "allocation_mode",
    "score_temperature",
    "max_weight",
    "transaction_cost_bps",
    "target_col",
    "label_horizon_days",
    "total_return",
    "mean_daily_return",
    "daily_volatility",
    "sharpe_like",
    "max_drawdown",
    "final_equity",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
    "epochs_trained",
]


@dataclass(frozen=True)
class SweepUniverseConfig:
    """One named local market-data universe for a sweep."""

    name: str
    data_path: Path | str


@dataclass(frozen=True)
class SweepFoldConfig:
    """One named walk-forward fold for a sweep."""

    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class SweepModelConfig:
    """Baseline or MLP model configuration for a sweep."""

    model_family: str
    model_type: str = "ridge"
    hidden_units: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    random_state: int = 42
    device: str = "cpu"
    scale_target: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_units", tuple(self.hidden_units))
        if self.model_family not in {"baseline", "mlp"}:
            raise ValueError("model_family must be 'baseline' or 'mlp'.")


@dataclass(frozen=True)
class SweepSignalConfig:
    """Top-K signal configuration for a sweep."""

    top_k: int
    min_holding_days: int | None = None
    switch_margin: float | None = None
    allocation_mode: str = "equal_weight"
    score_temperature: float = 1.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        TopKSignalConfig(
            top_k=self.top_k,
            allocation_mode=self.allocation_mode,
            score_temperature=self.score_temperature,
            max_weight=self.max_weight,
        )
        if self.min_holding_days is not None and self.min_holding_days < 1:
            raise ValueError("min_holding_days must be greater than or equal to 1.")
        if self.switch_margin is not None and self.switch_margin < 0.0:
            raise ValueError("switch_margin must be greater than or equal to 0.")


@dataclass(frozen=True)
class SweepRunConfig:
    """Complete structured sweep configuration."""

    universes: list[SweepUniverseConfig]
    folds: list[SweepFoldConfig]
    models: list[SweepModelConfig]
    signals: list[SweepSignalConfig]
    target_col: str = "excess_return_vs_spy_20d"
    initial_equity: float = 1.0
    prediction_col: str = "prediction"
    label_horizon_days: int = 0
    feature_set_versions: list[str] = field(
        default_factory=lambda: [DEFAULT_FEATURE_SET_VERSION]
    )
    min_holding_days: int = 1
    switch_margin: float = 0.0
    transaction_cost_bps_values: list[float] = field(default_factory=lambda: [0.0])
    output_dir: Path | str = "artifacts/sweeps"
    run_name: str = "sweep"
    resume: bool = True
    max_workers: int = 1


@dataclass(frozen=True)
class _SweepTask:
    universe: SweepUniverseConfig
    fold_config: SweepFoldConfig
    model_config: SweepModelConfig
    signal_config: SweepSignalConfig
    run_key: str
    transaction_cost_bps: float
    feature_set_version: str


def build_sweep_run_key(
    universe: SweepUniverseConfig,
    fold: SweepFoldConfig,
    model: SweepModelConfig,
    signal: SweepSignalConfig,
    label_horizon_days: int = 0,
    min_holding_days: int | None = None,
    switch_margin: float | None = None,
    transaction_cost_bps: float = 0.0,
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
) -> str:
    """Return a deterministic key for one universe/fold/model/signal run."""

    effective_min_holding_days = _effective_min_holding_days(signal, min_holding_days)
    effective_switch_margin = _effective_switch_margin(signal, switch_margin)
    signal_payload = _jsonable_dataclass(signal)
    signal_payload["min_holding_days"] = effective_min_holding_days
    signal_payload["switch_margin"] = effective_switch_margin
    payload = {
        "universe": _jsonable_dataclass(universe),
        "fold": _jsonable_dataclass(fold),
        "model": _jsonable_dataclass(model),
        "signal": signal_payload,
        "label_horizon_days": label_horizon_days,
        "min_holding_days": effective_min_holding_days,
        "switch_margin": effective_switch_margin,
        "transaction_cost_bps": transaction_cost_bps,
        "feature_set_version": feature_set_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    if model.model_family == "baseline":
        model_part = f"baseline-{model.model_type}"
    else:
        hidden_units = "x".join(str(unit) for unit in model.hidden_units)
        model_part = (
            f"mlp-hu-{hidden_units}-lr-{model.learning_rate:g}-"
            f"dropout-{model.dropout:g}-"
            f"scaletarget-{str(model.scale_target).lower()}"
        )

    return "__".join(
        [
            f"universe-{_slug(universe.name)}",
            f"fold-{_slug(fold.name)}",
            model_part,
            f"topk-{signal.top_k}",
            f"hold-{effective_min_holding_days}d",
            f"margin-{effective_switch_margin:g}",
            f"alloc-{_slug(signal.allocation_mode)}",
            f"temp-{signal.score_temperature:g}",
            f"maxw-{signal.max_weight:g}",
            f"cost-{transaction_cost_bps:g}bps",
            f"horizon-{label_horizon_days}d",
            f"features-{_slug(feature_set_version)}",
            f"seed-{model.random_state}",
            digest,
        ]
    )


def run_sweep(config: SweepRunConfig) -> pd.DataFrame:
    """Run a local-cache-only structured sweep and return all result rows."""

    _validate_sweep_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{config.run_name}_results.csv"
    completed_path = output_dir / f"{config.run_name}_completed_keys.txt"
    failed_path = output_dir / f"{config.run_name}_failed_keys.txt"

    if not config.resume:
        for path in (results_path, completed_path, failed_path):
            path.unlink(missing_ok=True)

    _ensure_results_file(results_path)
    completed_path.touch(exist_ok=True)
    failed_path.touch(exist_ok=True)

    completed_keys = _read_checkpoint_keys(completed_path) if config.resume else set()
    results_frames: list[pd.DataFrame] = []
    if config.resume and results_path.exists():
        existing_results = _read_results(results_path)
        if not existing_results.empty:
            results_frames.append(existing_results)

    pending_tasks = _build_pending_tasks(config, completed_keys)
    if config.max_workers > 1:
        return _run_sweep_parallel(
            config=config,
            tasks=pending_tasks,
            results_path=results_path,
            completed_path=completed_path,
            failed_path=failed_path,
            completed_keys=completed_keys,
            results_frames=results_frames,
        )

    for universe in config.universes:
        raw_df = read_local_market_data(universe.data_path)
        for fold_config in config.folds:
            fold = _to_walk_forward_fold(fold_config)
            for model_config in config.models:
                for signal_config in config.signals:
                    for feature_set_version in config.feature_set_versions:
                        for transaction_cost_bps in config.transaction_cost_bps_values:
                            run_key = build_sweep_run_key(
                                universe=universe,
                                fold=fold_config,
                                model=model_config,
                                signal=signal_config,
                                label_horizon_days=config.label_horizon_days,
                                min_holding_days=config.min_holding_days,
                                switch_margin=config.switch_margin,
                                transaction_cost_bps=transaction_cost_bps,
                                feature_set_version=feature_set_version,
                            )
                            if config.resume and run_key in completed_keys:
                                continue

                            try:
                                run_rows = _run_one_configuration(
                                    raw_df=raw_df,
                                    universe=universe,
                                    fold=fold,
                                    model=model_config,
                                    signal=signal_config,
                                    config=config,
                                    run_key=run_key,
                                    transaction_cost_bps=transaction_cost_bps,
                                    feature_set_version=feature_set_version,
                                )
                            except Exception as exc:  # noqa: BLE001 - continue remaining sweep runs.
                                _append_failed_key(failed_path, run_key, exc)
                                continue

                            run_results = pd.DataFrame(run_rows, columns=RESULT_COLUMNS)
                            _append_results(results_path, run_results)
                            _append_line(completed_path, run_key)
                            completed_keys.add(run_key)
                            results_frames.append(run_results)

    if not results_frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    return pd.concat(results_frames, ignore_index=True)[RESULT_COLUMNS]


def _build_pending_tasks(
    config: SweepRunConfig,
    completed_keys: set[str],
) -> list[_SweepTask]:
    tasks: list[_SweepTask] = []
    for universe in config.universes:
        for fold_config in config.folds:
            for model_config in config.models:
                for signal_config in config.signals:
                    for feature_set_version in config.feature_set_versions:
                        for transaction_cost_bps in config.transaction_cost_bps_values:
                            run_key = build_sweep_run_key(
                                universe=universe,
                                fold=fold_config,
                                model=model_config,
                                signal=signal_config,
                                label_horizon_days=config.label_horizon_days,
                                min_holding_days=config.min_holding_days,
                                switch_margin=config.switch_margin,
                                transaction_cost_bps=transaction_cost_bps,
                                feature_set_version=feature_set_version,
                            )
                            if config.resume and run_key in completed_keys:
                                continue
                            tasks.append(
                                _SweepTask(
                                    universe=universe,
                                    fold_config=fold_config,
                                    model_config=model_config,
                                    signal_config=signal_config,
                                    run_key=run_key,
                                    transaction_cost_bps=transaction_cost_bps,
                                    feature_set_version=feature_set_version,
                                )
                            )
    return tasks


def _run_sweep_parallel(
    *,
    config: SweepRunConfig,
    tasks: list[_SweepTask],
    results_path: Path,
    completed_path: Path,
    failed_path: Path,
    completed_keys: set[str],
    results_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if tasks:
        executor = ProcessPoolExecutor(
            max_workers=config.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures: dict[Future[list[dict[str, object]]], _SweepTask] = {}
        try:
            futures = {
                executor.submit(_run_one_task_worker, task, config): task for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    run_rows = future.result()
                except Exception as exc:  # noqa: BLE001 - continue remaining sweep runs.
                    _append_failed_key(failed_path, task.run_key, exc)
                    continue

                run_results = pd.DataFrame(run_rows, columns=RESULT_COLUMNS)
                _append_results(results_path, run_results)
                _append_line(completed_path, task.run_key)
                completed_keys.add(task.run_key)
                results_frames.append(run_results)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            _terminate_executor_processes(executor)
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    if not results_frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    return pd.concat(results_frames, ignore_index=True)[RESULT_COLUMNS]


_WORKER_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _run_one_task_worker(
    task: _SweepTask,
    config: SweepRunConfig,
) -> list[dict[str, object]]:
    data_path = str(task.universe.data_path)
    raw_df = _WORKER_DATA_CACHE.get(data_path)
    if raw_df is None:
        raw_df = read_local_market_data(task.universe.data_path)
        _WORKER_DATA_CACHE[data_path] = raw_df

    return _run_one_configuration(
        raw_df=raw_df,
        universe=task.universe,
        fold=_to_walk_forward_fold(task.fold_config),
        model=task.model_config,
        signal=task.signal_config,
        config=config,
        run_key=task.run_key,
        transaction_cost_bps=task.transaction_cost_bps,
        feature_set_version=task.feature_set_version,
    )


def _terminate_executor_processes(executor: ProcessPoolExecutor) -> None:
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for process in processes.values():
        if process.is_alive():
            process.terminate()


def summarize_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize test-split sweep rows by universe/model/top-K settings."""

    summary_columns = [
        "universe",
        "model_family",
        "model_type",
        "feature_set_version",
        "feature_count",
        "top_k",
        "min_holding_days",
        "switch_margin",
        "allocation_mode",
        "score_temperature",
        "max_weight",
        "transaction_cost_bps",
        "hidden_units",
        "learning_rate",
        "dropout",
        "scale_target",
        "avg_test_return",
        "avg_sharpe_like",
        "avg_max_drawdown",
        "positive_rate",
        "runs",
    ]
    if results.empty:
        return pd.DataFrame(columns=summary_columns)

    test_rows = results.loc[results["split"] == "test"].copy()
    if test_rows.empty:
        return pd.DataFrame(columns=summary_columns)
    if "scale_target" not in test_rows.columns:
        test_rows["scale_target"] = False
    if "feature_set_version" not in test_rows.columns:
        test_rows["feature_set_version"] = DEFAULT_FEATURE_SET_VERSION
    if "feature_count" not in test_rows.columns:
        test_rows["feature_count"] = np.nan
    if "min_holding_days" not in test_rows.columns:
        test_rows["min_holding_days"] = 1
    if "switch_margin" not in test_rows.columns:
        test_rows["switch_margin"] = 0.0
    if "allocation_mode" not in test_rows.columns:
        test_rows["allocation_mode"] = "equal_weight"
    else:
        test_rows["allocation_mode"] = test_rows["allocation_mode"].fillna("equal_weight")
    if "score_temperature" not in test_rows.columns:
        test_rows["score_temperature"] = 1.0
    else:
        test_rows["score_temperature"] = test_rows["score_temperature"].fillna(1.0)
    if "max_weight" not in test_rows.columns:
        test_rows["max_weight"] = 1.0
    else:
        test_rows["max_weight"] = test_rows["max_weight"].fillna(1.0)
    if "transaction_cost_bps" not in test_rows.columns:
        test_rows["transaction_cost_bps"] = 0.0
    else:
        test_rows["transaction_cost_bps"] = test_rows["transaction_cost_bps"].fillna(0.0)

    group_cols = [
        "universe",
        "model_family",
        "model_type",
        "feature_set_version",
        "feature_count",
        "top_k",
        "min_holding_days",
        "switch_margin",
        "allocation_mode",
        "score_temperature",
        "max_weight",
        "transaction_cost_bps",
        "hidden_units",
        "learning_rate",
        "dropout",
        "scale_target",
    ]

    grouped = (
        test_rows.groupby(group_cols, dropna=False)
        .agg(
            avg_test_return=("total_return", "mean"),
            avg_sharpe_like=("sharpe_like", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            positive_rate=("total_return", lambda values: float((values > 0).mean())),
            runs=("run_key", "nunique"),
        )
        .reset_index()
    )
    return grouped[summary_columns]


def _run_one_configuration(
    raw_df: pd.DataFrame,
    universe: SweepUniverseConfig,
    fold: WalkForwardFold,
    model: SweepModelConfig,
    signal: SweepSignalConfig,
    config: SweepRunConfig,
    run_key: str,
    transaction_cost_bps: float = 0.0,
    feature_set_version: str | None = None,
) -> list[dict[str, object]]:
    resolved_feature_set_version = (
        feature_set_version
        if feature_set_version is not None
        else config.feature_set_versions[0]
    )
    if model.model_family == "baseline":
        summary = _run_baseline(
            raw_df,
            fold,
            model,
            signal,
            config,
            transaction_cost_bps,
            resolved_feature_set_version,
        )
    elif model.model_family == "mlp":
        summary = _run_mlp(
            raw_df,
            fold,
            model,
            signal,
            config,
            transaction_cost_bps,
            resolved_feature_set_version,
        )
    else:
        raise ValueError(f"Unsupported model_family: {model.model_family!r}")

    rows: list[dict[str, object]] = []
    for _, summary_row in summary.iterrows():
        rows.append(
            {
                "run_key": run_key,
                "universe": universe.name,
                "data_path": str(universe.data_path),
                "fold": fold.name,
                "split": summary_row["split"],
                "model_family": model.model_family,
                "model_type": summary_row["model_type"],
                "feature_set_version": resolved_feature_set_version,
                "feature_count": summary_row.get("feature_count", np.nan),
                "hidden_units": _format_hidden_units(model),
                "dropout": model.dropout if model.model_family == "mlp" else np.nan,
                "scale_target": model.scale_target if model.model_family == "mlp" else False,
                "learning_rate": model.learning_rate if model.model_family == "mlp" else np.nan,
                "weight_decay": model.weight_decay if model.model_family == "mlp" else np.nan,
                "batch_size": model.batch_size if model.model_family == "mlp" else np.nan,
                "max_epochs": model.max_epochs if model.model_family == "mlp" else np.nan,
                "patience": model.patience if model.model_family == "mlp" else np.nan,
                "random_state": model.random_state,
                "top_k": signal.top_k,
                "min_holding_days": _effective_min_holding_days(
                    signal,
                    config.min_holding_days,
                ),
                "switch_margin": _effective_switch_margin(signal, config.switch_margin),
                "allocation_mode": signal.allocation_mode,
                "score_temperature": signal.score_temperature,
                "max_weight": signal.max_weight,
                "transaction_cost_bps": transaction_cost_bps,
                "target_col": summary_row["target_col"],
                "label_horizon_days": config.label_horizon_days,
                "total_return": summary_row["total_return"],
                "mean_daily_return": summary_row["mean_daily_return"],
                "daily_volatility": summary_row["daily_volatility"],
                "sharpe_like": summary_row["sharpe_like"],
                "max_drawdown": summary_row["max_drawdown"],
                "final_equity": summary_row["final_equity"],
                "total_transaction_cost": summary_row.get("total_transaction_cost", np.nan),
                "mean_daily_turnover": summary_row.get("mean_daily_turnover", np.nan),
                "max_daily_turnover": summary_row.get("max_daily_turnover", np.nan),
                "epochs_trained": summary_row.get("epochs_trained", np.nan),
            }
        )
    return rows


def _run_baseline(
    raw_df: pd.DataFrame,
    fold: WalkForwardFold,
    model: SweepModelConfig,
    signal: SweepSignalConfig,
    config: SweepRunConfig,
    transaction_cost_bps: float,
    feature_set_version: str,
) -> pd.DataFrame:
    experiment_config = BaselineE2EExperimentConfig(
        baseline_config=BaselineExperimentConfig(
            target_col=config.target_col,
            model_config=BaselineModelConfig(
                model_type=model.model_type,
                random_state=model.random_state,
            ),
            prediction_col=config.prediction_col,
            feature_set_version=feature_set_version,
            purge_config=_purge_config_from_sweep(config),
        ),
        signal_config=TopKSignalConfig(
            prediction_col=config.prediction_col,
            top_k=signal.top_k,
            allocation_mode=signal.allocation_mode,
            score_temperature=signal.score_temperature,
            max_weight=signal.max_weight,
        ),
        backtest_config=SignalBacktestConfig(
            initial_equity=config.initial_equity,
            transaction_cost_bps=transaction_cost_bps,
        ),
        smoothing_config=_smoothing_config_from_sweep_signal(signal, config),
    )
    result = run_baseline_e2e_experiment(raw_df, [fold], experiment_config)
    summary = summarize_baseline_e2e_experiment(result)
    summary["feature_count"] = len(result.baseline_result.feature_cols)
    return summary


def _run_mlp(
    raw_df: pd.DataFrame,
    fold: WalkForwardFold,
    model: SweepModelConfig,
    signal: SweepSignalConfig,
    config: SweepRunConfig,
    transaction_cost_bps: float,
    feature_set_version: str,
) -> pd.DataFrame:
    experiment_config = MLPE2EExperimentConfig(
        mlp_config=MLPExperimentConfig(
            target_col=config.target_col,
            model_config=MLPRegressorConfig(
                hidden_units=model.hidden_units,
                dropout=model.dropout,
                learning_rate=model.learning_rate,
                weight_decay=model.weight_decay,
                batch_size=model.batch_size,
                max_epochs=model.max_epochs,
                patience=model.patience,
                random_state=model.random_state,
                device=model.device,
                scale_target=model.scale_target,
            ),
            prediction_col=config.prediction_col,
            feature_set_version=feature_set_version,
            purge_config=_purge_config_from_sweep(config),
        ),
        signal_config=TopKSignalConfig(
            prediction_col=config.prediction_col,
            top_k=signal.top_k,
            allocation_mode=signal.allocation_mode,
            score_temperature=signal.score_temperature,
            max_weight=signal.max_weight,
        ),
        backtest_config=SignalBacktestConfig(
            initial_equity=config.initial_equity,
            transaction_cost_bps=transaction_cost_bps,
        ),
        smoothing_config=_smoothing_config_from_sweep_signal(signal, config),
    )
    result = run_mlp_e2e_experiment(raw_df, [fold], experiment_config)
    summary = summarize_mlp_e2e_experiment(result)
    summary["feature_count"] = len(result.mlp_result.feature_cols)
    return summary


def _validate_sweep_config(config: SweepRunConfig) -> None:
    if not config.universes:
        raise ValueError("At least one universe is required.")
    if not config.folds:
        raise ValueError("At least one fold is required.")
    if not config.models:
        raise ValueError("At least one model is required.")
    if not config.signals:
        raise ValueError("At least one signal is required.")
    if not config.feature_set_versions:
        raise ValueError("At least one feature_set_version is required.")
    for feature_set_version in config.feature_set_versions:
        validate_feature_set_version(feature_set_version)
    if config.label_horizon_days < 0:
        raise ValueError("label_horizon_days must be greater than or equal to 0.")
    if config.min_holding_days < 1:
        raise ValueError("min_holding_days must be greater than or equal to 1.")
    if config.switch_margin < 0.0:
        raise ValueError("switch_margin must be greater than or equal to 0.")
    if config.max_workers < 1:
        raise ValueError("max_workers must be greater than or equal to 1.")
    if not config.transaction_cost_bps_values:
        raise ValueError("At least one transaction_cost_bps value is required.")
    if any(value < 0.0 for value in config.transaction_cost_bps_values):
        raise ValueError("transaction_cost_bps values must be greater than or equal to 0.")


def _to_walk_forward_fold(config: SweepFoldConfig) -> WalkForwardFold:
    return WalkForwardFold(
        name=config.name,
        train_start=pd.Timestamp(config.train_start),
        train_end=pd.Timestamp(config.train_end),
        validation_start=pd.Timestamp(config.validation_start),
        validation_end=pd.Timestamp(config.validation_end),
        test_start=pd.Timestamp(config.test_start),
        test_end=pd.Timestamp(config.test_end),
    )


def _append_results(path: Path, results: pd.DataFrame) -> None:
    _ensure_results_file(path)
    results.to_csv(path, mode="a", header=False, index=False)


def _ensure_results_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(path, index=False)


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.read_csv(path)
    for column in RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    return results[RESULT_COLUMNS]


def _read_checkpoint_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_failed_key(path: Path, run_key: str, exc: Exception) -> None:
    message = " ".join(str(exc).splitlines())
    _append_line(path, f"{run_key}\t{type(exc).__name__}: {message}")


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    return {key: str(item) if isinstance(item, Path) else item for key, item in payload.items()}


def _slug(value: str) -> str:
    safe = [character.lower() if character.isalnum() else "-" for character in value]
    return "-".join("".join(safe).split("-")) or "unnamed"


def _format_hidden_units(model: SweepModelConfig) -> str:
    if model.model_family != "mlp":
        return "none"
    return ",".join(str(unit) for unit in model.hidden_units)


def _effective_min_holding_days(
    signal: SweepSignalConfig,
    config_min_holding_days: int | None,
) -> int:
    if signal.min_holding_days is not None:
        return signal.min_holding_days
    if config_min_holding_days is not None:
        return config_min_holding_days
    return 1


def _effective_switch_margin(
    signal: SweepSignalConfig,
    config_switch_margin: float | None,
) -> float:
    if signal.switch_margin is not None:
        return signal.switch_margin
    if config_switch_margin is not None:
        return config_switch_margin
    return 0.0


def _smoothing_config_from_sweep_signal(
    signal: SweepSignalConfig,
    config: SweepRunConfig,
) -> SignalSmoothingConfig | None:
    min_holding_days = _effective_min_holding_days(signal, config.min_holding_days)
    switch_margin = _effective_switch_margin(signal, config.switch_margin)
    if min_holding_days <= 1 and switch_margin <= 0.0:
        return None
    return SignalSmoothingConfig(
        min_holding_days=min_holding_days,
        switch_margin=switch_margin,
        prediction_col=config.prediction_col,
    )


def _purge_config_from_sweep(config: SweepRunConfig) -> PurgeConfig | None:
    if config.label_horizon_days <= 0:
        return None
    return PurgeConfig(label_horizon_days=config.label_horizon_days)


__all__ = [
    "SweepFoldConfig",
    "SweepModelConfig",
    "SweepRunConfig",
    "SweepSignalConfig",
    "SweepUniverseConfig",
    "build_sweep_run_key",
    "run_sweep",
    "summarize_sweep_results",
]
