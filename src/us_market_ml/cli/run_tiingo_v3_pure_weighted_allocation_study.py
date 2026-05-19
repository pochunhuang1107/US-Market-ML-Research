"""Run the Tiingo-only benchmark-gap v3 pure weighted-allocation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.allocation_overlays import (
    AllocationOverlaySpec,
    apply_allocation_overlay,
    default_pure_weighted_allocation_specs,
    validate_overlay_weights,
)
from us_market_ml.backtest import (
    SignalBacktestConfig,
    run_signal_backtest,
)
from us_market_ml.cli.run_tiingo_modern_baseline_sweep import (
    ALLOCATION_MODE,
    DATASET_SOURCE_LABEL,
    DEFAULT_DATASET_GLOB,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_HISTORY_ROWS,
    DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    FEATURE_SELECTION_SCOPE,
    INCLUDE_PARTIAL_FINAL_TEST,
    LABEL_HORIZON_DAYS,
    MIN_PARTIAL_TEST_DAYS,
    OLD_BRIDGED_DATASET_GLOB,
    RANDOM_STATE,
    TARGET_COL,
    UNIVERSE_MODE,
    _ensure_dataset_available,
    _format_command,
    _parse_date,
    _parse_max_workers,
    _parse_non_negative_float_values,
    _parse_positive_int,
    _parse_positive_int_values,
)
from us_market_ml.cli.run_tiingo_v3_allocation_overlay_study import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_STEP_MONTHS,
    DEFAULT_TEST_WINDOW_MONTHS,
    DEFAULT_TRAIN_WINDOW_YEARS,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_VALIDATION_WINDOW_MONTHS,
    MIN_HOLDING_DAYS,
    MODEL_FAMILY,
    MODEL_TYPE,
    SWITCH_MARGIN,
    TOP_K,
    _append_failed_key,
    _append_line,
    _append_results,
    _append_terminal_log,
    _benchmark_metrics,
    _eligible_splits,
    _ensure_benchmark_signal_rows,
    _ensure_results_file,
    _equity_metrics,
    _generate_smoothed_top5_signals,
    _invalid_reason,
    _markdown_table,
    _max_or_nan,
    _mean_or_nan,
    _non_empty_line_count,
    _overlay_weight_summary,
    _preflight_counts,
    _read_key_checkpoint,
    _read_results,
    _slug,
    _sum_or_nan,
    _values_arg,
)
from us_market_ml.data import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    get_feature_columns,
    read_local_market_data,
)
from us_market_ml.experiments import BaselineExperimentConfig, run_baseline_experiment
from us_market_ml.features import BENCHMARK_GAP_V3_FEATURE_SET_VERSION, validate_feature_set_version
from us_market_ml.models import BaselineModelConfig
from us_market_ml.splits import WalkForwardFold, make_sliding_window_folds
from us_market_ml.universe import select_fold_train_feature_columns

DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_v3_pure_weighted_allocation_study")
DEFAULT_RUN_NAME = "tiingo_v3_pure_weighted_allocation_study"
DEFAULT_HIGH_CONVICTION_GAP_THRESHOLD = 0.010
DEFAULT_SOFTMAX_TEMPERATURE = 0.050

OUTPUT_PREFIX = "tiingo_v3_pure_weighted_allocation"
RESULTS_NAME = f"{OUTPUT_PREFIX}_results.csv"
PREFLIGHT_CSV_NAME = f"{OUTPUT_PREFIX}_preflight.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"
COMPLETED_KEYS_NAME = f"{OUTPUT_PREFIX}_completed_keys.txt"
FAILED_KEYS_NAME = f"{OUTPUT_PREFIX}_failed_keys.txt"
TERMINAL_LOG_NAME = f"{OUTPUT_PREFIX}_terminal.log"

UNIVERSE_NAME = "tiingo_v3_pure_weighted_allocation_dynamic"
STUDY_STAGE = "pure_weighted_allocation"
PROGRESS_INTERVAL_SECONDS = 60

PREFLIGHT_COLUMNS = [
    "dataset_source",
    "study_stage",
    "run_name",
    "run_key",
    "allocation_rule_name",
    "allocation_rule_kind",
    "universe",
    "fold_name",
    "feature_set_version",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "model_family",
    "model_type",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "selected_feature_count",
    "min_train_eligible_rows",
    "train_eligible_row_count",
    "validation_eligible_row_count",
    "test_eligible_row_count",
    "train_symbol_count",
    "validation_symbol_count",
    "test_symbol_count",
    "min_symbols_per_test_date",
    "supports_top_k",
    "valid_config",
    "invalid_reason",
]

RESULT_COLUMNS = [
    "run_key",
    "dataset_source",
    "study_stage",
    "run_name",
    "allocation_rule_name",
    "allocation_rule_kind",
    "fold",
    "train_window_years",
    "validation_window_months",
    "test_window_months",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "feature_set_version",
    "model_family",
    "model_type",
    "top_k",
    "allocation_mode",
    "min_holding_days",
    "switch_margin",
    "transaction_cost_bps",
    "target_col",
    "label_horizon_days",
    "feature_count",
    "strategy_total_return",
    "spy_total_return",
    "qqq_total_return",
    "strategy_vs_spy",
    "strategy_vs_qqq",
    "strategy_sharpe_like",
    "strategy_max_drawdown",
    "total_transaction_cost",
    "mean_daily_turnover",
    "max_daily_turnover",
    "selected_symbol_count_mean",
    "selected_weight_sum_min",
    "selected_weight_sum_max",
    "overlay_weight_valid",
    "spy_sharpe_like",
    "qqq_sharpe_like",
    "spy_max_drawdown",
    "qqq_max_drawdown",
]


@dataclass(frozen=True)
class PureWeightedAllocationFoldPlan:
    """A replay fold plus the window metadata used to create it."""

    fold: WalkForwardFold
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass(frozen=True)
class TiingoV3PureWeightedAllocationStudyPlan:
    """Executable Tiingo-only pure weighted-allocation study plan."""

    fold_plans: tuple[PureWeightedAllocationFoldPlan, ...]
    allocation_rule_specs: tuple[AllocationOverlaySpec, ...]
    market_dataset_glob: str
    output_dir: Path
    run_name: str
    resume: bool
    max_workers: int
    start_date: str
    end_date: str
    step_months: int
    min_history_rows: int
    min_train_eligible_rows: int
    feature_set_version: str
    transaction_cost_bps: float

    @property
    def estimated_total_allocation_evaluations(self) -> int:
        return len(self.fold_plans) * len(self.allocation_rule_specs)


@dataclass(frozen=True)
class TiingoV3PureWeightedAllocationPreflight:
    """Diagnostic outputs generated before model fitting."""

    csv_path: Path
    markdown_path: Path
    results_csv_path: Path
    completed_keys_path: Path
    failed_keys_path: Path
    terminal_log_path: Path
    total_allocation_evaluation_count: int
    valid_allocation_evaluation_count: int
    invalid_allocation_evaluation_count: int
    fold_count: int
    allocation_rule_count: int
    completed_checkpoint_count: int
    failed_checkpoint_count: int
    pending_allocation_evaluation_count: int


def run_tiingo_v3_pure_weighted_allocation_study_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the Tiingo-only v3 pure weighted-allocation study."""

    args = _parse_args(argv)
    _ensure_dataset_available(args.market_dataset_glob)
    fold_plans = build_tiingo_v3_pure_weighted_allocation_study_folds(
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_years_values=args.train_window_years,
        validation_window_months_values=args.validation_window_months,
        test_window_months_values=args.test_window_months,
        step_months=args.step_months,
    )
    if not fold_plans:
        raise SystemExit("No Tiingo v3 pure weighted-allocation folds were generated.")

    allocation_rule_specs = default_pure_weighted_allocation_specs(
        high_conviction_gap_threshold=args.high_conviction_gap_threshold,
        softmax_temperature=args.softmax_temperature,
    )
    plan = build_tiingo_v3_pure_weighted_allocation_study_plan(
        market_dataset_glob=args.market_dataset_glob,
        fold_plans=fold_plans,
        allocation_rule_specs=allocation_rule_specs,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
        step_months=args.step_months,
        feature_set_version=args.feature_set_version,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    preflight = preflight_tiingo_v3_pure_weighted_allocation_study(plan)
    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    try:
        results = execute_tiingo_v3_pure_weighted_allocation_study(
            plan=plan,
            preflight=preflight,
        )
    except KeyboardInterrupt:
        print("Interrupt received; Tiingo v3 pure weighted-allocation study stopped cleanly.")
        return 130

    print()
    print("Pure weighted-allocation study complete.")
    print(f"results CSV: {preflight.results_csv_path}")
    print(f"completed checkpoint: {preflight.completed_keys_path}")
    print(f"failed checkpoint: {preflight.failed_keys_path}")
    print(f"terminal/progress log: {preflight.terminal_log_path}")
    if not results.empty:
        print()
        print(_quick_allocation_rule_summary(results).to_string(index=False))
    return 0


def build_tiingo_v3_pure_weighted_allocation_study_folds(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    train_window_years_values: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months_values: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months_values: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[PureWeightedAllocationFoldPlan]:
    """Build the pure weighted-allocation rolling window grid."""

    fold_plans: list[PureWeightedAllocationFoldPlan] = []
    for train_window_years in train_window_years_values:
        for validation_window_months in validation_window_months_values:
            for test_window_months in test_window_months_values:
                folds = make_sliding_window_folds(
                    data_start=start_date,
                    data_end=end_date,
                    train_window_years=train_window_years,
                    validation_window_months=validation_window_months,
                    test_window_months=test_window_months,
                    step_months=step_months,
                    include_partial_final_test=INCLUDE_PARTIAL_FINAL_TEST,
                    min_partial_test_days=MIN_PARTIAL_TEST_DAYS,
                )
                for fold in folds:
                    fold_plans.append(
                        PureWeightedAllocationFoldPlan(
                            fold=fold,
                            train_window_years=train_window_years,
                            validation_window_months=validation_window_months,
                            test_window_months=test_window_months,
                        )
                    )
    return fold_plans


def build_tiingo_v3_pure_weighted_allocation_study_plan(
    *,
    market_dataset_glob: str,
    fold_plans: Sequence[PureWeightedAllocationFoldPlan],
    allocation_rule_specs: Sequence[AllocationOverlaySpec],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
    min_train_eligible_rows: int = DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    step_months: int = DEFAULT_STEP_MONTHS,
    feature_set_version: str = BENCHMARK_GAP_V3_FEATURE_SET_VERSION,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> TiingoV3PureWeightedAllocationStudyPlan:
    """Build the focused Tiingo-only v3 pure weighted-allocation study plan."""

    if not fold_plans:
        raise ValueError("At least one fold plan is required.")
    if not allocation_rule_specs:
        raise ValueError("At least one allocation rule spec is required.")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be greater than or equal to 0.")
    feature_set_version = validate_feature_set_version(feature_set_version)
    if feature_set_version != BENCHMARK_GAP_V3_FEATURE_SET_VERSION:
        raise ValueError(
            "The pure weighted-allocation study is pinned to "
            f"{BENCHMARK_GAP_V3_FEATURE_SET_VERSION!r}."
        )
    _validate_pure_allocation_rule_specs(allocation_rule_specs)

    return TiingoV3PureWeightedAllocationStudyPlan(
        fold_plans=tuple(fold_plans),
        allocation_rule_specs=tuple(allocation_rule_specs),
        market_dataset_glob=market_dataset_glob,
        output_dir=output_dir,
        run_name=run_name,
        resume=resume,
        max_workers=max_workers,
        start_date=start_date,
        end_date=end_date,
        step_months=step_months,
        min_history_rows=min_history_rows,
        min_train_eligible_rows=min_train_eligible_rows,
        feature_set_version=feature_set_version,
        transaction_cost_bps=transaction_cost_bps,
    )


def preflight_tiingo_v3_pure_weighted_allocation_study(
    plan: TiingoV3PureWeightedAllocationStudyPlan,
) -> TiingoV3PureWeightedAllocationPreflight:
    """Write fold/rule diagnostics before model fitting."""

    output_dir = Path(plan.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / PREFLIGHT_CSV_NAME
    markdown_path = output_dir / PREFLIGHT_MD_NAME
    results_csv_path = output_dir / RESULTS_NAME
    completed_keys_path = output_dir / COMPLETED_KEYS_NAME
    failed_keys_path = output_dir / FAILED_KEYS_NAME
    terminal_log_path = output_dir / TERMINAL_LOG_NAME

    rows = _build_preflight_rows(plan)
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    frame.to_csv(csv_path, index=False)

    valid_keys = set(frame.loc[frame["valid_config"], "run_key"].astype(str).tolist())
    completed_keys_path.touch(exist_ok=True)
    failed_keys_path.touch(exist_ok=True)
    terminal_log_path.touch(exist_ok=True)
    completed_keys = _read_key_checkpoint(completed_keys_path) if plan.resume else set()
    completed_in_plan = valid_keys.intersection(completed_keys)
    failed_count = _non_empty_line_count(failed_keys_path) if plan.resume else 0
    pending_count = max(0, len(valid_keys) - len(completed_in_plan))

    preflight = TiingoV3PureWeightedAllocationPreflight(
        csv_path=csv_path,
        markdown_path=markdown_path,
        results_csv_path=results_csv_path,
        completed_keys_path=completed_keys_path,
        failed_keys_path=failed_keys_path,
        terminal_log_path=terminal_log_path,
        total_allocation_evaluation_count=int(len(frame)),
        valid_allocation_evaluation_count=(
            int(frame["valid_config"].sum()) if not frame.empty else 0
        ),
        invalid_allocation_evaluation_count=(
            int((~frame["valid_config"]).sum()) if not frame.empty else 0
        ),
        fold_count=len(plan.fold_plans),
        allocation_rule_count=len(plan.allocation_rule_specs),
        completed_checkpoint_count=len(completed_in_plan),
        failed_checkpoint_count=failed_count,
        pending_allocation_evaluation_count=pending_count,
    )
    markdown_path.write_text(
        _render_preflight_markdown(frame=frame, plan=plan, preflight=preflight),
        encoding="utf-8",
    )
    _append_terminal_log(
        terminal_log_path,
        f"preflight rows={len(frame)} valid={preflight.valid_allocation_evaluation_count}",
    )
    return preflight


def execute_tiingo_v3_pure_weighted_allocation_study(
    *,
    plan: TiingoV3PureWeightedAllocationStudyPlan,
    preflight: TiingoV3PureWeightedAllocationPreflight,
) -> pd.DataFrame:
    """Fit ridge once per pending fold and evaluate all pending allocation rules."""

    result_columns = list(RESULT_COLUMNS)
    if not plan.resume:
        preflight.results_csv_path.unlink(missing_ok=True)
        preflight.completed_keys_path.unlink(missing_ok=True)
        preflight.failed_keys_path.unlink(missing_ok=True)
    _ensure_results_file(preflight.results_csv_path, result_columns)
    preflight.completed_keys_path.touch(exist_ok=True)
    preflight.failed_keys_path.touch(exist_ok=True)

    preflight_frame = pd.read_csv(preflight.csv_path)
    valid_frame = preflight_frame.loc[preflight_frame["valid_config"]].copy()
    completed_keys = _read_key_checkpoint(preflight.completed_keys_path) if plan.resume else set()
    pending_frame = valid_frame.loc[~valid_frame["run_key"].astype(str).isin(completed_keys)]
    if pending_frame.empty:
        _append_terminal_log(preflight.terminal_log_path, "no pending allocation evaluations")
        return _read_results(preflight.results_csv_path, result_columns)

    pending_fold_names = set(pending_frame["fold_name"].astype(str).tolist())
    pending_fold_plans = [
        fold_plan for fold_plan in plan.fold_plans if fold_plan.fold.name in pending_fold_names
    ]
    fold_plan_by_name = {fold_plan.fold.name: fold_plan for fold_plan in pending_fold_plans}
    pending_keys = set(pending_frame["run_key"].astype(str).tolist())

    _append_terminal_log(
        preflight.terminal_log_path,
        (
            "starting shared ridge model fit for "
            f"{len(pending_fold_plans)} folds and {len(pending_keys)} allocation evaluations"
        ),
    )
    raw_df = read_local_market_data(plan.market_dataset_glob)
    baseline_result = run_baseline_experiment(
        raw_df=raw_df,
        folds=[fold_plan.fold for fold_plan in pending_fold_plans],
        config=_baseline_experiment_config(plan),
    )
    feature_count_by_fold = {
        fold_result.fold_name: len(fold_result.feature_cols or [])
        for fold_result in baseline_result.fold_results
    }

    completed_now = 0
    for fold_result in baseline_result.fold_results:
        fold_plan = fold_plan_by_name[fold_result.fold_name]
        base_signals = _ensure_benchmark_signal_rows(
            _generate_smoothed_top5_signals(fold_result.test_predictions),
            raw_df=raw_df,
            fold=fold_plan.fold,
        )
        for rule_spec in plan.allocation_rule_specs:
            run_key = build_pure_weighted_allocation_run_key(
                fold_plan=fold_plan,
                allocation_rule_spec=rule_spec,
                feature_set_version=plan.feature_set_version,
                transaction_cost_bps=plan.transaction_cost_bps,
                min_history_rows=plan.min_history_rows,
            )
            if run_key not in pending_keys:
                continue
            try:
                rule_signals = apply_allocation_overlay(base_signals, rule_spec, top_k=TOP_K)
                backtest = run_signal_backtest(
                    rule_signals,
                    SignalBacktestConfig(
                        initial_equity=1.0,
                        transaction_cost_bps=plan.transaction_cost_bps,
                    ),
                )
                row = _build_result_row(
                    run_key=run_key,
                    plan=plan,
                    fold_plan=fold_plan,
                    allocation_rule_spec=rule_spec,
                    rule_signals=rule_signals,
                    backtest_equity=backtest.equity_curve,
                    raw_df=raw_df,
                    feature_count=feature_count_by_fold.get(fold_result.fold_name, np.nan),
                )
            except Exception as exc:  # noqa: BLE001 - keep remaining rule work moving.
                _append_failed_key(preflight.failed_keys_path, run_key, exc)
                _append_terminal_log(
                    preflight.terminal_log_path,
                    f"failed {run_key}: {type(exc).__name__}: {exc}",
                )
                continue

            run_results = pd.DataFrame([row], columns=result_columns)
            _append_results(preflight.results_csv_path, run_results, result_columns)
            _append_line(preflight.completed_keys_path, run_key)
            completed_now += 1
            if completed_now == 1 or completed_now % len(plan.allocation_rule_specs) == 0:
                _append_terminal_log(
                    preflight.terminal_log_path,
                    f"completed_allocation_evaluations_this_run={completed_now}",
                )

    return _read_results(preflight.results_csv_path, result_columns)


def build_pure_weighted_allocation_run_key(
    *,
    fold_plan: PureWeightedAllocationFoldPlan,
    allocation_rule_spec: AllocationOverlaySpec,
    feature_set_version: str,
    transaction_cost_bps: float,
    min_history_rows: int,
) -> str:
    """Return a deterministic key for one fold/rule evaluation."""

    payload = {
        "dataset_source": DATASET_SOURCE_LABEL,
        "study_stage": STUDY_STAGE,
        "fold": {
            "name": fold_plan.fold.name,
            "train_start": str(fold_plan.fold.train_start.date()),
            "train_end": str(fold_plan.fold.train_end.date()),
            "validation_start": str(fold_plan.fold.validation_start.date()),
            "validation_end": str(fold_plan.fold.validation_end.date()),
            "test_start": str(fold_plan.fold.test_start.date()),
            "test_end": str(fold_plan.fold.test_end.date()),
        },
        "allocation_rule": _allocation_rule_payload(allocation_rule_spec),
        "feature_set_version": feature_set_version,
        "model_type": MODEL_TYPE,
        "top_k": TOP_K,
        "min_holding_days": MIN_HOLDING_DAYS,
        "switch_margin": SWITCH_MARGIN,
        "transaction_cost_bps": transaction_cost_bps,
        "universe_mode": UNIVERSE_MODE,
        "feature_selection_scope": FEATURE_SELECTION_SCOPE,
        "min_history_rows": min_history_rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return "__".join(
        [
            f"fold-{_slug(fold_plan.fold.name)}",
            f"rule-{_slug(allocation_rule_spec.name)}",
            f"ridge-top{TOP_K}",
            f"hold-{MIN_HOLDING_DAYS}d",
            f"margin-{SWITCH_MARGIN:g}",
            f"cost-{transaction_cost_bps:g}bps",
            f"features-{_slug(feature_set_version)}",
            digest,
        ]
    )


def _build_preflight_rows(
    plan: TiingoV3PureWeightedAllocationStudyPlan,
) -> list[dict[str, object]]:
    raw_df = read_local_market_data(plan.market_dataset_glob)
    dataset = build_supervised_dataset(
        raw_df,
        SupervisedDatasetConfig(
            forward_horizons=(5, 10, LABEL_HORIZON_DAYS),
            benchmark_symbol="SPY",
            excess_return_horizons=(LABEL_HORIZON_DAYS,),
            rank_target_col=TARGET_COL,
            rank_col=f"rank_{TARGET_COL}",
            feature_set_version=plan.feature_set_version,
        ),
    )
    if TARGET_COL not in dataset.columns:
        raise ValueError(
            "Configured target_col is missing after supervised dataset construction: "
            f"{TARGET_COL}"
        )
    candidate_feature_cols = get_feature_columns(
        dataset,
        feature_set_version=plan.feature_set_version,
    )

    rows: list[dict[str, object]] = []
    for fold_plan in plan.fold_plans:
        train_df, validation_df, test_df = fold_plan.fold.split(dataset)
        selected_feature_cols = select_fold_train_feature_columns(
            train_df,
            candidate_feature_cols,
            min_non_null_ratio=0.0,
        )
        train_eligible, validation_eligible, test_eligible = _eligible_splits(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            dataset=dataset,
            feature_cols=selected_feature_cols,
            min_history_rows=plan.min_history_rows,
        )
        counts = _preflight_counts(
            selected_feature_cols=selected_feature_cols,
            train_eligible=train_eligible,
            validation_eligible=validation_eligible,
            test_eligible=test_eligible,
        )
        supports_top_k = counts["min_symbols_per_test_date"] >= TOP_K
        invalid_reason = _invalid_reason(
            counts=counts,
            supports_top_k=supports_top_k,
            min_train_eligible_rows=plan.min_train_eligible_rows,
        )
        for rule_spec in plan.allocation_rule_specs:
            run_key = build_pure_weighted_allocation_run_key(
                fold_plan=fold_plan,
                allocation_rule_spec=rule_spec,
                feature_set_version=plan.feature_set_version,
                transaction_cost_bps=plan.transaction_cost_bps,
                min_history_rows=plan.min_history_rows,
            )
            rows.append(
                {
                    "dataset_source": DATASET_SOURCE_LABEL,
                    "study_stage": STUDY_STAGE,
                    "run_name": plan.run_name,
                    "run_key": run_key,
                    "allocation_rule_name": rule_spec.name,
                    "allocation_rule_kind": rule_spec.kind,
                    "universe": UNIVERSE_NAME,
                    "fold_name": fold_plan.fold.name,
                    "feature_set_version": plan.feature_set_version,
                    "train_window_years": fold_plan.train_window_years,
                    "validation_window_months": fold_plan.validation_window_months,
                    "test_window_months": fold_plan.test_window_months,
                    "train_start": fold_plan.fold.train_start.date().isoformat(),
                    "train_end": fold_plan.fold.train_end.date().isoformat(),
                    "validation_start": fold_plan.fold.validation_start.date().isoformat(),
                    "validation_end": fold_plan.fold.validation_end.date().isoformat(),
                    "test_start": fold_plan.fold.test_start.date().isoformat(),
                    "test_end": fold_plan.fold.test_end.date().isoformat(),
                    "model_family": MODEL_FAMILY,
                    "model_type": MODEL_TYPE,
                    "top_k": TOP_K,
                    "allocation_mode": ALLOCATION_MODE,
                    "min_holding_days": MIN_HOLDING_DAYS,
                    "switch_margin": SWITCH_MARGIN,
                    "transaction_cost_bps": plan.transaction_cost_bps,
                    "min_train_eligible_rows": plan.min_train_eligible_rows,
                    **counts,
                    "supports_top_k": supports_top_k,
                    "valid_config": invalid_reason == "",
                    "invalid_reason": invalid_reason,
                }
            )
    return rows


def _baseline_experiment_config(
    plan: TiingoV3PureWeightedAllocationStudyPlan,
) -> BaselineExperimentConfig:
    return BaselineExperimentConfig(
        target_col=TARGET_COL,
        model_config=BaselineModelConfig(
            model_type=MODEL_TYPE,
            random_state=RANDOM_STATE,
        ),
        dataset_config=SupervisedDatasetConfig(
            forward_horizons=(5, 10, LABEL_HORIZON_DAYS),
            benchmark_symbol="SPY",
            excess_return_horizons=(LABEL_HORIZON_DAYS,),
            rank_target_col=TARGET_COL,
            rank_col=f"rank_{TARGET_COL}",
            feature_set_version=plan.feature_set_version,
        ),
        feature_set_version=plan.feature_set_version,
        prediction_col="prediction",
        universe_mode=UNIVERSE_MODE,
        min_history_rows=plan.min_history_rows,
        feature_selection_scope=FEATURE_SELECTION_SCOPE,
        min_feature_non_null_ratio=0.0,
    )


def _build_result_row(
    *,
    run_key: str,
    plan: TiingoV3PureWeightedAllocationStudyPlan,
    fold_plan: PureWeightedAllocationFoldPlan,
    allocation_rule_spec: AllocationOverlaySpec,
    rule_signals: pd.DataFrame,
    backtest_equity: pd.DataFrame,
    raw_df: pd.DataFrame,
    feature_count: int | float,
) -> dict[str, object]:
    strategy_metrics = _equity_metrics(
        backtest_equity,
        return_col="portfolio_return",
        equity_col="equity",
    )
    spy_metrics = _benchmark_metrics(raw_df, "SPY", fold_plan.fold)
    qqq_metrics = _benchmark_metrics(raw_df, "QQQ", fold_plan.fold)
    weight_summary = _overlay_weight_summary(rule_signals)
    return {
        "run_key": run_key,
        "dataset_source": DATASET_SOURCE_LABEL,
        "study_stage": STUDY_STAGE,
        "run_name": plan.run_name,
        "allocation_rule_name": allocation_rule_spec.name,
        "allocation_rule_kind": allocation_rule_spec.kind,
        "fold": fold_plan.fold.name,
        "train_window_years": fold_plan.train_window_years,
        "validation_window_months": fold_plan.validation_window_months,
        "test_window_months": fold_plan.test_window_months,
        "train_start": fold_plan.fold.train_start.date().isoformat(),
        "train_end": fold_plan.fold.train_end.date().isoformat(),
        "validation_start": fold_plan.fold.validation_start.date().isoformat(),
        "validation_end": fold_plan.fold.validation_end.date().isoformat(),
        "test_start": fold_plan.fold.test_start.date().isoformat(),
        "test_end": fold_plan.fold.test_end.date().isoformat(),
        "feature_set_version": plan.feature_set_version,
        "model_family": MODEL_FAMILY,
        "model_type": MODEL_TYPE,
        "top_k": TOP_K,
        "allocation_mode": ALLOCATION_MODE,
        "min_holding_days": MIN_HOLDING_DAYS,
        "switch_margin": SWITCH_MARGIN,
        "transaction_cost_bps": plan.transaction_cost_bps,
        "target_col": TARGET_COL,
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "feature_count": feature_count,
        "strategy_total_return": strategy_metrics["total_return"],
        "spy_total_return": spy_metrics["total_return"],
        "qqq_total_return": qqq_metrics["total_return"],
        "strategy_vs_spy": strategy_metrics["total_return"] - spy_metrics["total_return"],
        "strategy_vs_qqq": strategy_metrics["total_return"] - qqq_metrics["total_return"],
        "strategy_sharpe_like": strategy_metrics["sharpe_like"],
        "strategy_max_drawdown": strategy_metrics["max_drawdown"],
        "total_transaction_cost": _sum_or_nan(backtest_equity.get("transaction_cost")),
        "mean_daily_turnover": _mean_or_nan(backtest_equity.get("turnover")),
        "max_daily_turnover": _max_or_nan(backtest_equity.get("turnover")),
        "selected_symbol_count_mean": weight_summary["selected_symbol_count_mean"],
        "selected_weight_sum_min": weight_summary["selected_weight_sum_min"],
        "selected_weight_sum_max": weight_summary["selected_weight_sum_max"],
        "overlay_weight_valid": validate_overlay_weights(rule_signals),
        "spy_sharpe_like": spy_metrics["sharpe_like"],
        "qqq_sharpe_like": qqq_metrics["sharpe_like"],
        "spy_max_drawdown": spy_metrics["max_drawdown"],
        "qqq_max_drawdown": qqq_metrics["max_drawdown"],
    }


def _render_preflight_markdown(
    *,
    frame: pd.DataFrame,
    plan: TiingoV3PureWeightedAllocationStudyPlan,
    preflight: TiingoV3PureWeightedAllocationPreflight,
) -> str:
    rule_summary = (
        frame.groupby("allocation_rule_name", sort=False)
        .agg(
            allocation_rule_kind=("allocation_rule_kind", "first"),
            config_count=("run_key", "nunique"),
            valid_config_count=("valid_config", "sum"),
        )
        .reset_index()
        if not frame.empty
        else pd.DataFrame(
            columns=[
                "allocation_rule_name",
                "allocation_rule_kind",
                "config_count",
                "valid_config_count",
            ]
        )
    )
    invalid_preview = frame.loc[~frame["valid_config"]].head(20)
    lines = [
        "# Tiingo V3 Pure Weighted-Allocation Study Preflight",
        "",
        "Local-only diagnostic before the pure rotation/weighted-allocation study.",
        "",
        "## Dataset",
        "",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        "- Source provider: `tiingo`",
        f"- Dataset path/glob: `{plan.market_dataset_glob}`",
        f"- Research start date: `{plan.start_date}`",
        f"- Research end date: `{plan.end_date}`",
        "",
        "## Candidate Lock",
        "",
        f"- Feature set version: `{plan.feature_set_version}`",
        f"- Model type: `{MODEL_TYPE}`",
        f"- Top K source signals: `{TOP_K}`",
        f"- Allocation baseline: `{ALLOCATION_MODE}`",
        f"- Minimum holding days: `{MIN_HOLDING_DAYS}`",
        f"- Switch margin: `{SWITCH_MARGIN:g}`",
        f"- Transaction cost bps: `{plan.transaction_cost_bps:g}`",
        f"- Target: `{TARGET_COL}`",
        f"- Label horizon days: `{LABEL_HORIZON_DAYS}`",
        "",
        "## Allocation Counts",
        "",
        f"- Generated fold count: `{preflight.fold_count}`",
        f"- Allocation rule count: `{preflight.allocation_rule_count}`",
        (
            "- Total allocation evaluations: "
            f"`{preflight.total_allocation_evaluation_count}`"
        ),
        (
            "- Valid allocation evaluations: "
            f"`{preflight.valid_allocation_evaluation_count}`"
        ),
        (
            "- Invalid allocation evaluations: "
            f"`{preflight.invalid_allocation_evaluation_count}`"
        ),
        (
            "- Completed allocation evaluations from checkpoint: "
            f"`{preflight.completed_checkpoint_count}`"
        ),
        f"- Failed checkpoint count if RESUME=1: `{preflight.failed_checkpoint_count}`",
        (
            "- Pending allocation evaluations: "
            f"`{preflight.pending_allocation_evaluation_count}`"
        ),
        "",
        "## Pure Allocation Rules Tested",
        "",
        _markdown_table(
            rule_summary,
            [
                "allocation_rule_name",
                "allocation_rule_kind",
                "config_count",
                "valid_config_count",
            ],
        ),
        "",
        "## Window Grid",
        "",
        f"- Train window years: `{_unique_train_windows(plan)}`",
        f"- Validation window months: `{_unique_validation_windows(plan)}`",
        f"- Test window months: `{_unique_test_windows(plan)}`",
        f"- Step months: `{plan.step_months}`",
        f"- Include partial final test: `{INCLUDE_PARTIAL_FINAL_TEST}`",
        f"- Minimum partial test days: `{MIN_PARTIAL_TEST_DAYS}`",
        "",
        "## Research Controls",
        "",
        f"- Universe mode: `{UNIVERSE_MODE}`",
        f"- Feature selection scope: `{FEATURE_SELECTION_SCOPE}`",
        f"- Minimum history rows: `{plan.min_history_rows}`",
        f"- Minimum train eligible rows: `{plan.min_train_eligible_rows}`",
        "- Local-only: `true`",
        "- Uses S3/cloud/GPU/neural networks/tree models: `false`",
        "- Production signal export: `false`",
        "- Fixed QQQ core exposure: `false`",
        "- Fixed SPY core exposure: `false`",
        (
            "- Model fitting is shared once per pending fold; rules reuse the same "
            "smoothed top-5 signals."
        ),
        "",
        "## Output Paths",
        "",
        f"- Results CSV: `{preflight.results_csv_path}`",
        f"- Completed keys: `{preflight.completed_keys_path}`",
        f"- Failed keys: `{preflight.failed_keys_path}`",
        f"- Terminal/progress log: `{preflight.terminal_log_path}`",
        "",
        "## Invalid/Thin Fold Preview",
        "",
        _markdown_table(
            invalid_preview,
            [
                "fold_name",
                "allocation_rule_name",
                "selected_feature_count",
                "train_eligible_row_count",
                "min_train_eligible_rows",
                "min_symbols_per_test_date",
                "invalid_reason",
            ],
        ),
        "",
        "## Deferral Notes",
        "",
        (
            "`tree_screen`, MLP, feature v4, S3 readers, production signal export, "
            "and QQQ-core sizing are intentionally not part of this pass."
        ),
        "",
    ]
    return "\n".join(lines)


def _print_plan(
    *,
    plan: TiingoV3PureWeightedAllocationStudyPlan,
    preflight: TiingoV3PureWeightedAllocationPreflight,
    dry_run: bool,
) -> None:
    print("US Market ML Tiingo-only benchmark-gap v3 pure weighted-allocation study")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("uses GPU: false")
    print("includes neural networks/MLP: false")
    print("includes tree models: false")
    print("exports production signals: false")
    print("fixed QQQ core exposure: false")
    print("fixed SPY core exposure: false")
    print(f"dry_run: {dry_run}")
    print(f"dataset glob: {plan.market_dataset_glob}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"start date: {plan.start_date}")
    print(f"end date: {plan.end_date}")
    print(f"feature_set_version: {plan.feature_set_version}")
    print(f"model_type: {MODEL_TYPE}")
    print(f"top_k source signals: {TOP_K}")
    print(f"allocation_mode baseline: {ALLOCATION_MODE}")
    print(f"min_holding_days: {MIN_HOLDING_DAYS}")
    print(f"switch_margin: {SWITCH_MARGIN:g}")
    print(f"transaction_cost_bps: {plan.transaction_cost_bps:g}")
    print(f"universe_mode: {UNIVERSE_MODE}")
    print(f"feature_selection_scope: {FEATURE_SELECTION_SCOPE}")
    print(f"target_col: {TARGET_COL}")
    print(f"label_horizon_days: {LABEL_HORIZON_DAYS}")
    print(f"generated fold count: {preflight.fold_count}")
    print(f"allocation rule count: {preflight.allocation_rule_count}")
    print(f"total allocation evaluations: {preflight.total_allocation_evaluation_count}")
    print(f"valid allocation evaluations: {preflight.valid_allocation_evaluation_count}")
    print(f"pending allocation evaluations: {preflight.pending_allocation_evaluation_count}")
    print("allocation rule names:")
    for rule_spec in plan.allocation_rule_specs:
        print(f"  - {rule_spec.name}")
    print(f"output directory: {plan.output_dir}")
    print(f"results CSV path: {preflight.results_csv_path}")
    print(f"preflight CSV path: {preflight.csv_path}")
    print(f"preflight Markdown path: {preflight.markdown_path}")
    print(f"completed checkpoint path: {preflight.completed_keys_path}")
    print(f"failed checkpoint path: {preflight.failed_keys_path}")
    print(f"terminal/progress log path: {preflight.terminal_log_path}")
    print(f"max_workers argument accepted: {plan.max_workers}")
    print("model fit sharing: one ridge fit per pending fold; rules reuse smoothed signals")
    print("tree_screen: not run")
    print("MLP: not run")
    print("feature v4: not added")
    print("production signal export: not run")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(plan)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Tiingo-only benchmark-gap v3 pure allocation study."
    )
    parser.add_argument("--market-dataset-glob", default=_default_market_dataset_glob())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", DEFAULT_RUN_NAME))
    parser.add_argument("--start-date", default=os.environ.get("START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.environ.get("END_DATE", DEFAULT_END_DATE))
    parser.add_argument(
        "--max-workers",
        type=_parse_max_workers,
        default=_parse_max_workers(os.environ.get("MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
        help="Accepted for runner symmetry; model fits are shared per fold.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("RESUME", True),
        help="Resume from local checkpoint files by default.",
    )
    parser.add_argument("--dry-run", action="store_true", default=_env_bool("DRY_RUN", False))
    parser.add_argument(
        "--min-history-rows",
        type=_parse_positive_int,
        default=DEFAULT_MIN_HISTORY_ROWS,
    )
    parser.add_argument(
        "--min-train-eligible-rows",
        type=_parse_positive_int,
        default=DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    )
    parser.add_argument("--step-months", default=os.environ.get("STEP_MONTHS"))
    parser.add_argument(
        "--train-window-years",
        nargs="+",
        default=_env_values("TRAIN_WINDOW_YEARS"),
    )
    parser.add_argument(
        "--validation-window-months",
        nargs="+",
        default=_env_values("VALIDATION_WINDOW_MONTHS"),
    )
    parser.add_argument(
        "--test-window-months",
        nargs="+",
        default=_env_values("TEST_WINDOW_MONTHS"),
    )
    parser.add_argument(
        "--feature-set-version",
        default=os.environ.get("FEATURE_SET_VERSION", BENCHMARK_GAP_V3_FEATURE_SET_VERSION),
    )
    parser.add_argument("--transaction-cost-bps", default=os.environ.get("TRANSACTION_COST_BPS"))
    parser.add_argument(
        "--high-conviction-gap-threshold",
        default=os.environ.get(
            "HIGH_CONVICTION_GAP_THRESHOLD",
            str(DEFAULT_HIGH_CONVICTION_GAP_THRESHOLD),
        ),
    )
    parser.add_argument(
        "--softmax-temperature",
        default=os.environ.get("SOFTMAX_TEMPERATURE", str(DEFAULT_SOFTMAX_TEMPERATURE)),
    )
    args = parser.parse_args(argv)

    try:
        args.step_months = (
            _parse_positive_int(str(args.step_months), "--step-months")
            if args.step_months is not None
            else DEFAULT_STEP_MONTHS
        )
        args.train_window_years = (
            _parse_positive_int_values(args.train_window_years, "--train-window-years")
            if args.train_window_years is not None
            else DEFAULT_TRAIN_WINDOW_YEARS
        )
        args.validation_window_months = (
            _parse_positive_int_values(
                args.validation_window_months,
                "--validation-window-months",
            )
            if args.validation_window_months is not None
            else DEFAULT_VALIDATION_WINDOW_MONTHS
        )
        args.test_window_months = (
            _parse_positive_int_values(args.test_window_months, "--test-window-months")
            if args.test_window_months is not None
            else DEFAULT_TEST_WINDOW_MONTHS
        )
        args.transaction_cost_bps = (
            _parse_non_negative_float_values(
                [str(args.transaction_cost_bps)],
                "--transaction-cost-bps",
            )[0]
            if args.transaction_cost_bps is not None
            else DEFAULT_TRANSACTION_COST_BPS
        )
        args.high_conviction_gap_threshold = _parse_non_negative_float_values(
            [str(args.high_conviction_gap_threshold)],
            "--high-conviction-gap-threshold",
        )[0]
        args.softmax_temperature = float(args.softmax_temperature)
        if args.softmax_temperature <= 0.0:
            raise argparse.ArgumentTypeError("--softmax-temperature must be greater than 0.")
        args.feature_set_version = validate_feature_set_version(str(args.feature_set_version))
        _parse_date(args.start_date, "--start-date")
        _parse_date(args.end_date, "--end-date")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    if args.feature_set_version != BENCHMARK_GAP_V3_FEATURE_SET_VERSION:
        parser.error(
            "This study is Tiingo v3-only; --feature-set-version must be "
            f"{BENCHMARK_GAP_V3_FEATURE_SET_VERSION}."
        )
    if str(args.market_dataset_glob) == OLD_BRIDGED_DATASET_GLOB:
        parser.error(
            "The old Tiingo/Massive bridged dataset is not allowed for this "
            "Tiingo-only pure weighted-allocation study."
        )
    if str(args.run_name).strip() == "":
        args.run_name = DEFAULT_RUN_NAME
    return args


def _equivalent_command(plan: TiingoV3PureWeightedAllocationStudyPlan) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_tiingo_v3_pure_weighted_allocation_study",
        "--market-dataset-glob",
        plan.market_dataset_glob,
        "--output-dir",
        str(plan.output_dir),
        "--run-name",
        plan.run_name,
        "--start-date",
        plan.start_date,
        "--end-date",
        plan.end_date,
        "--max-workers",
        str(plan.max_workers),
        "--min-history-rows",
        str(plan.min_history_rows),
        "--min-train-eligible-rows",
        str(plan.min_train_eligible_rows),
        "--step-months",
        str(plan.step_months),
        "--feature-set-version",
        plan.feature_set_version,
        "--transaction-cost-bps",
        f"{plan.transaction_cost_bps:g}",
    ]
    command.extend(_values_arg("--train-window-years", _unique_train_windows(plan)))
    command.extend(
        _values_arg("--validation-window-months", _unique_validation_windows(plan))
    )
    command.extend(_values_arg("--test-window-months", _unique_test_windows(plan)))
    command.append("--resume" if plan.resume else "--no-resume")
    return command


def _quick_allocation_rule_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby("allocation_rule_name", sort=False)
        .agg(
            folds=("fold", "nunique"),
            average_total_return=("strategy_total_return", "mean"),
            average_strategy_vs_spy=("strategy_vs_spy", "mean"),
            average_strategy_vs_qqq=("strategy_vs_qqq", "mean"),
            worst_max_drawdown=("strategy_max_drawdown", "min"),
        )
        .reset_index()
        .sort_values(
            ["average_strategy_vs_qqq", "average_total_return"],
            ascending=[False, False],
            kind="mergesort",
        )
    )


def _validate_pure_allocation_rule_specs(specs: Sequence[AllocationOverlaySpec]) -> None:
    disallowed_kinds = {
        "static_core_satellite",
        "dynamic_core_satellite",
        "qqq_floor_if_not_selected",
    }
    disallowed = [
        spec.name
        for spec in specs
        if spec.kind in disallowed_kinds or spec.core_weights or _has_core_token(spec.name)
    ]
    if disallowed:
        raise ValueError(
            "Pure weighted-allocation study cannot include fixed/core rules: "
            + ", ".join(disallowed)
        )


def _has_core_token(value: str) -> bool:
    return "core" in str(value).lower().replace("-", "_").split("_")


def _allocation_rule_payload(spec: AllocationOverlaySpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "selection_count": spec.selection_count,
        "rank_weights": spec.rank_weights,
        "max_weight": spec.max_weight,
        "score_temperature": spec.score_temperature,
        "conviction_gap_threshold": spec.conviction_gap_threshold,
        "trigger_symbol": spec.trigger_symbol,
        "trigger_rank_max": spec.trigger_rank_max,
        "trigger_selection_count": spec.trigger_selection_count,
        "fallback_selection_count": spec.fallback_selection_count,
        "core_weights": spec.core_weights,
        "satellite_weight": spec.satellite_weight,
        "regime_columns": spec.regime_columns,
    }


def _unique_train_windows(plan: TiingoV3PureWeightedAllocationStudyPlan) -> list[int]:
    return sorted({fold_plan.train_window_years for fold_plan in plan.fold_plans})


def _unique_validation_windows(plan: TiingoV3PureWeightedAllocationStudyPlan) -> list[int]:
    return sorted({fold_plan.validation_window_months for fold_plan in plan.fold_plans})


def _unique_test_windows(plan: TiingoV3PureWeightedAllocationStudyPlan) -> list[int]:
    return sorted({fold_plan.test_window_months for fold_plan in plan.fold_plans})


def _default_market_dataset_glob() -> str:
    return os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB)


def _env_values(name: str) -> list[str] | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return shlex.split(raw_value)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


run_tiingo_v3_pure_weighted_allocation_from_args = (
    run_tiingo_v3_pure_weighted_allocation_study_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_v3_pure_weighted_allocation_study_from_args())
