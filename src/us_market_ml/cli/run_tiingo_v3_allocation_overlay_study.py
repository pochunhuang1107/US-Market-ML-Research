"""Run the Tiingo-only benchmark-gap v3 allocation overlay study."""

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
    default_allocation_overlay_specs,
    validate_overlay_weights,
)
from us_market_ml.backtest import (
    SignalBacktestConfig,
    SignalSmoothingConfig,
    TopKSignalConfig,
    calculate_max_drawdown,
    generate_top_k_signals,
    run_signal_backtest,
    smooth_top_k_signals,
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
from us_market_ml.universe import (
    DynamicEligibilityConfig,
    apply_dynamic_universe_eligibility,
    select_fold_train_feature_columns,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_v3_allocation_overlay_study")
DEFAULT_RUN_NAME = "tiingo_v3_allocation_overlay_study"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_TRAIN_WINDOW_YEARS = (3, 5)
DEFAULT_VALIDATION_WINDOW_MONTHS = (12,)
DEFAULT_TEST_WINDOW_MONTHS = (3, 6, 12)
DEFAULT_STEP_MONTHS = 3
DEFAULT_TRANSACTION_COST_BPS = 10.0

OUTPUT_PREFIX = "tiingo_v3_allocation_overlay"
RESULTS_NAME = f"{OUTPUT_PREFIX}_results.csv"
PREFLIGHT_CSV_NAME = f"{OUTPUT_PREFIX}_preflight.csv"
PREFLIGHT_MD_NAME = f"{OUTPUT_PREFIX}_preflight.md"
COMPLETED_KEYS_NAME = f"{OUTPUT_PREFIX}_completed_keys.txt"
FAILED_KEYS_NAME = f"{OUTPUT_PREFIX}_failed_keys.txt"
TERMINAL_LOG_NAME = f"{OUTPUT_PREFIX}_terminal.log"

MODEL_FAMILY = "baseline"
MODEL_TYPE = "ridge"
TOP_K = 5
MIN_HOLDING_DAYS = 20
SWITCH_MARGIN = 0.005
UNIVERSE_NAME = "tiingo_v3_allocation_overlay_dynamic"
STUDY_STAGE = "allocation_risk_overlay"
PROGRESS_INTERVAL_SECONDS = 60

PREFLIGHT_COLUMNS = [
    "dataset_source",
    "study_stage",
    "run_name",
    "run_key",
    "overlay_name",
    "overlay_kind",
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
    "overlay_name",
    "overlay_kind",
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
class AllocationOverlayFoldPlan:
    """A replay fold plus the window metadata used to create it."""

    fold: WalkForwardFold
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass(frozen=True)
class TiingoV3AllocationOverlayStudyPlan:
    """Executable Tiingo-only allocation overlay study plan."""

    fold_plans: tuple[AllocationOverlayFoldPlan, ...]
    overlay_specs: tuple[AllocationOverlaySpec, ...]
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
    def estimated_total_overlay_evaluations(self) -> int:
        return len(self.fold_plans) * len(self.overlay_specs)


@dataclass(frozen=True)
class TiingoV3AllocationOverlayPreflight:
    """Diagnostic outputs generated before model fitting."""

    csv_path: Path
    markdown_path: Path
    results_csv_path: Path
    completed_keys_path: Path
    failed_keys_path: Path
    terminal_log_path: Path
    total_overlay_evaluation_count: int
    valid_overlay_evaluation_count: int
    invalid_overlay_evaluation_count: int
    fold_count: int
    overlay_count: int
    completed_checkpoint_count: int
    failed_checkpoint_count: int
    pending_overlay_evaluation_count: int


def run_tiingo_v3_allocation_overlay_study_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the Tiingo-only v3 allocation overlay study."""

    args = _parse_args(argv)
    _ensure_dataset_available(args.market_dataset_glob)
    fold_plans = build_tiingo_v3_allocation_overlay_study_folds(
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_years_values=args.train_window_years,
        validation_window_months_values=args.validation_window_months,
        test_window_months_values=args.test_window_months,
        step_months=args.step_months,
    )
    if not fold_plans:
        raise SystemExit("No Tiingo v3 allocation overlay folds were generated.")

    overlay_specs = default_allocation_overlay_specs(
        include_optional_qqq_floor=args.include_optional_qqq_floor,
    )
    plan = build_tiingo_v3_allocation_overlay_study_plan(
        market_dataset_glob=args.market_dataset_glob,
        fold_plans=fold_plans,
        overlay_specs=overlay_specs,
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
    preflight = preflight_tiingo_v3_allocation_overlay_study(plan)
    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    try:
        results = execute_tiingo_v3_allocation_overlay_study(
            plan=plan,
            preflight=preflight,
        )
    except KeyboardInterrupt:
        print("Interrupt received; Tiingo v3 allocation overlay study stopped cleanly.")
        return 130

    print()
    print("Allocation overlay study complete.")
    print(f"results CSV: {preflight.results_csv_path}")
    print(f"completed checkpoint: {preflight.completed_keys_path}")
    print(f"failed checkpoint: {preflight.failed_keys_path}")
    print(f"terminal/progress log: {preflight.terminal_log_path}")
    if not results.empty:
        summary = _quick_overlay_summary(results)
        print()
        print(summary.to_string(index=False))
    return 0


def build_tiingo_v3_allocation_overlay_study_folds(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    train_window_years_values: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months_values: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months_values: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[AllocationOverlayFoldPlan]:
    """Build the allocation-overlay rolling window grid."""

    fold_plans: list[AllocationOverlayFoldPlan] = []
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
                        AllocationOverlayFoldPlan(
                            fold=fold,
                            train_window_years=train_window_years,
                            validation_window_months=validation_window_months,
                            test_window_months=test_window_months,
                        )
                    )
    return fold_plans


def build_tiingo_v3_allocation_overlay_study_plan(
    *,
    market_dataset_glob: str,
    fold_plans: Sequence[AllocationOverlayFoldPlan],
    overlay_specs: Sequence[AllocationOverlaySpec],
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
) -> TiingoV3AllocationOverlayStudyPlan:
    """Build the focused Tiingo-only v3 allocation overlay study plan."""

    if not fold_plans:
        raise ValueError("At least one fold plan is required.")
    if not overlay_specs:
        raise ValueError("At least one allocation overlay spec is required.")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be greater than or equal to 0.")
    feature_set_version = validate_feature_set_version(feature_set_version)
    if feature_set_version != BENCHMARK_GAP_V3_FEATURE_SET_VERSION:
        raise ValueError(
            "The v3 allocation overlay study is pinned to "
            f"{BENCHMARK_GAP_V3_FEATURE_SET_VERSION!r}."
        )

    return TiingoV3AllocationOverlayStudyPlan(
        fold_plans=tuple(fold_plans),
        overlay_specs=tuple(overlay_specs),
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


def preflight_tiingo_v3_allocation_overlay_study(
    plan: TiingoV3AllocationOverlayStudyPlan,
) -> TiingoV3AllocationOverlayPreflight:
    """Write fold/overlay diagnostics before model fitting."""

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

    preflight = TiingoV3AllocationOverlayPreflight(
        csv_path=csv_path,
        markdown_path=markdown_path,
        results_csv_path=results_csv_path,
        completed_keys_path=completed_keys_path,
        failed_keys_path=failed_keys_path,
        terminal_log_path=terminal_log_path,
        total_overlay_evaluation_count=int(len(frame)),
        valid_overlay_evaluation_count=(
            int(frame["valid_config"].sum()) if not frame.empty else 0
        ),
        invalid_overlay_evaluation_count=(
            int((~frame["valid_config"]).sum()) if not frame.empty else 0
        ),
        fold_count=len(plan.fold_plans),
        overlay_count=len(plan.overlay_specs),
        completed_checkpoint_count=len(completed_in_plan),
        failed_checkpoint_count=failed_count,
        pending_overlay_evaluation_count=pending_count,
    )
    markdown_path.write_text(
        _render_preflight_markdown(frame=frame, plan=plan, preflight=preflight),
        encoding="utf-8",
    )
    _append_terminal_log(
        terminal_log_path,
        f"preflight rows={len(frame)} valid={preflight.valid_overlay_evaluation_count}",
    )
    return preflight


def execute_tiingo_v3_allocation_overlay_study(
    *,
    plan: TiingoV3AllocationOverlayStudyPlan,
    preflight: TiingoV3AllocationOverlayPreflight,
) -> pd.DataFrame:
    """Fit ridge once per pending fold and evaluate all pending overlays."""

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
        _append_terminal_log(preflight.terminal_log_path, "no pending overlay evaluations")
        return _read_results(preflight.results_csv_path, result_columns)

    pending_fold_names = set(pending_frame["fold_name"].astype(str).tolist())
    pending_fold_plans = [
        fold_plan for fold_plan in plan.fold_plans if fold_plan.fold.name in pending_fold_names
    ]
    fold_plan_by_name = {fold_plan.fold.name: fold_plan for fold_plan in pending_fold_plans}
    overlay_by_name = {overlay.name: overlay for overlay in plan.overlay_specs}
    pending_keys = set(pending_frame["run_key"].astype(str).tolist())

    _append_terminal_log(
        preflight.terminal_log_path,
        (
            "starting model fit for "
            f"{len(pending_fold_plans)} folds and {len(pending_keys)} overlay evaluations"
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
        for overlay_spec in plan.overlay_specs:
            run_key = build_allocation_overlay_run_key(
                fold_plan=fold_plan,
                overlay_spec=overlay_spec,
                feature_set_version=plan.feature_set_version,
                transaction_cost_bps=plan.transaction_cost_bps,
                min_history_rows=plan.min_history_rows,
            )
            if run_key not in pending_keys:
                continue
            try:
                overlay_signals = apply_allocation_overlay(
                    base_signals,
                    overlay_spec,
                    top_k=TOP_K,
                )
                backtest = run_signal_backtest(
                    overlay_signals,
                    SignalBacktestConfig(
                        initial_equity=1.0,
                        transaction_cost_bps=plan.transaction_cost_bps,
                    ),
                )
                row = _build_result_row(
                    run_key=run_key,
                    plan=plan,
                    fold_plan=fold_plan,
                    overlay_spec=overlay_spec,
                    overlay_signals=overlay_signals,
                    backtest_equity=backtest.equity_curve,
                    raw_df=raw_df,
                    feature_count=feature_count_by_fold.get(fold_result.fold_name, np.nan),
                )
            except Exception as exc:  # noqa: BLE001 - preserve remaining overlay work.
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
            if completed_now == 1 or completed_now % len(plan.overlay_specs) == 0:
                _append_terminal_log(
                    preflight.terminal_log_path,
                    f"completed_overlay_evaluations_this_run={completed_now}",
                )

    return _read_results(preflight.results_csv_path, result_columns)


def build_allocation_overlay_run_key(
    *,
    fold_plan: AllocationOverlayFoldPlan,
    overlay_spec: AllocationOverlaySpec,
    feature_set_version: str,
    transaction_cost_bps: float,
    min_history_rows: int,
) -> str:
    """Return a deterministic key for one fold/overlay evaluation."""

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
        "overlay": {
            "name": overlay_spec.name,
            "kind": overlay_spec.kind,
            "rank_weights": overlay_spec.rank_weights,
            "max_weight": overlay_spec.max_weight,
            "core_weights": overlay_spec.core_weights,
            "satellite_weight": overlay_spec.satellite_weight,
            "regime_columns": overlay_spec.regime_columns,
        },
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
            f"overlay-{_slug(overlay_spec.name)}",
            f"ridge-top{TOP_K}",
            f"hold-{MIN_HOLDING_DAYS}d",
            f"margin-{SWITCH_MARGIN:g}",
            f"cost-{transaction_cost_bps:g}bps",
            f"features-{_slug(feature_set_version)}",
            digest,
        ]
    )


def _build_preflight_rows(plan: TiingoV3AllocationOverlayStudyPlan) -> list[dict[str, object]]:
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
        for overlay_spec in plan.overlay_specs:
            run_key = build_allocation_overlay_run_key(
                fold_plan=fold_plan,
                overlay_spec=overlay_spec,
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
                    "overlay_name": overlay_spec.name,
                    "overlay_kind": overlay_spec.kind,
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


def _eligible_splits(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_cols: list[str],
    min_history_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = DynamicEligibilityConfig(min_history_rows=min_history_rows)
    return (
        apply_dynamic_universe_eligibility(
            train_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            require_target=True,
            config=config,
            history_df=dataset,
        ),
        apply_dynamic_universe_eligibility(
            validation_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            require_target=False,
            config=config,
            history_df=dataset,
        ),
        apply_dynamic_universe_eligibility(
            test_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            require_target=False,
            config=config,
            history_df=dataset,
        ),
    )


def _preflight_counts(
    *,
    selected_feature_cols: list[str],
    train_eligible: pd.DataFrame,
    validation_eligible: pd.DataFrame,
    test_eligible: pd.DataFrame,
) -> dict[str, int]:
    min_symbols = 0
    if not test_eligible.empty:
        min_symbols = int(test_eligible.groupby("date")["symbol"].nunique().min())
    return {
        "selected_feature_count": len(selected_feature_cols),
        "train_eligible_row_count": int(len(train_eligible)),
        "validation_eligible_row_count": int(len(validation_eligible)),
        "test_eligible_row_count": int(len(test_eligible)),
        "train_symbol_count": int(train_eligible["symbol"].nunique()) if not train_eligible.empty else 0,
        "validation_symbol_count": (
            int(validation_eligible["symbol"].nunique()) if not validation_eligible.empty else 0
        ),
        "test_symbol_count": int(test_eligible["symbol"].nunique()) if not test_eligible.empty else 0,
        "min_symbols_per_test_date": min_symbols,
    }


def _invalid_reason(
    *,
    counts: dict[str, int],
    supports_top_k: bool,
    min_train_eligible_rows: int,
) -> str:
    reasons: list[str] = []
    if counts["selected_feature_count"] <= 0:
        reasons.append("no selected feature columns")
    if counts["train_eligible_row_count"] <= 0:
        reasons.append("no train eligible rows")
    if counts["validation_eligible_row_count"] <= 0:
        reasons.append("no validation eligible rows")
    if counts["test_eligible_row_count"] <= 0:
        reasons.append("no test eligible rows")
    if counts["train_symbol_count"] <= 0:
        reasons.append("no train symbols")
    if counts["validation_symbol_count"] <= 0:
        reasons.append("no validation symbols")
    if counts["test_symbol_count"] <= 0:
        reasons.append("no test symbols")
    if not supports_top_k:
        reasons.append("test fold does not support top_k")
    if counts["train_eligible_row_count"] < min_train_eligible_rows:
        reasons.append(
            "train_eligible_row_count below min_train_eligible_rows: "
            f"{counts['train_eligible_row_count']} < {min_train_eligible_rows}"
        )
    return ";".join(reasons)


def _baseline_experiment_config(
    plan: TiingoV3AllocationOverlayStudyPlan,
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


def _generate_smoothed_top5_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    signal_config = TopKSignalConfig(
        prediction_col="prediction",
        top_k=TOP_K,
        allocation_mode=ALLOCATION_MODE,
    )
    smoothing_config = SignalSmoothingConfig(
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
    )
    signals = generate_top_k_signals(predictions, signal_config)
    return smooth_top_k_signals(
        signals,
        smoothing_config,
        allocation_config=signal_config,
    )


def _ensure_benchmark_signal_rows(
    signals: pd.DataFrame,
    *,
    raw_df: pd.DataFrame,
    fold: WalkForwardFold,
) -> pd.DataFrame:
    """Add zero-weight SPY/QQQ rows when eligibility filtered them out."""

    if signals.empty:
        return signals
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"])
    existing = {
        (pd.Timestamp(row.date), str(row.symbol))
        for row in out[["date", "symbol"]].itertuples(index=False)
    }
    signal_dates = set(pd.to_datetime(out["date"]).dropna())
    raw_prices = raw_df.loc[
        (raw_df["symbol"].astype(str).isin(["SPY", "QQQ"]))
        & (pd.to_datetime(raw_df["date"]) >= fold.test_start)
        & (pd.to_datetime(raw_df["date"]) <= fold.test_end),
        ["date", "symbol", "close"],
    ].copy()
    if raw_prices.empty:
        return out
    raw_prices["date"] = pd.to_datetime(raw_prices["date"])
    raw_prices = raw_prices.loc[raw_prices["date"].isin(signal_dates)]
    raw_prices = raw_prices.drop_duplicates(subset=["date", "symbol"], keep="last")
    rows: list[dict[str, object]] = []
    for row in raw_prices.itertuples(index=False):
        key = (pd.Timestamp(row.date), str(row.symbol))
        if key in existing:
            continue
        new_row = {column: np.nan for column in out.columns}
        new_row["date"] = pd.Timestamp(row.date)
        new_row["symbol"] = str(row.symbol)
        new_row["close"] = float(row.close)
        new_row["prediction"] = np.nan
        new_row["selected"] = False
        new_row["signal_rank"] = pd.NA
        new_row["target_weight"] = 0.0
        rows.append(new_row)
    if not rows:
        return out
    return (
        pd.concat([out, pd.DataFrame(rows, columns=out.columns)], ignore_index=True)
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_result_row(
    *,
    run_key: str,
    plan: TiingoV3AllocationOverlayStudyPlan,
    fold_plan: AllocationOverlayFoldPlan,
    overlay_spec: AllocationOverlaySpec,
    overlay_signals: pd.DataFrame,
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
    weight_summary = _overlay_weight_summary(overlay_signals)
    return {
        "run_key": run_key,
        "dataset_source": DATASET_SOURCE_LABEL,
        "study_stage": STUDY_STAGE,
        "run_name": plan.run_name,
        "overlay_name": overlay_spec.name,
        "overlay_kind": overlay_spec.kind,
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
        "overlay_weight_valid": validate_overlay_weights(overlay_signals),
        "spy_sharpe_like": spy_metrics["sharpe_like"],
        "qqq_sharpe_like": qqq_metrics["sharpe_like"],
        "spy_max_drawdown": spy_metrics["max_drawdown"],
        "qqq_max_drawdown": qqq_metrics["max_drawdown"],
    }


def _benchmark_metrics(
    raw_df: pd.DataFrame,
    symbol: str,
    fold: WalkForwardFold,
) -> dict[str, float]:
    frame = raw_df.loc[
        (raw_df["symbol"].astype(str) == symbol)
        & (pd.to_datetime(raw_df["date"]) >= fold.test_start)
        & (pd.to_datetime(raw_df["date"]) <= fold.test_end),
        ["date", "close"],
    ].copy()
    if frame.empty:
        return {"total_return": np.nan, "sharpe_like": np.nan, "max_drawdown": np.nan}
    frame["date"] = pd.to_datetime(frame["date"])
    frame = (
        frame.sort_values("date", kind="mergesort")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    frame["daily_return"] = pd.to_numeric(frame["close"], errors="coerce").pct_change().fillna(0.0)
    frame["equity"] = (1.0 + frame["daily_return"]).cumprod()
    return _equity_metrics(frame, return_col="daily_return", equity_col="equity")


def _equity_metrics(
    frame: pd.DataFrame,
    *,
    return_col: str,
    equity_col: str,
) -> dict[str, float]:
    if frame.empty:
        return {"total_return": np.nan, "sharpe_like": np.nan, "max_drawdown": np.nan}
    returns = pd.to_numeric(frame[return_col], errors="coerce")
    equity = pd.to_numeric(frame[equity_col], errors="coerce")
    final_equity = float(equity.iloc[-1])
    volatility = float(returns.std())
    mean_return = float(returns.mean())
    return {
        "total_return": final_equity - 1.0,
        "sharpe_like": (
            mean_return / volatility
            if volatility > 0.0 and not np.isnan(volatility)
            else np.nan
        ),
        "max_drawdown": calculate_max_drawdown(equity),
    }


def _overlay_weight_summary(overlay_signals: pd.DataFrame) -> dict[str, float]:
    signals = overlay_signals.copy()
    signals["selected"] = signals["selected"].fillna(False).astype(bool)
    grouped = (
        signals.loc[signals["selected"]]
        .groupby("date", sort=False)
        .agg(
            selected_symbol_count=("symbol", "nunique"),
            selected_weight_sum=("target_weight", "sum"),
        )
    )
    if grouped.empty:
        return {
            "selected_symbol_count_mean": np.nan,
            "selected_weight_sum_min": np.nan,
            "selected_weight_sum_max": np.nan,
        }
    return {
        "selected_symbol_count_mean": float(grouped["selected_symbol_count"].mean()),
        "selected_weight_sum_min": float(grouped["selected_weight_sum"].min()),
        "selected_weight_sum_max": float(grouped["selected_weight_sum"].max()),
    }


def _render_preflight_markdown(
    *,
    frame: pd.DataFrame,
    plan: TiingoV3AllocationOverlayStudyPlan,
    preflight: TiingoV3AllocationOverlayPreflight,
) -> str:
    overlay_summary = (
        frame.groupby("overlay_name", sort=False)
        .agg(
            overlay_kind=("overlay_kind", "first"),
            config_count=("run_key", "nunique"),
            valid_config_count=("valid_config", "sum"),
        )
        .reset_index()
        if not frame.empty
        else pd.DataFrame(columns=["overlay_name", "overlay_kind", "config_count", "valid_config_count"])
    )
    invalid_preview = frame.loc[~frame["valid_config"]].head(20)
    lines = [
        "# Tiingo V3 Allocation Overlay Study Preflight",
        "",
        "Local-only diagnostic before the allocation/risk overlay study.",
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
        f"- Top K: `{TOP_K}`",
        f"- Allocation baseline: `{ALLOCATION_MODE}`",
        f"- Minimum holding days: `{MIN_HOLDING_DAYS}`",
        f"- Switch margin: `{SWITCH_MARGIN:g}`",
        f"- Transaction cost bps: `{plan.transaction_cost_bps:g}`",
        f"- Target: `{TARGET_COL}`",
        f"- Label horizon days: `{LABEL_HORIZON_DAYS}`",
        "",
        "## Overlay Counts",
        "",
        f"- Generated fold count: `{preflight.fold_count}`",
        f"- Overlay count: `{preflight.overlay_count}`",
        f"- Total overlay evaluations: `{preflight.total_overlay_evaluation_count}`",
        f"- Valid overlay evaluations: `{preflight.valid_overlay_evaluation_count}`",
        f"- Invalid overlay evaluations: `{preflight.invalid_overlay_evaluation_count}`",
        f"- Completed overlay evaluations from checkpoint: `{preflight.completed_checkpoint_count}`",
        f"- Failed checkpoint count if RESUME=1: `{preflight.failed_checkpoint_count}`",
        f"- Pending overlay evaluations: `{preflight.pending_overlay_evaluation_count}`",
        "",
        "## Overlay Families Tested",
        "",
        _markdown_table(
            overlay_summary,
            ["overlay_name", "overlay_kind", "config_count", "valid_config_count"],
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
        "- Model fitting is shared once per pending fold; overlays reuse the same smoothed top-5 signals.",
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
                "overlay_name",
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
        "`tree_screen`, MLP, feature v4, S3 readers, and production signal export are intentionally not part of this pass.",
        "",
    ]
    return "\n".join(lines)


def _print_plan(
    *,
    plan: TiingoV3AllocationOverlayStudyPlan,
    preflight: TiingoV3AllocationOverlayPreflight,
    dry_run: bool,
) -> None:
    print("US Market ML Tiingo-only benchmark-gap v3 allocation overlay study")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("uses GPU: false")
    print("includes neural networks/MLP: false")
    print("includes tree models: false")
    print("exports production signals: false")
    print(f"dry_run: {dry_run}")
    print(f"dataset glob: {plan.market_dataset_glob}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"start date: {plan.start_date}")
    print(f"end date: {plan.end_date}")
    print(f"feature_set_version: {plan.feature_set_version}")
    print(f"model_type: {MODEL_TYPE}")
    print(f"top_k: {TOP_K}")
    print(f"allocation_mode baseline: {ALLOCATION_MODE}")
    print(f"min_holding_days: {MIN_HOLDING_DAYS}")
    print(f"switch_margin: {SWITCH_MARGIN:g}")
    print(f"transaction_cost_bps: {plan.transaction_cost_bps:g}")
    print(f"universe_mode: {UNIVERSE_MODE}")
    print(f"feature_selection_scope: {FEATURE_SELECTION_SCOPE}")
    print(f"target_col: {TARGET_COL}")
    print(f"label_horizon_days: {LABEL_HORIZON_DAYS}")
    print(f"generated fold count: {preflight.fold_count}")
    print(f"overlay count: {preflight.overlay_count}")
    print(f"total overlay evaluations: {preflight.total_overlay_evaluation_count}")
    print(f"valid overlay evaluations: {preflight.valid_overlay_evaluation_count}")
    print(f"pending overlay evaluations: {preflight.pending_overlay_evaluation_count}")
    print("overlay names:")
    for overlay_spec in plan.overlay_specs:
        print(f"  - {overlay_spec.name}")
    print(f"output directory: {plan.output_dir}")
    print(f"results CSV path: {preflight.results_csv_path}")
    print(f"preflight CSV path: {preflight.csv_path}")
    print(f"preflight Markdown path: {preflight.markdown_path}")
    print(f"completed checkpoint path: {preflight.completed_keys_path}")
    print(f"failed checkpoint path: {preflight.failed_keys_path}")
    print(f"terminal/progress log path: {preflight.terminal_log_path}")
    print(f"max_workers argument accepted: {plan.max_workers}")
    print("model fit sharing: one ridge fit per pending fold; overlays reuse smoothed signals")
    print("tree_screen: not run")
    print("MLP: not run")
    print("feature v4: not added")
    print("production signal export: not run")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(plan)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Tiingo-only benchmark-gap v3 allocation overlay study."
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
        "--include-optional-qqq-floor",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("INCLUDE_OPTIONAL_QQQ_FLOOR", False),
    )
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
    parser.add_argument("--train-window-years", nargs="+", default=_env_values("TRAIN_WINDOW_YEARS"))
    parser.add_argument(
        "--validation-window-months",
        nargs="+",
        default=_env_values("VALIDATION_WINDOW_MONTHS"),
    )
    parser.add_argument("--test-window-months", nargs="+", default=_env_values("TEST_WINDOW_MONTHS"))
    parser.add_argument(
        "--feature-set-version",
        default=os.environ.get("FEATURE_SET_VERSION", BENCHMARK_GAP_V3_FEATURE_SET_VERSION),
    )
    parser.add_argument("--transaction-cost-bps", default=os.environ.get("TRANSACTION_COST_BPS"))
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
            "Tiingo-only allocation overlay study."
        )
    if str(args.run_name).strip() == "":
        args.run_name = DEFAULT_RUN_NAME
    return args


def _equivalent_command(plan: TiingoV3AllocationOverlayStudyPlan) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_tiingo_v3_allocation_overlay_study",
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


def _quick_overlay_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby("overlay_name", sort=False)
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


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    lines = [
        "| " + " | ".join(visible.columns) + " |",
        "| " + " | ".join("---" for _ in visible.columns) + " |",
    ]
    for _, row in visible.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in visible.columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _ensure_results_file(path: Path, result_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=result_columns).to_csv(path, index=False)


def _append_results(path: Path, rows: pd.DataFrame, result_columns: list[str]) -> None:
    header = not path.exists() or path.stat().st_size == 0
    rows.loc[:, result_columns].to_csv(path, mode="a", header=header, index=False)


def _read_results(path: Path, result_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=result_columns)
    frame = pd.read_csv(path)
    for column in result_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.loc[:, result_columns]


def _read_key_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().split(",", maxsplit=1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _append_line(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{value}\n")


def _append_failed_key(path: Path, key: str, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    message = f"{key},{type(exc).__name__},{str(exc).replace(os.linesep, ' ')}"
    _append_line(path, message)


def _append_terminal_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}\n")


def _non_empty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _mean_or_nan(values: pd.Series | None) -> float:
    if values is None:
        return np.nan
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if not numeric.empty else np.nan


def _sum_or_nan(values: pd.Series | None) -> float:
    if values is None:
        return np.nan
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.sum()) if not numeric.empty else np.nan


def _max_or_nan(values: pd.Series | None) -> float:
    if values is None:
        return np.nan
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if not numeric.empty else np.nan


def _values_arg(name: str, values: Sequence[object]) -> list[str]:
    return [name, *[str(value) for value in values]]


def _unique_train_windows(plan: TiingoV3AllocationOverlayStudyPlan) -> list[int]:
    return sorted({fold_plan.train_window_years for fold_plan in plan.fold_plans})


def _unique_validation_windows(plan: TiingoV3AllocationOverlayStudyPlan) -> list[int]:
    return sorted({fold_plan.validation_window_months for fold_plan in plan.fold_plans})


def _unique_test_windows(plan: TiingoV3AllocationOverlayStudyPlan) -> list[int]:
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


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


if __name__ == "__main__":
    raise SystemExit(run_tiingo_v3_allocation_overlay_study_from_args())
