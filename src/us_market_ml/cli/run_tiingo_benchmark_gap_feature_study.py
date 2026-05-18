"""Run the Tiingo-only benchmark-gap feature engineering study."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_market_ml.cli.run_dynamic_extended_history_study import (
    _eligible_preflight_splits,
    _preflight_fold_counts,
    _read_key_checkpoint,
    _selected_preflight_features,
    _split_for_preflight,
    _to_walk_forward_fold,
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
    PRICE_MODE,
    RANDOM_STATE,
    TARGET_COL,
    UNIVERSE_MODE,
    _clear_fresh_run_checkpoints,
    _ensure_dataset_available,
    _format_command,
    _install_interrupt_handlers,
    _markdown_table,
    _non_empty_line_count,
    _parse_date,
    _parse_max_workers,
    _parse_non_negative_float_values,
    _parse_positive_int,
    _parse_positive_int_values,
    _tiingo_invalid_reason,
    _to_sweep_fold_config,
)
from us_market_ml.data import (
    SupervisedDatasetConfig,
    build_supervised_dataset,
    get_feature_columns,
    read_local_market_data,
)
from us_market_ml.experiments import (
    SweepFoldConfig,
    SweepModelConfig,
    SweepRunConfig,
    SweepSignalConfig,
    SweepUniverseConfig,
    build_sweep_run_key,
    run_sweep,
)
from us_market_ml.features import (
    BENCHMARK_GAP_FEATURE_SET_VERSION,
    DEFAULT_FEATURE_SET_VERSION,
    validate_feature_set_version,
)
from us_market_ml.splits import make_sliding_window_folds

DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_benchmark_gap_feature_study")
DEFAULT_RUN_NAME = "tiingo_benchmark_gap_feature_study"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_STEP_MONTHS = 3
DEFAULT_TRAIN_WINDOW_YEARS = (3, 5)
DEFAULT_VALIDATION_WINDOW_MONTHS = (12,)
DEFAULT_TEST_WINDOW_MONTHS = (3, 6, 12)
DEFAULT_TRANSACTION_COST_BPS_VALUES = (10.0,)
DEFAULT_FEATURE_SET_VERSIONS = (
    DEFAULT_FEATURE_SET_VERSION,
    BENCHMARK_GAP_FEATURE_SET_VERSION,
)

SWEEP_PROFILE = "benchmark_gap_feature_study"
UNIVERSE_NAME = "tiingo_benchmark_gap_feature_dynamic"
MODEL_FAMILY = "baseline"
SKIPPED_KEYS_SUFFIX = "_skipped_keys.txt"

PREFLIGHT_COLUMNS = [
    "dataset_source",
    "sweep_profile",
    "run_key",
    "universe",
    "fold_name",
    "candidate_family",
    "candidate_role",
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


@dataclass(frozen=True)
class BenchmarkGapCandidateSpec:
    """One exact model/signal candidate in the feature study."""

    candidate_family: str
    candidate_role: str
    model_type: str
    top_k: int
    min_holding_days: int
    switch_margin: float

    @property
    def label(self) -> str:
        return (
            f"{self.model_type} / top{self.top_k} / hold{self.min_holding_days} / "
            f"margin{self.switch_margin:g} / cost10"
        )


SELECTED_CANDIDATE_SPEC = BenchmarkGapCandidateSpec(
    candidate_family="selected_ridge_top5",
    candidate_role="selected_candidate",
    model_type="ridge",
    top_k=5,
    min_holding_days=20,
    switch_margin=0.005,
)
RETURN_REFERENCE_SPEC = BenchmarkGapCandidateSpec(
    candidate_family="ridge_top1_return_reference",
    candidate_role="return_reference",
    model_type="ridge",
    top_k=1,
    min_holding_days=20,
    switch_margin=0.005,
)
HISTORICAL_REFERENCE_SPEC = BenchmarkGapCandidateSpec(
    candidate_family="historical_linear_top1",
    candidate_role="historical_reference",
    model_type="linear_regression",
    top_k=1,
    min_holding_days=10,
    switch_margin=0.002,
)
REFERENCE_CANDIDATE_SPECS = (RETURN_REFERENCE_SPEC, HISTORICAL_REFERENCE_SPEC)


@dataclass(frozen=True)
class BenchmarkGapFoldPlan:
    """A sweep fold plus the window metadata used to create it."""

    fold: SweepFoldConfig
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass(frozen=True)
class BenchmarkGapFeatureStudyPlan:
    """Executable Tiingo-only benchmark-gap feature study plan."""

    fold_plans: tuple[BenchmarkGapFoldPlan, ...]
    candidate_specs: tuple[BenchmarkGapCandidateSpec, ...]
    feature_set_versions: tuple[str, ...]
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
    transaction_cost_bps_values: tuple[float, ...]
    include_reference_candidates: bool

    @property
    def estimated_total_configs(self) -> int:
        return (
            len(self.fold_plans)
            * len(self.candidate_specs)
            * len(self.feature_set_versions)
            * len(self.transaction_cost_bps_values)
        )


@dataclass(frozen=True)
class BenchmarkGapFeatureStudyPreflight:
    """Diagnostic outputs generated before model training."""

    csv_path: Path
    markdown_path: Path
    skipped_keys_path: Path
    results_csv_path: Path
    completed_keys_path: Path
    failed_keys_path: Path
    terminal_log_path: Path
    total_config_count: int
    valid_config_count: int
    skipped_config_count: int
    completed_checkpoint_count: int
    failed_checkpoint_count: int
    pending_config_count: int


def run_tiingo_benchmark_gap_feature_study_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the Tiingo-only benchmark-gap feature study."""

    args = _parse_args(argv)
    _install_interrupt_handlers()
    _ensure_dataset_available(args.market_dataset_glob)

    fold_plans = build_tiingo_benchmark_gap_feature_study_folds(
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_years_values=args.train_window_years,
        validation_window_months_values=args.validation_window_months,
        test_window_months_values=args.test_window_months,
        step_months=args.step_months,
    )
    if not fold_plans:
        raise SystemExit("No Tiingo benchmark-gap feature study folds were generated.")

    plan = build_tiingo_benchmark_gap_feature_study_plan(
        market_dataset_glob=args.market_dataset_glob,
        fold_plans=fold_plans,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
        step_months=args.step_months,
        transaction_cost_bps_values=args.transaction_cost_bps_values,
        feature_set_versions=args.feature_set_versions,
        include_reference_candidates=args.include_reference_candidates,
    )
    preflight = preflight_tiingo_benchmark_gap_feature_study(plan)

    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    if not args.resume:
        _clear_fresh_run_checkpoints(_sweep_config_for_candidate(plan, plan.candidate_specs[0]))

    try:
        for candidate_spec in plan.candidate_specs:
            config = _sweep_config_for_candidate(plan, candidate_spec, resume=True)
            run_sweep(config)
    except KeyboardInterrupt:
        print("Interrupt received; Tiingo benchmark-gap feature study stopped cleanly.")
        return 130
    return 0


def build_tiingo_benchmark_gap_feature_study_folds(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    train_window_years_values: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months_values: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months_values: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[BenchmarkGapFoldPlan]:
    """Build the benchmark-gap feature-study rolling window grid."""

    fold_plans: list[BenchmarkGapFoldPlan] = []
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
                        BenchmarkGapFoldPlan(
                            fold=_to_sweep_fold_config(fold),
                            train_window_years=train_window_years,
                            validation_window_months=validation_window_months,
                            test_window_months=test_window_months,
                        )
                    )
    return fold_plans


def build_tiingo_benchmark_gap_feature_study_plan(
    *,
    market_dataset_glob: str,
    fold_plans: Sequence[BenchmarkGapFoldPlan],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
    min_train_eligible_rows: int = DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    step_months: int = DEFAULT_STEP_MONTHS,
    transaction_cost_bps_values: Sequence[float] = DEFAULT_TRANSACTION_COST_BPS_VALUES,
    feature_set_versions: Sequence[str] = DEFAULT_FEATURE_SET_VERSIONS,
    include_reference_candidates: bool = True,
    candidate_specs: Sequence[BenchmarkGapCandidateSpec] | None = None,
) -> BenchmarkGapFeatureStudyPlan:
    """Build the compact Tiingo-only benchmark-gap feature study plan."""

    if not fold_plans:
        raise ValueError("At least one fold plan is required.")
    if not transaction_cost_bps_values:
        raise ValueError("At least one transaction cost value is required.")
    if any(value < 0.0 for value in transaction_cost_bps_values):
        raise ValueError("transaction_cost_bps values must be greater than or equal to 0.")
    for feature_set_version in feature_set_versions:
        validate_feature_set_version(feature_set_version)

    resolved_candidate_specs = (
        tuple(candidate_specs)
        if candidate_specs is not None
        else _candidate_specs(include_reference_candidates)
    )
    if not resolved_candidate_specs:
        raise ValueError("At least one candidate spec is required.")

    return BenchmarkGapFeatureStudyPlan(
        fold_plans=tuple(fold_plans),
        candidate_specs=resolved_candidate_specs,
        feature_set_versions=tuple(dict.fromkeys(feature_set_versions)),
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
        transaction_cost_bps_values=tuple(transaction_cost_bps_values),
        include_reference_candidates=include_reference_candidates,
    )


def preflight_tiingo_benchmark_gap_feature_study(
    plan: BenchmarkGapFeatureStudyPlan,
) -> BenchmarkGapFeatureStudyPreflight:
    """Write fold/config diagnostics and seed skipped invalid keys."""

    output_dir = Path(plan.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{plan.run_name}_preflight.csv"
    markdown_path = output_dir / f"{plan.run_name}_preflight.md"
    skipped_keys_path = output_dir / f"{plan.run_name}{SKIPPED_KEYS_SUFFIX}"
    results_csv_path = output_dir / f"{plan.run_name}_results.csv"
    completed_keys_path = output_dir / f"{plan.run_name}_completed_keys.txt"
    failed_keys_path = output_dir / f"{plan.run_name}_failed_keys.txt"
    terminal_log_path = output_dir / f"{plan.run_name}_terminal.log"

    rows = _build_preflight_rows(plan)
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    frame.to_csv(csv_path, index=False)

    invalid_keys = set(frame.loc[~frame["valid_config"], "run_key"].astype(str).tolist())
    existing_skipped = _read_key_checkpoint(skipped_keys_path) if plan.resume else set()
    skipped_keys = sorted(existing_skipped.union(invalid_keys))
    skipped_keys_path.write_text(
        "".join(f"{key}\n" for key in skipped_keys),
        encoding="utf-8",
    )

    valid_keys = set(frame.loc[frame["valid_config"], "run_key"].astype(str).tolist())
    completed_keys = _read_key_checkpoint(completed_keys_path) if plan.resume else set()
    completed_in_plan = valid_keys.intersection(completed_keys)
    failed_count = _non_empty_line_count(failed_keys_path) if plan.resume else 0
    pending_count = max(0, len(valid_keys) - len(completed_in_plan))

    preflight = BenchmarkGapFeatureStudyPreflight(
        csv_path=csv_path,
        markdown_path=markdown_path,
        skipped_keys_path=skipped_keys_path,
        results_csv_path=results_csv_path,
        completed_keys_path=completed_keys_path,
        failed_keys_path=failed_keys_path,
        terminal_log_path=terminal_log_path,
        total_config_count=int(len(frame)),
        valid_config_count=int(frame["valid_config"].sum()) if not frame.empty else 0,
        skipped_config_count=int((~frame["valid_config"]).sum()) if not frame.empty else 0,
        completed_checkpoint_count=len(completed_in_plan),
        failed_checkpoint_count=failed_count,
        pending_config_count=pending_count,
    )
    markdown_path.write_text(
        _render_preflight_markdown(frame=frame, plan=plan, preflight=preflight),
        encoding="utf-8",
    )
    return preflight


def _build_preflight_rows(plan: BenchmarkGapFeatureStudyPlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_df = read_local_market_data(plan.market_dataset_glob)
    base_config = _sweep_config_for_candidate(plan, plan.candidate_specs[0])
    universe = base_config.universes[0]

    for feature_set_version in plan.feature_set_versions:
        dataset = build_supervised_dataset(
            raw_df,
            SupervisedDatasetConfig(
                forward_horizons=(5, 10, LABEL_HORIZON_DAYS),
                benchmark_symbol="SPY",
                excess_return_horizons=(LABEL_HORIZON_DAYS,),
                rank_target_col=TARGET_COL,
                rank_col=f"rank_{TARGET_COL}",
                feature_set_version=feature_set_version,
            ),
        )
        if TARGET_COL not in dataset.columns:
            raise ValueError(
                "Configured target_col is missing after supervised dataset construction: "
                f"{TARGET_COL}"
            )
        candidate_feature_cols = get_feature_columns(
            dataset,
            feature_set_version=feature_set_version,
        )

        for fold_plan in plan.fold_plans:
            fold_config = fold_plan.fold
            fold = _to_walk_forward_fold(fold_config)
            train_df, validation_df, test_df = _split_for_preflight(
                dataset,
                fold,
                base_config,
            )
            selected_feature_cols = _selected_preflight_features(
                train_df=train_df,
                candidate_feature_cols=candidate_feature_cols,
                config=base_config,
            )
            train_eligible, validation_eligible, test_eligible = (
                _eligible_preflight_splits(
                    train_df=train_df,
                    validation_df=validation_df,
                    test_df=test_df,
                    eligibility_history_df=dataset,
                    feature_cols=selected_feature_cols,
                    config=base_config,
                )
            )
            fold_counts = _preflight_fold_counts(
                selected_feature_cols=selected_feature_cols,
                train_eligible=train_eligible,
                validation_eligible=validation_eligible,
                test_eligible=test_eligible,
            )

            for candidate_spec in plan.candidate_specs:
                model_config = _model_config(candidate_spec)
                signal_config = _signal_config(candidate_spec)
                supports_top_k = fold_counts["min_symbols_per_test_date"] >= candidate_spec.top_k
                invalid_reason = _tiingo_invalid_reason(
                    selected_feature_count=fold_counts["selected_feature_count"],
                    train_eligible_row_count=fold_counts["train_eligible_row_count"],
                    validation_eligible_row_count=fold_counts["validation_eligible_row_count"],
                    test_eligible_row_count=fold_counts["test_eligible_row_count"],
                    train_symbol_count=fold_counts["train_symbol_count"],
                    validation_symbol_count=fold_counts["validation_symbol_count"],
                    test_symbol_count=fold_counts["test_symbol_count"],
                    supports_top_k=supports_top_k,
                    min_train_eligible_rows=plan.min_train_eligible_rows,
                )
                for transaction_cost_bps in plan.transaction_cost_bps_values:
                    run_key = build_sweep_run_key(
                        universe=universe,
                        fold=fold_config,
                        model=model_config,
                        signal=signal_config,
                        label_horizon_days=LABEL_HORIZON_DAYS,
                        min_holding_days=candidate_spec.min_holding_days,
                        switch_margin=candidate_spec.switch_margin,
                        transaction_cost_bps=transaction_cost_bps,
                        feature_set_version=feature_set_version,
                        universe_mode=UNIVERSE_MODE,
                        min_history_rows=plan.min_history_rows,
                        feature_selection_scope=FEATURE_SELECTION_SCOPE,
                        min_feature_non_null_ratio=0.0,
                    )
                    rows.append(
                        {
                            "dataset_source": DATASET_SOURCE_LABEL,
                            "sweep_profile": SWEEP_PROFILE,
                            "run_key": run_key,
                            "universe": universe.name,
                            "fold_name": fold_config.name,
                            "candidate_family": candidate_spec.candidate_family,
                            "candidate_role": candidate_spec.candidate_role,
                            "feature_set_version": feature_set_version,
                            "train_window_years": fold_plan.train_window_years,
                            "validation_window_months": (
                                fold_plan.validation_window_months
                            ),
                            "test_window_months": fold_plan.test_window_months,
                            "train_start": fold_config.train_start,
                            "train_end": fold_config.train_end,
                            "validation_start": fold_config.validation_start,
                            "validation_end": fold_config.validation_end,
                            "test_start": fold_config.test_start,
                            "test_end": fold_config.test_end,
                            "model_family": MODEL_FAMILY,
                            "model_type": candidate_spec.model_type,
                            "top_k": candidate_spec.top_k,
                            "allocation_mode": ALLOCATION_MODE,
                            "min_holding_days": candidate_spec.min_holding_days,
                            "switch_margin": candidate_spec.switch_margin,
                            "transaction_cost_bps": transaction_cost_bps,
                            "min_train_eligible_rows": plan.min_train_eligible_rows,
                            **fold_counts,
                            "supports_top_k": supports_top_k,
                            "valid_config": invalid_reason == "",
                            "invalid_reason": invalid_reason,
                        }
                    )
    return rows


def _render_preflight_markdown(
    *,
    frame: pd.DataFrame,
    plan: BenchmarkGapFeatureStudyPlan,
    preflight: BenchmarkGapFeatureStudyPreflight,
) -> str:
    invalid_preview = frame.loc[~frame["valid_config"]].head(20)
    feature_summary = _feature_set_preflight_summary(frame)
    candidate_summary = _candidate_preflight_summary(frame)
    fold_summary = _fold_summary(frame)

    lines = [
        "# Tiingo Benchmark-Gap Feature Study Preflight",
        "",
        "Local-only diagnostic before compact feature-engineering confirmation.",
        "",
        "## Dataset",
        "",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        "- Source provider: `tiingo`",
        f"- Price mode: `{PRICE_MODE}`",
        f"- Dataset path/glob: `{plan.market_dataset_glob}`",
        f"- Research start date: `{plan.start_date}`",
        f"- Research end date: `{plan.end_date}`",
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
        "## Feature Comparison",
        "",
        f"- Feature set versions: `{list(plan.feature_set_versions)}`",
        f"- Baseline feature set: `{DEFAULT_FEATURE_SET_VERSION}`",
        f"- Candidate feature set: `{BENCHMARK_GAP_FEATURE_SET_VERSION}`",
        "",
        _markdown_table(
            feature_summary,
            [
                "feature_set_version",
                "config_count",
                "valid_config_count",
                "invalid_config_count",
                "selected_feature_count_min",
                "selected_feature_count_max",
            ],
        ),
        "",
        "## Candidate Specs",
        "",
        f"- Candidate spec count: `{len(plan.candidate_specs)}`",
        f"- Include reference candidates: `{plan.include_reference_candidates}`",
        f"- Transaction costs bps: `{_transaction_cost_bps_values(plan)}`",
        "",
        _markdown_table(
            candidate_summary,
            [
                "candidate_family",
                "candidate_role",
                "model_type",
                "top_k",
                "min_holding_days",
                "switch_margin",
                "config_count",
                "valid_config_count",
                "invalid_config_count",
            ],
        ),
        "",
        "## Research Controls",
        "",
        f"- Universe mode: `{UNIVERSE_MODE}`",
        f"- Feature selection scope: `{FEATURE_SELECTION_SCOPE}`",
        f"- Minimum history rows: `{plan.min_history_rows}`",
        f"- Minimum train eligible rows: `{plan.min_train_eligible_rows}`",
        f"- Target: `{TARGET_COL}`",
        f"- Label horizon days: `{LABEL_HORIZON_DAYS}`",
        f"- Allocation mode: `{ALLOCATION_MODE}`",
        "- Local-only: `true`",
        "- Uses S3/cloud/GPU/neural networks/tree models: `false`",
        "- Production signal export: `false`",
        "",
        "## Counts",
        "",
        f"- Generated fold count: `{len(plan.fold_plans)}`",
        f"- Candidate spec count: `{len(plan.candidate_specs)}`",
        f"- Feature set count: `{len(plan.feature_set_versions)}`",
        f"- Total configs: `{preflight.total_config_count}`",
        f"- Valid configs: `{preflight.valid_config_count}`",
        f"- Skipped configs: `{preflight.skipped_config_count}`",
        f"- Completed configs from checkpoint: `{preflight.completed_checkpoint_count}`",
        f"- Failed configs from checkpoint: `{preflight.failed_checkpoint_count}`",
        f"- Pending configs: `{preflight.pending_config_count}`",
        "",
        "## Fold Summary",
        "",
        _markdown_table(
            fold_summary.head(40),
            [
                "fold_name",
                "train_window_years",
                "validation_window_months",
                "test_window_months",
                "selected_feature_count",
                "train_eligible_row_count",
                "validation_eligible_row_count",
                "test_eligible_row_count",
                "test_symbol_count",
                "min_symbols_per_test_date",
                "valid_config_count",
                "invalid_config_count",
            ],
        ),
        "",
        "## Invalid/Thin Fold Preview",
        "",
        _markdown_table(
            invalid_preview,
            [
                "fold_name",
                "feature_set_version",
                "candidate_role",
                "model_type",
                "top_k",
                "selected_feature_count",
                "train_eligible_row_count",
                "min_train_eligible_rows",
                "min_symbols_per_test_date",
                "invalid_reason",
            ],
        ),
        "",
        "## Ctrl+C / Resume Notes",
        "",
        "The shell runner forwards a single Ctrl+C to the Python study process, "
        "then escalates to SIGTERM and SIGKILL if needed. Completed, skipped, "
        "and failed checkpoint files are plain text and remain usable after interruption.",
        "",
        "Resume with `RESUME=1`; completed keys are not rerun, invalid/thin folds stay "
        "in the skipped checkpoint, and failed keys are preserved for inspection.",
        "",
        "## Deferral Notes",
        "",
        "`tree_screen`, MLP, and production signal export are intentionally not part "
        "of this feature-engineering pass.",
        "",
    ]
    return "\n".join(lines)


def _print_plan(
    *,
    plan: BenchmarkGapFeatureStudyPlan,
    preflight: BenchmarkGapFeatureStudyPreflight,
    dry_run: bool,
) -> None:
    print("US Market ML Tiingo-only benchmark-gap feature study")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("uses GPU: false")
    print("includes neural networks/MLP: false")
    print("includes tree models: false")
    print("exports production signals: false")
    print(f"dry_run: {dry_run}")
    print(f"sweep_profile: {SWEEP_PROFILE}")
    print(f"dataset glob: {plan.market_dataset_glob}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"price mode: {PRICE_MODE}")
    print(f"start date: {plan.start_date}")
    print(f"end date: {plan.end_date}")
    print(f"universe: {UNIVERSE_NAME}")
    print(f"universe_mode: {UNIVERSE_MODE}")
    print(f"feature_selection_scope: {FEATURE_SELECTION_SCOPE}")
    print(f"min_history_rows: {plan.min_history_rows}")
    print(f"min_train_eligible_rows: {plan.min_train_eligible_rows}")
    print(f"target_col: {TARGET_COL}")
    print(f"label_horizon_days: {LABEL_HORIZON_DAYS}")
    print(f"feature_set_versions: {list(plan.feature_set_versions)}")
    print(f"allocation_mode: {ALLOCATION_MODE}")
    print(f"run_name: {plan.run_name}")
    print(f"output directory: {plan.output_dir}")
    print(f"results CSV path: {preflight.results_csv_path}")
    print(f"completed checkpoint path: {preflight.completed_keys_path}")
    print(f"skipped checkpoint path: {preflight.skipped_keys_path}")
    print(f"failed checkpoint path: {preflight.failed_keys_path}")
    print(f"terminal/progress log path: {preflight.terminal_log_path}")
    print(f"preflight CSV path: {preflight.csv_path}")
    print(f"preflight Markdown path: {preflight.markdown_path}")
    print(f"total configs/runs: {preflight.total_config_count}")
    print(f"valid configs/runs: {preflight.valid_config_count}")
    print(f"skipped configs/runs: {preflight.skipped_config_count}")
    print(
        "already completed configs if RESUME=1: "
        f"{preflight.completed_checkpoint_count}"
    )
    print(f"failed checkpoint count if RESUME=1: {preflight.failed_checkpoint_count}")
    print(f"pending configs for this batch: {preflight.pending_config_count}")
    print(f"candidate spec count: {len(plan.candidate_specs)}")
    print(f"candidate roles: {', '.join(_candidate_roles(plan))}")
    print(f"model types: {', '.join(_model_types(plan))}")
    print(f"top_k values: {_top_k_values(plan)}")
    print(f"smoothing values: {_smoothing_values(plan)}")
    print(f"transaction cost values: {_transaction_cost_bps_values(plan)}")
    print(f"train_window_years values: {_unique_train_windows(plan)}")
    print(f"validation_window_months values: {_unique_validation_windows(plan)}")
    print(f"test_window_months values: {_unique_test_windows(plan)}")
    print(f"step_months: {plan.step_months}")
    print(f"max workers: {plan.max_workers}")
    print(f"resume: {plan.resume}")
    print("tree_screen: not run")
    print("MLP: not run")
    print("production signal export: not run")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(plan)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Tiingo-only benchmark-gap feature study."
    )
    parser.add_argument("--market-dataset-glob", default=_default_market_dataset_glob())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", DEFAULT_RUN_NAME))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--max-workers",
        type=_parse_max_workers,
        default=_parse_max_workers(os.environ.get("MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from local checkpoint files by default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-reference-candidates",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("INCLUDE_REFERENCE_CANDIDATES", True),
        help="Include ridge top1 and historical linear top1 reference rows by default.",
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
        "--transaction-cost-bps-values",
        nargs="+",
        default=_env_values("TRANSACTION_COST_BPS_VALUES"),
    )
    parser.add_argument(
        "--feature-set-versions",
        nargs="+",
        default=_env_values("FEATURE_SET_VERSIONS"),
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
        args.transaction_cost_bps_values = (
            _parse_non_negative_float_values(
                args.transaction_cost_bps_values,
                "--transaction-cost-bps-values",
            )
            if args.transaction_cost_bps_values is not None
            else DEFAULT_TRANSACTION_COST_BPS_VALUES
        )
        args.feature_set_versions = (
            _parse_feature_set_version_values(
                args.feature_set_versions,
                "--feature-set-versions",
            )
            if args.feature_set_versions is not None
            else DEFAULT_FEATURE_SET_VERSIONS
        )
        _parse_date(args.start_date, "--start-date")
        _parse_date(args.end_date, "--end-date")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if str(args.run_name).strip() == "":
        args.run_name = DEFAULT_RUN_NAME
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    if str(args.market_dataset_glob) == OLD_BRIDGED_DATASET_GLOB:
        parser.error(
            "The old Tiingo/Massive bridged dataset is not allowed for this "
            "Tiingo-only benchmark-gap feature study. Use the Tiingo-only adjusted "
            "dataset glob."
        )
    return args


def _sweep_config_for_candidate(
    plan: BenchmarkGapFeatureStudyPlan,
    candidate_spec: BenchmarkGapCandidateSpec,
    *,
    resume: bool | None = None,
) -> SweepRunConfig:
    return SweepRunConfig(
        universes=[
            SweepUniverseConfig(name=UNIVERSE_NAME, data_path=plan.market_dataset_glob)
        ],
        folds=[fold_plan.fold for fold_plan in plan.fold_plans],
        models=[_model_config(candidate_spec)],
        signals=[_signal_config(candidate_spec)],
        target_col=TARGET_COL,
        label_horizon_days=LABEL_HORIZON_DAYS,
        feature_set_versions=list(plan.feature_set_versions),
        min_holding_days=candidate_spec.min_holding_days,
        switch_margin=candidate_spec.switch_margin,
        transaction_cost_bps_values=list(plan.transaction_cost_bps_values),
        output_dir=plan.output_dir,
        run_name=plan.run_name,
        resume=plan.resume if resume is None else resume,
        max_workers=plan.max_workers,
        skipped_keys_path=plan.output_dir / f"{plan.run_name}{SKIPPED_KEYS_SUFFIX}",
        skip_no_usable_training_rows=True,
        universe_mode=UNIVERSE_MODE,
        min_history_rows=plan.min_history_rows,
        feature_selection_scope=FEATURE_SELECTION_SCOPE,
        min_feature_non_null_ratio=0.0,
        result_metadata={
            "sweep_profile": SWEEP_PROFILE,
            "candidate_family": candidate_spec.candidate_family,
            "candidate_role": candidate_spec.candidate_role,
            "study_stage": "feature_engineering",
        },
    )


def _model_config(candidate_spec: BenchmarkGapCandidateSpec) -> SweepModelConfig:
    return SweepModelConfig(
        model_family=MODEL_FAMILY,
        model_type=candidate_spec.model_type,
        random_state=RANDOM_STATE,
    )


def _signal_config(candidate_spec: BenchmarkGapCandidateSpec) -> SweepSignalConfig:
    return SweepSignalConfig(
        top_k=candidate_spec.top_k,
        min_holding_days=candidate_spec.min_holding_days,
        switch_margin=candidate_spec.switch_margin,
        allocation_mode=ALLOCATION_MODE,
    )


def _candidate_specs(include_reference_candidates: bool) -> tuple[BenchmarkGapCandidateSpec, ...]:
    if include_reference_candidates:
        return (SELECTED_CANDIDATE_SPEC, *REFERENCE_CANDIDATE_SPECS)
    return (SELECTED_CANDIDATE_SPEC,)


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


def _parse_feature_set_version_values(
    values: Sequence[str],
    parameter_name: str,
) -> tuple[str, ...]:
    parsed_values: list[str] = []
    for value in values:
        for part in str(value).split(","):
            stripped = part.strip()
            if not stripped:
                continue
            try:
                parsed_values.append(validate_feature_set_version(stripped))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed_values:
        raise argparse.ArgumentTypeError(f"{parameter_name} requires at least one value.")
    return tuple(dict.fromkeys(parsed_values))


def _equivalent_command(plan: BenchmarkGapFeatureStudyPlan) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_tiingo_benchmark_gap_feature_study",
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
    ]
    command.extend(_values_arg("--train-window-years", _unique_train_windows(plan)))
    command.extend(
        _values_arg("--validation-window-months", _unique_validation_windows(plan))
    )
    command.extend(_values_arg("--test-window-months", _unique_test_windows(plan)))
    command.extend(
        _values_arg(
            "--transaction-cost-bps-values",
            _transaction_cost_bps_values(plan),
        )
    )
    command.extend(_values_arg("--feature-set-versions", plan.feature_set_versions))
    command.append(
        "--include-reference-candidates"
        if plan.include_reference_candidates
        else "--no-include-reference-candidates"
    )
    if not plan.resume:
        command.append("--no-resume")
    return command


def _values_arg(name: str, values: Sequence[object]) -> list[str]:
    return [name, *[str(value) for value in values]]


def _unique_train_windows(plan: BenchmarkGapFeatureStudyPlan) -> list[int]:
    return sorted({fold_plan.train_window_years for fold_plan in plan.fold_plans})


def _unique_validation_windows(plan: BenchmarkGapFeatureStudyPlan) -> list[int]:
    return sorted({fold_plan.validation_window_months for fold_plan in plan.fold_plans})


def _unique_test_windows(plan: BenchmarkGapFeatureStudyPlan) -> list[int]:
    return sorted({fold_plan.test_window_months for fold_plan in plan.fold_plans})


def _candidate_roles(plan: BenchmarkGapFeatureStudyPlan) -> list[str]:
    return sorted({candidate.candidate_role for candidate in plan.candidate_specs})


def _model_types(plan: BenchmarkGapFeatureStudyPlan) -> list[str]:
    return sorted({candidate.model_type for candidate in plan.candidate_specs})


def _top_k_values(plan: BenchmarkGapFeatureStudyPlan) -> list[int]:
    return sorted({candidate.top_k for candidate in plan.candidate_specs})


def _smoothing_values(plan: BenchmarkGapFeatureStudyPlan) -> list[str]:
    return sorted(
        {
            f"hold{candidate.min_holding_days}/margin{candidate.switch_margin:g}"
            for candidate in plan.candidate_specs
        }
    )


def _transaction_cost_bps_values(plan: BenchmarkGapFeatureStudyPlan) -> list[float]:
    return list(plan.transaction_cost_bps_values)


def _feature_set_preflight_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("feature_set_version", sort=False, dropna=False)
        .agg(
            config_count=("valid_config", "size"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
            selected_feature_count_min=("selected_feature_count", "min"),
            selected_feature_count_max=("selected_feature_count", "max"),
        )
        .reset_index()
    )


def _candidate_preflight_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(
            [
                "candidate_family",
                "candidate_role",
                "model_type",
                "top_k",
                "min_holding_days",
                "switch_margin",
            ],
            sort=False,
            dropna=False,
        )
        .agg(
            config_count=("valid_config", "size"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
        )
        .reset_index()
    )


def _fold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("fold_name", sort=False, dropna=False)
        .agg(
            train_window_years=("train_window_years", "first"),
            validation_window_months=("validation_window_months", "first"),
            test_window_months=("test_window_months", "first"),
            selected_feature_count=("selected_feature_count", "max"),
            train_eligible_row_count=("train_eligible_row_count", "min"),
            validation_eligible_row_count=("validation_eligible_row_count", "min"),
            test_eligible_row_count=("test_eligible_row_count", "min"),
            test_symbol_count=("test_symbol_count", "min"),
            min_symbols_per_test_date=("min_symbols_per_test_date", "min"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
        )
        .reset_index()
    )


def _format_table_value(value: object) -> str:
    if isinstance(value, float | np.floating):
        return _fmt(float(value))
    return str(value)


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.4f}"


create_tiingo_benchmark_gap_feature_study_from_args = (
    run_tiingo_benchmark_gap_feature_study_from_args
)


if __name__ == "__main__":
    raise SystemExit(run_tiingo_benchmark_gap_feature_study_from_args())

