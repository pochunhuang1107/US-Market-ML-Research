"""Run the local Tiingo-only modern classical baseline sweep."""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import numpy as np
import pandas as pd

from us_market_ml.cli.run_dynamic_extended_history_study import (
    _eligible_preflight_splits,
    _invalid_reason,
    _preflight_fold_counts,
    _read_key_checkpoint,
    _selected_preflight_features,
    _split_for_preflight,
    _to_walk_forward_fold,
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
from us_market_ml.splits import WalkForwardFold, make_sliding_window_folds

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_tiingo_only/**/*.parquet"
)
OLD_BRIDGED_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_extended_tiingo_split/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/tiingo_modern_baseline_sweep")
DEFAULT_RUN_NAME = "tiingo_modern_baseline_sweep"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-05-15"
DEFAULT_SWEEP_PROFILE = "core_linear"

DATASET_SOURCE_LABEL = "tiingo_only"
PRICE_MODE = "tiingo_adjusted"
TARGET_COL = "excess_return_vs_spy_20d"
FEATURE_SET_VERSION = "default_features_v1"
LABEL_HORIZON_DAYS = 20
RANDOM_STATE = 42
ALLOCATION_MODE = "equal_weight"
UNIVERSE_NAME = "tiingo_modern_dynamic"
UNIVERSE_MODE = "dynamic"
FEATURE_SELECTION_SCOPE = "fold_train"

DEFAULT_MIN_HISTORY_ROWS = 252
DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS = 5_000
DEFAULT_MAX_WORKERS = 4
DEFAULT_STEP_MONTHS = 3
DEFAULT_TRAIN_WINDOW_YEARS = (2, 3, 5)
DEFAULT_VALIDATION_WINDOW_MONTHS = (6, 12)
DEFAULT_TEST_WINDOW_MONTHS = (3, 6, 12)
INCLUDE_PARTIAL_FINAL_TEST = True
MIN_PARTIAL_TEST_DAYS = 60

BASELINE_MODEL_TYPES = (
    "linear_regression",
    "ridge",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)
TOP_K_VALUES = (1, 3, 5)
TRANSACTION_COST_BPS_VALUES = (5.0, 10.0, 20.0)
MIN_HOLDING_DAYS_VALUES = (5, 10, 20)
SWITCH_MARGIN_VALUES = (0.0, 0.002, 0.005)

PREFLIGHT_CSV_NAME = "tiingo_modern_baseline_sweep_preflight.csv"
PREFLIGHT_MD_NAME = "tiingo_modern_baseline_sweep_preflight.md"
SKIPPED_KEYS_SUFFIX = "_skipped_keys.txt"

PREFLIGHT_COLUMNS = [
    "dataset_source",
    "sweep_profile",
    "run_key",
    "universe",
    "fold_name",
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
class TiingoModernBaselineSweepProfile:
    """Named staged grid defaults for the Tiingo-only baseline sweep."""

    name: str
    purpose: str
    model_types: tuple[str, ...]
    top_k_values: tuple[int, ...]
    transaction_cost_bps_values: tuple[float, ...]
    min_holding_days_values: tuple[int, ...]
    switch_margin_values: tuple[float, ...]
    train_window_years_values: tuple[int, ...]
    validation_window_months_values: tuple[int, ...]
    test_window_months_values: tuple[int, ...]
    step_months: int


SWEEP_PROFILES: dict[str, TiingoModernBaselineSweepProfile] = {
    "core_linear": TiingoModernBaselineSweepProfile(
        name="core_linear",
        purpose="First serious overnight baseline run.",
        model_types=("linear_regression", "ridge"),
        top_k_values=TOP_K_VALUES,
        transaction_cost_bps_values=TRANSACTION_COST_BPS_VALUES,
        min_holding_days_values=MIN_HOLDING_DAYS_VALUES,
        switch_margin_values=SWITCH_MARGIN_VALUES,
        train_window_years_values=DEFAULT_TRAIN_WINDOW_YEARS,
        validation_window_months_values=DEFAULT_VALIDATION_WINDOW_MONTHS,
        test_window_months_values=DEFAULT_TEST_WINDOW_MONTHS,
        step_months=DEFAULT_STEP_MONTHS,
    ),
    "tree_screen": TiingoModernBaselineSweepProfile(
        name="tree_screen",
        purpose="Controlled tree-model screen after core_linear.",
        model_types=("random_forest", "extra_trees", "gradient_boosting"),
        top_k_values=TOP_K_VALUES,
        transaction_cost_bps_values=(10.0, 20.0),
        min_holding_days_values=(10, 20),
        switch_margin_values=(0.002, 0.005),
        train_window_years_values=(3, 5),
        validation_window_months_values=(12,),
        test_window_months_values=(6, 12),
        step_months=6,
    ),
    "linear_quick": TiingoModernBaselineSweepProfile(
        name="linear_quick",
        purpose="Small smoke run / quick validation.",
        model_types=("linear_regression", "ridge"),
        top_k_values=TOP_K_VALUES,
        transaction_cost_bps_values=(10.0,),
        min_holding_days_values=(10,),
        switch_margin_values=(0.002,),
        train_window_years_values=(3, 5),
        validation_window_months_values=(12,),
        test_window_months_values=(6, 12),
        step_months=6,
    ),
    "full": TiingoModernBaselineSweepProfile(
        name="full",
        purpose="The previous full broad baseline sweep.",
        model_types=BASELINE_MODEL_TYPES,
        top_k_values=TOP_K_VALUES,
        transaction_cost_bps_values=TRANSACTION_COST_BPS_VALUES,
        min_holding_days_values=MIN_HOLDING_DAYS_VALUES,
        switch_margin_values=SWITCH_MARGIN_VALUES,
        train_window_years_values=DEFAULT_TRAIN_WINDOW_YEARS,
        validation_window_months_values=DEFAULT_VALIDATION_WINDOW_MONTHS,
        test_window_months_values=DEFAULT_TEST_WINDOW_MONTHS,
        step_months=DEFAULT_STEP_MONTHS,
    ),
}


@dataclass(frozen=True)
class TiingoModernBaselineFoldPlan:
    """A sweep fold plus the window metadata used to create it."""

    fold: SweepFoldConfig
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass(frozen=True)
class TiingoModernBaselineSweepPlan:
    """Executable Tiingo-only modern baseline sweep plan."""

    fold_plans: tuple[TiingoModernBaselineFoldPlan, ...]
    config: SweepRunConfig
    sweep_profile: str
    start_date: str
    end_date: str
    step_months: int
    min_train_eligible_rows: int

    @property
    def estimated_total_configs(self) -> int:
        return (
            len(self.fold_plans)
            * len(self.config.models)
            * len(self.config.signals)
            * len(self.config.transaction_cost_bps_values)
            * len(self.config.feature_set_versions)
            * len(self.config.universes)
        )


@dataclass(frozen=True)
class TiingoModernBaselinePreflight:
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


def run_tiingo_modern_baseline_sweep_from_args(argv: list[str] | None = None) -> int:
    """Run or dry-run the Tiingo-only modern classical baseline sweep."""

    args = _parse_args(argv)
    _install_interrupt_handlers()
    _ensure_dataset_available(args.market_dataset_glob)

    fold_plans = build_tiingo_modern_baseline_folds(
        start_date=args.start_date,
        end_date=args.end_date,
        train_window_years_values=args.train_window_years,
        validation_window_months_values=args.validation_window_months,
        test_window_months_values=args.test_window_months,
        step_months=args.step_months,
    )
    if not fold_plans:
        raise SystemExit("No Tiingo modern baseline folds were generated.")

    plan = build_tiingo_modern_baseline_sweep_plan(
        market_dataset_glob=args.market_dataset_glob,
        fold_plans=fold_plans,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
        sweep_profile=args.sweep_profile,
        model_types=args.model_types,
        top_k_values=args.top_k_values,
        transaction_cost_bps_values=args.transaction_cost_bps_values,
        min_holding_days_values=args.min_holding_days_values,
        switch_margin_values=args.switch_margin_values,
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=args.min_history_rows,
        min_train_eligible_rows=args.min_train_eligible_rows,
        step_months=args.step_months,
    )
    preflight = preflight_tiingo_modern_baseline_sweep(plan)

    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    if not args.resume:
        _clear_fresh_run_checkpoints(plan.config)

    try:
        run_sweep(plan.config)
    except KeyboardInterrupt:
        print("Interrupt received; Tiingo modern baseline sweep stopped cleanly.")
        return 130
    return 0


def build_tiingo_modern_baseline_folds(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    train_window_years_values: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months_values: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months_values: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[TiingoModernBaselineFoldPlan]:
    """Build the Tiingo-only modern rolling window grid."""

    fold_plans: list[TiingoModernBaselineFoldPlan] = []
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
                        TiingoModernBaselineFoldPlan(
                            fold=_to_sweep_fold_config(fold),
                            train_window_years=train_window_years,
                            validation_window_months=validation_window_months,
                            test_window_months=test_window_months,
                        )
                    )
    return fold_plans


def build_tiingo_modern_baseline_sweep_plan(
    *,
    market_dataset_glob: str,
    fold_plans: Sequence[TiingoModernBaselineFoldPlan],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
    sweep_profile: str = DEFAULT_SWEEP_PROFILE,
    model_types: Sequence[str] | None = None,
    top_k_values: Sequence[int] | None = None,
    transaction_cost_bps_values: Sequence[float] | None = None,
    min_holding_days_values: Sequence[int] | None = None,
    switch_margin_values: Sequence[float] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
    min_train_eligible_rows: int = DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> TiingoModernBaselineSweepPlan:
    """Build the broad Tiingo-only classical baseline sweep config."""

    if not fold_plans:
        raise ValueError("At least one fold plan is required.")

    profile = _sweep_profile(sweep_profile)
    resolved_model_types = tuple(model_types) if model_types is not None else profile.model_types
    resolved_top_k_values = (
        tuple(top_k_values) if top_k_values is not None else profile.top_k_values
    )
    resolved_transaction_cost_bps_values = (
        tuple(transaction_cost_bps_values)
        if transaction_cost_bps_values is not None
        else profile.transaction_cost_bps_values
    )
    resolved_min_holding_days_values = (
        tuple(min_holding_days_values)
        if min_holding_days_values is not None
        else profile.min_holding_days_values
    )
    resolved_switch_margin_values = (
        tuple(switch_margin_values)
        if switch_margin_values is not None
        else profile.switch_margin_values
    )
    _validate_resolved_grid(
        model_types=resolved_model_types,
        top_k_values=resolved_top_k_values,
        transaction_cost_bps_values=resolved_transaction_cost_bps_values,
        min_holding_days_values=resolved_min_holding_days_values,
        switch_margin_values=resolved_switch_margin_values,
    )

    folds = [fold_plan.fold for fold_plan in fold_plans]
    universe = SweepUniverseConfig(name=UNIVERSE_NAME, data_path=market_dataset_glob)
    config = SweepRunConfig(
        universes=[universe],
        folds=folds,
        models=[
            SweepModelConfig(
                model_family="baseline",
                model_type=model_type,
                random_state=RANDOM_STATE,
            )
            for model_type in resolved_model_types
        ],
        signals=[
            SweepSignalConfig(
                top_k=top_k,
                min_holding_days=min_holding_days,
                switch_margin=switch_margin,
                allocation_mode=ALLOCATION_MODE,
            )
            for top_k in resolved_top_k_values
            for min_holding_days in resolved_min_holding_days_values
            for switch_margin in resolved_switch_margin_values
        ],
        target_col=TARGET_COL,
        label_horizon_days=LABEL_HORIZON_DAYS,
        feature_set_versions=[FEATURE_SET_VERSION],
        min_holding_days=resolved_min_holding_days_values[0],
        switch_margin=resolved_switch_margin_values[0],
        transaction_cost_bps_values=list(resolved_transaction_cost_bps_values),
        output_dir=output_dir,
        run_name=run_name,
        resume=resume,
        max_workers=max_workers,
        skipped_keys_path=output_dir / f"{run_name}{SKIPPED_KEYS_SUFFIX}",
        skip_no_usable_training_rows=True,
        universe_mode=UNIVERSE_MODE,
        min_history_rows=min_history_rows,
        feature_selection_scope=FEATURE_SELECTION_SCOPE,
        min_feature_non_null_ratio=0.0,
        result_metadata={"sweep_profile": profile.name},
    )
    return TiingoModernBaselineSweepPlan(
        fold_plans=tuple(fold_plans),
        config=config,
        sweep_profile=profile.name,
        start_date=start_date,
        end_date=end_date,
        step_months=step_months,
        min_train_eligible_rows=min_train_eligible_rows,
    )


def preflight_tiingo_modern_baseline_sweep(
    plan: TiingoModernBaselineSweepPlan,
) -> TiingoModernBaselinePreflight:
    """Write fold/config diagnostics and seed skipped invalid keys."""

    config = plan.config
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / _preflight_csv_name(config.run_name)
    markdown_path = output_dir / _preflight_markdown_name(config.run_name)
    skipped_keys_path = output_dir / f"{config.run_name}{SKIPPED_KEYS_SUFFIX}"
    results_csv_path = output_dir / f"{config.run_name}_results.csv"
    completed_keys_path = output_dir / f"{config.run_name}_completed_keys.txt"
    failed_keys_path = output_dir / f"{config.run_name}_failed_keys.txt"
    terminal_log_path = output_dir / f"{config.run_name}_terminal.log"

    rows = _build_preflight_rows(plan)
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    frame.to_csv(csv_path, index=False)

    invalid_keys = set(frame.loc[~frame["valid_config"], "run_key"].astype(str).tolist())
    existing_skipped = _read_key_checkpoint(skipped_keys_path) if config.resume else set()
    skipped_keys = sorted(existing_skipped.union(invalid_keys))
    skipped_keys_path.write_text(
        "".join(f"{key}\n" for key in skipped_keys),
        encoding="utf-8",
    )

    valid_keys = set(frame.loc[frame["valid_config"], "run_key"].astype(str).tolist())
    completed_keys = _read_key_checkpoint(completed_keys_path) if config.resume else set()
    completed_in_plan = valid_keys.intersection(completed_keys)
    failed_count = _non_empty_line_count(failed_keys_path) if config.resume else 0
    pending_count = max(0, len(valid_keys) - len(completed_in_plan))

    preflight = TiingoModernBaselinePreflight(
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


def _build_preflight_rows(
    plan: TiingoModernBaselineSweepPlan,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = plan.config
    universe = config.universes[0]
    raw_df = read_local_market_data(universe.data_path)

    for feature_set_version in config.feature_set_versions:
        dataset = build_supervised_dataset(
            raw_df,
            SupervisedDatasetConfig(
                forward_horizons=(5, 10, config.label_horizon_days),
                benchmark_symbol="SPY",
                excess_return_horizons=(config.label_horizon_days,),
                rank_target_col=config.target_col,
                rank_col=f"rank_{config.target_col}",
                feature_set_version=feature_set_version,
            ),
        )
        if config.target_col not in dataset.columns:
            raise ValueError(
                "Configured target_col is missing after supervised dataset construction: "
                f"{config.target_col}"
            )
        candidate_feature_cols = get_feature_columns(
            dataset,
            feature_set_version=feature_set_version,
        )

        for fold_plan in plan.fold_plans:
            fold_config = fold_plan.fold
            fold = _to_walk_forward_fold(fold_config)
            train_df, validation_df, test_df = _split_for_preflight(dataset, fold, config)
            selected_feature_cols = _selected_preflight_features(
                train_df=train_df,
                candidate_feature_cols=candidate_feature_cols,
                config=config,
            )
            train_eligible, validation_eligible, test_eligible = _eligible_preflight_splits(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                eligibility_history_df=dataset,
                feature_cols=selected_feature_cols,
                config=config,
            )
            fold_counts = _preflight_fold_counts(
                selected_feature_cols=selected_feature_cols,
                train_eligible=train_eligible,
                validation_eligible=validation_eligible,
                test_eligible=test_eligible,
            )

            for model_config in config.models:
                for signal_config in config.signals:
                    for transaction_cost_bps in config.transaction_cost_bps_values:
                        supports_top_k = (
                            fold_counts["min_symbols_per_test_date"] >= signal_config.top_k
                        )
                        invalid_reason = _tiingo_invalid_reason(
                            selected_feature_count=fold_counts["selected_feature_count"],
                            train_eligible_row_count=fold_counts["train_eligible_row_count"],
                            validation_eligible_row_count=fold_counts[
                                "validation_eligible_row_count"
                            ],
                            test_eligible_row_count=fold_counts["test_eligible_row_count"],
                            train_symbol_count=fold_counts["train_symbol_count"],
                            validation_symbol_count=fold_counts["validation_symbol_count"],
                            test_symbol_count=fold_counts["test_symbol_count"],
                            supports_top_k=supports_top_k,
                            min_train_eligible_rows=plan.min_train_eligible_rows,
                        )
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
                            universe_mode=config.universe_mode,
                            min_history_rows=config.min_history_rows,
                            feature_selection_scope=config.feature_selection_scope,
                            min_feature_non_null_ratio=config.min_feature_non_null_ratio,
                        )
                        rows.append(
                            {
                                "dataset_source": DATASET_SOURCE_LABEL,
                                "sweep_profile": plan.sweep_profile,
                                "run_key": run_key,
                                "universe": universe.name,
                                "fold_name": fold_config.name,
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
                                "model_family": model_config.model_family,
                                "model_type": model_config.model_type,
                                "top_k": signal_config.top_k,
                                "allocation_mode": signal_config.allocation_mode,
                                "min_holding_days": signal_config.min_holding_days,
                                "switch_margin": signal_config.switch_margin,
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
    plan: TiingoModernBaselineSweepPlan,
    preflight: TiingoModernBaselinePreflight,
) -> str:
    config = plan.config
    invalid_preview = frame.loc[~frame["valid_config"]].head(20)
    model_summary = _model_preflight_summary(frame)
    fold_summary = _fold_summary(frame)

    lines = [
        "# Tiingo Modern Baseline Sweep Preflight",
        "",
        "Local-only diagnostic before model training/backtesting.",
        "",
        "## Sweep Profile",
        "",
        f"- Sweep profile: `{plan.sweep_profile}`",
        f"- Profile purpose: `{_sweep_profile(plan.sweep_profile).purpose}`",
        "",
        "## Dataset",
        "",
        f"- Dataset source: `{DATASET_SOURCE_LABEL}`",
        "- Source provider: `tiingo`",
        f"- Price mode: `{PRICE_MODE}`",
        f"- Dataset path/glob: `{config.universes[0].data_path}`",
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
        "## Baseline Model Grid",
        "",
        "- Model family: `baseline`",
        f"- Model types: `{_model_types(config)}`",
        f"- Random state: `{RANDOM_STATE}`",
        "- Momentum baseline: deferred; no supported momentum model exists in this sweep runner.",
        "",
        "## Signal And Cost Grid",
        "",
        f"- Top-K values: `{_top_k_values(config)}`",
        f"- Allocation mode: `{ALLOCATION_MODE}`",
        f"- Min holding days: `{_min_holding_days_values(config)}`",
        f"- Switch margins: `{_switch_margin_values(config)}`",
        f"- Transaction costs bps: `{_transaction_cost_bps_values(config)}`",
        "",
        "## Research Controls",
        "",
        f"- Universe mode: `{config.universe_mode}`",
        f"- Feature selection scope: `{config.feature_selection_scope}`",
        f"- Minimum history rows: `{config.min_history_rows}`",
        f"- Minimum train eligible rows: `{plan.min_train_eligible_rows}`",
        f"- Target: `{config.target_col}`",
        f"- Label horizon days: `{config.label_horizon_days}`",
        f"- Feature set: `{config.feature_set_versions[0]}`",
        "- Local-only: `true`",
        "- Uses S3/cloud/GPU/neural networks: `false`",
        "",
        "## Counts",
        "",
        f"- Generated fold count: `{len(plan.fold_plans)}`",
        f"- Model count: `{len(config.models)}`",
        f"- Signal count: `{len(config.signals)}`",
        f"- Total configs: `{preflight.total_config_count}`",
        f"- Valid configs: `{preflight.valid_config_count}`",
        f"- Skipped configs: `{preflight.skipped_config_count}`",
        f"- Completed configs from checkpoint: `{preflight.completed_checkpoint_count}`",
        f"- Failed configs from checkpoint: `{preflight.failed_checkpoint_count}`",
        f"- Pending configs: `{preflight.pending_config_count}`",
        "",
        "## Model Summary",
        "",
        _markdown_table(
            model_summary,
            [
                "model_family",
                "model_type",
                "config_count",
                "valid_config_count",
                "invalid_config_count",
            ],
        ),
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
                "train_window_years",
                "validation_window_months",
                "test_window_months",
                "model_type",
                "top_k",
                "min_holding_days",
                "switch_margin",
                "transaction_cost_bps",
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
        "## MLP Deferral Note",
        "",
        "MLP is intentionally deferred until after this Tiingo-only baseline sweep "
        "identifies competitive classical candidates.",
        "",
    ]
    return "\n".join(lines)


def _print_plan(
    *,
    plan: TiingoModernBaselineSweepPlan,
    preflight: TiingoModernBaselinePreflight,
    dry_run: bool,
) -> None:
    config = plan.config
    output_dir = Path(config.output_dir)
    print("US Market ML Tiingo-only modern baseline sweep")
    print("local-only: true")
    print("uses S3/cloud: false")
    print("uses GPU: false")
    print("includes neural networks/MLP: false")
    print("baseline-only: true")
    print(f"dry_run: {dry_run}")
    print(f"sweep_profile: {plan.sweep_profile}")
    print(f"dataset glob: {config.universes[0].data_path}")
    print(f"dataset source label: {DATASET_SOURCE_LABEL}")
    print(f"price mode: {PRICE_MODE}")
    print(f"start date: {plan.start_date}")
    print(f"end date: {plan.end_date}")
    print(f"universe: {config.universes[0].name}")
    print(f"universe_mode: {config.universe_mode}")
    print(f"feature_selection_scope: {config.feature_selection_scope}")
    print(f"min_history_rows: {config.min_history_rows}")
    print(f"min_train_eligible_rows: {plan.min_train_eligible_rows}")
    print(f"target_col: {config.target_col}")
    print(f"label_horizon_days: {config.label_horizon_days}")
    print(f"feature_set_version: {config.feature_set_versions[0]}")
    print(f"allocation_mode: {ALLOCATION_MODE}")
    print(f"run_name: {config.run_name}")
    print(f"output directory: {output_dir}")
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
    print(f"model count: {len(config.models)}")
    print(f"model types: {', '.join(model.model_type for model in config.models)}")
    print(f"top_k values: {_top_k_values(config)}")
    print(f"transaction costs bps: {_transaction_cost_bps_values(config)}")
    print(f"min_holding_days values: {_min_holding_days_values(config)}")
    print(f"switch_margin values: {_switch_margin_values(config)}")
    print(f"train_window_years values: {_unique_train_windows(plan)}")
    print(f"validation_window_months values: {_unique_validation_windows(plan)}")
    print(f"test_window_months values: {_unique_test_windows(plan)}")
    print(f"step_months: {plan.step_months}")
    print(f"max workers: {config.max_workers}")
    print(f"resume: {config.resume}")
    print("momentum baseline: deferred")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(plan)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Tiingo-only modern classical baseline sweep."
    )
    parser.add_argument("--market-dataset-glob", default=_default_market_dataset_glob())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME"))
    parser.add_argument(
        "--sweep-profile",
        default=os.environ.get("SWEEP_PROFILE", DEFAULT_SWEEP_PROFILE),
        choices=tuple(SWEEP_PROFILES),
        help="Named staged grid profile. Manual grid arguments override profile defaults.",
    )
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
        "--min-history-rows",
        type=_parse_positive_int,
        default=DEFAULT_MIN_HISTORY_ROWS,
    )
    parser.add_argument(
        "--min-train-eligible-rows",
        type=_parse_positive_int,
        default=DEFAULT_MIN_TRAIN_ELIGIBLE_ROWS,
    )
    parser.add_argument(
        "--step-months",
        default=os.environ.get("STEP_MONTHS"),
    )
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
        "--model-types",
        nargs="+",
        default=_env_values("MODEL_TYPES"),
    )
    parser.add_argument(
        "--top-k-values",
        nargs="+",
        default=_env_values("TOP_K_VALUES"),
    )
    parser.add_argument(
        "--transaction-cost-bps-values",
        nargs="+",
        default=_env_values("TRANSACTION_COST_BPS_VALUES"),
    )
    parser.add_argument(
        "--min-holding-days-values",
        nargs="+",
        default=_env_values("MIN_HOLDING_DAYS_VALUES"),
    )
    parser.add_argument(
        "--switch-margin-values",
        nargs="+",
        default=_env_values("SWITCH_MARGIN_VALUES"),
    )
    args = parser.parse_args(argv)

    try:
        profile = _sweep_profile(args.sweep_profile)
        args.step_months = (
            _parse_positive_int(str(args.step_months), "--step-months")
            if args.step_months is not None
            else profile.step_months
        )
        args.train_window_years = (
            _parse_positive_int_values(args.train_window_years, "--train-window-years")
            if args.train_window_years is not None
            else profile.train_window_years_values
        )
        args.validation_window_months = (
            _parse_positive_int_values(
                args.validation_window_months,
                "--validation-window-months",
            )
            if args.validation_window_months is not None
            else profile.validation_window_months_values
        )
        args.test_window_months = (
            _parse_positive_int_values(args.test_window_months, "--test-window-months")
            if args.test_window_months is not None
            else profile.test_window_months_values
        )
        args.model_types = (
            _parse_model_type_values(args.model_types, "--model-types")
            if args.model_types is not None
            else profile.model_types
        )
        args.top_k_values = (
            _parse_positive_int_values(args.top_k_values, "--top-k-values")
            if args.top_k_values is not None
            else profile.top_k_values
        )
        args.transaction_cost_bps_values = (
            _parse_non_negative_float_values(
                args.transaction_cost_bps_values,
                "--transaction-cost-bps-values",
            )
            if args.transaction_cost_bps_values is not None
            else profile.transaction_cost_bps_values
        )
        args.min_holding_days_values = (
            _parse_positive_int_values(
                args.min_holding_days_values,
                "--min-holding-days-values",
            )
            if args.min_holding_days_values is not None
            else profile.min_holding_days_values
        )
        args.switch_margin_values = (
            _parse_non_negative_float_values(
                args.switch_margin_values,
                "--switch-margin-values",
            )
            if args.switch_margin_values is not None
            else profile.switch_margin_values
        )
        _parse_date(args.start_date, "--start-date")
        _parse_date(args.end_date, "--end-date")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.run_name is None or str(args.run_name).strip() == "":
        args.run_name = _default_run_name_for_profile(args.sweep_profile)
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        parser.error("--start-date must be earlier than or equal to --end-date.")
    if str(args.market_dataset_glob) == OLD_BRIDGED_DATASET_GLOB:
        parser.error(
            "The old Tiingo/Massive bridged dataset is not allowed as the default "
            "for this Tiingo-only sweep. Use the Tiingo-only adjusted dataset glob."
        )
    return args


def _default_market_dataset_glob() -> str:
    return os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB)


def _default_run_name_for_profile(sweep_profile: str) -> str:
    return f"{DEFAULT_RUN_NAME}_{_sweep_profile(sweep_profile).name}"


def _preflight_csv_name(run_name: str) -> str:
    if run_name == DEFAULT_RUN_NAME:
        return PREFLIGHT_CSV_NAME
    return f"{run_name}_preflight.csv"


def _preflight_markdown_name(run_name: str) -> str:
    if run_name == DEFAULT_RUN_NAME:
        return PREFLIGHT_MD_NAME
    return f"{run_name}_preflight.md"


def _env_values(name: str) -> list[str] | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return shlex.split(raw_value)


def _sweep_profile(name: str) -> TiingoModernBaselineSweepProfile:
    try:
        return SWEEP_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(SWEEP_PROFILES)
        raise argparse.ArgumentTypeError(
            f"--sweep-profile must be one of: {valid}."
        ) from exc


def _validate_resolved_grid(
    *,
    model_types: Sequence[str],
    top_k_values: Sequence[int],
    transaction_cost_bps_values: Sequence[float],
    min_holding_days_values: Sequence[int],
    switch_margin_values: Sequence[float],
) -> None:
    if not model_types:
        raise ValueError("At least one model type is required.")
    if not top_k_values:
        raise ValueError("At least one top_k value is required.")
    if not transaction_cost_bps_values:
        raise ValueError("At least one transaction cost value is required.")
    if not min_holding_days_values:
        raise ValueError("At least one min_holding_days value is required.")
    if not switch_margin_values:
        raise ValueError("At least one switch_margin value is required.")
    unsupported = [value for value in model_types if value not in BASELINE_MODEL_TYPES]
    if unsupported:
        valid = ", ".join(BASELINE_MODEL_TYPES)
        raise ValueError(
            f"Unsupported baseline model type(s): {unsupported}. Expected one of: {valid}."
        )


def _ensure_dataset_available(path: str) -> None:
    path_text = str(path)
    if glob.has_magic(path_text):
        matches = [match for match in glob.glob(path_text, recursive=True) if Path(match).is_file()]
        if matches:
            return
        message = f"Tiingo-only market dataset glob matched no files: {path_text}"
    else:
        if Path(path_text).exists():
            return
        message = f"Tiingo-only market dataset path does not exist: {path_text}"

    if path_text == DEFAULT_DATASET_GLOB:
        message += (
            ". The default Tiingo-only dataset was not found; set MARKET_DATASET_GLOB "
            "or pass --market-dataset-glob to a local synthetic/test dataset."
        )
    raise FileNotFoundError(message)


def _tiingo_invalid_reason(
    *,
    selected_feature_count: int,
    train_eligible_row_count: int,
    validation_eligible_row_count: int,
    test_eligible_row_count: int,
    train_symbol_count: int,
    validation_symbol_count: int,
    test_symbol_count: int,
    supports_top_k: bool,
    min_train_eligible_rows: int,
) -> str:
    base_reason = _invalid_reason(
        selected_feature_count=selected_feature_count,
        train_eligible_row_count=train_eligible_row_count,
        validation_eligible_row_count=validation_eligible_row_count,
        test_eligible_row_count=test_eligible_row_count,
        train_symbol_count=train_symbol_count,
        validation_symbol_count=validation_symbol_count,
        test_symbol_count=test_symbol_count,
        supports_top_k=supports_top_k,
    )
    reasons = [base_reason] if base_reason else []
    if train_eligible_row_count < min_train_eligible_rows:
        reasons.append(
            "train_eligible_row_count below min_train_eligible_rows: "
            f"{train_eligible_row_count} < {min_train_eligible_rows}"
        )
    return ";".join(reasons)


def _fold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("fold_name", sort=False, dropna=False)
        .agg(
            train_window_years=("train_window_years", "first"),
            validation_window_months=("validation_window_months", "first"),
            test_window_months=("test_window_months", "first"),
            selected_feature_count=("selected_feature_count", "first"),
            train_eligible_row_count=("train_eligible_row_count", "first"),
            validation_eligible_row_count=("validation_eligible_row_count", "first"),
            test_eligible_row_count=("test_eligible_row_count", "first"),
            test_symbol_count=("test_symbol_count", "first"),
            min_symbols_per_test_date=("min_symbols_per_test_date", "first"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
        )
        .reset_index()
    )


def _model_preflight_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["model_family", "model_type"], sort=False, dropna=False)
        .agg(
            config_count=("valid_config", "size"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
        )
        .reset_index()
    )


def _to_sweep_fold_config(fold: WalkForwardFold) -> SweepFoldConfig:
    return SweepFoldConfig(
        name=fold.name,
        train_start=_date_string(fold.train_start),
        train_end=_date_string(fold.train_end),
        validation_start=_date_string(fold.validation_start),
        validation_end=_date_string(fold.validation_end),
        test_start=_date_string(fold.test_start),
        test_end=_date_string(fold.test_end),
    )


def _date_string(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _equivalent_command(plan: TiingoModernBaselineSweepPlan) -> list[str]:
    config = plan.config
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_tiingo_modern_baseline_sweep",
        "--market-dataset-glob",
        str(config.universes[0].data_path),
        "--output-dir",
        str(config.output_dir),
        "--run-name",
        config.run_name,
        "--sweep-profile",
        plan.sweep_profile,
        "--start-date",
        plan.start_date,
        "--end-date",
        plan.end_date,
        "--max-workers",
        str(config.max_workers),
        "--min-history-rows",
        str(config.min_history_rows),
        "--min-train-eligible-rows",
        str(plan.min_train_eligible_rows),
        "--step-months",
        str(plan.step_months),
    ]
    command.extend(_values_arg("--model-types", _model_types(config)))
    command.extend(_values_arg("--top-k-values", _top_k_values(config)))
    command.extend(
        _values_arg(
            "--transaction-cost-bps-values",
            _transaction_cost_bps_values(config),
        )
    )
    command.extend(
        _values_arg("--min-holding-days-values", _min_holding_days_values(config))
    )
    command.extend(_values_arg("--switch-margin-values", _switch_margin_values(config)))
    command.extend(_values_arg("--train-window-years", _unique_train_windows(plan)))
    command.extend(
        _values_arg("--validation-window-months", _unique_validation_windows(plan))
    )
    command.extend(_values_arg("--test-window-months", _unique_test_windows(plan)))
    if not config.resume:
        command.append("--no-resume")
    return command


def _unique_train_windows(plan: TiingoModernBaselineSweepPlan) -> list[int]:
    return sorted({fold_plan.train_window_years for fold_plan in plan.fold_plans})


def _unique_validation_windows(plan: TiingoModernBaselineSweepPlan) -> list[int]:
    return sorted({fold_plan.validation_window_months for fold_plan in plan.fold_plans})


def _unique_test_windows(plan: TiingoModernBaselineSweepPlan) -> list[int]:
    return sorted({fold_plan.test_window_months for fold_plan in plan.fold_plans})


def _model_types(config: SweepRunConfig) -> list[str]:
    return [model.model_type for model in config.models]


def _top_k_values(config: SweepRunConfig) -> list[int]:
    return sorted({signal.top_k for signal in config.signals})


def _min_holding_days_values(config: SweepRunConfig) -> list[int]:
    return sorted(
        {
            int(
                signal.min_holding_days
                if signal.min_holding_days is not None
                else config.min_holding_days
            )
            for signal in config.signals
        }
    )


def _switch_margin_values(config: SweepRunConfig) -> list[float]:
    return sorted(
        {
            float(
                signal.switch_margin
                if signal.switch_margin is not None
                else config.switch_margin
            )
            for signal in config.signals
        }
    )


def _transaction_cost_bps_values(config: SweepRunConfig) -> list[float]:
    return list(config.transaction_cost_bps_values)


def _values_arg(name: str, values: Sequence[object]) -> list[str]:
    return [name, *[str(value) for value in values]]


def _clear_fresh_run_checkpoints(config: SweepRunConfig) -> None:
    output_dir = Path(config.output_dir)
    for suffix in ("_results.csv", "_completed_keys.txt", "_failed_keys.txt"):
        (output_dir / f"{config.run_name}{suffix}").unlink(missing_ok=True)


def _non_empty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _parse_date(value: str, parameter_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{parameter_name} must be a valid date.") from exc


def _parse_max_workers(value: str) -> int:
    return _parse_positive_int(value, parameter_name="--max-workers")


def _parse_positive_int(value: str, parameter_name: str = "value") -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid integer for {parameter_name}: {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{parameter_name} must be greater than or equal to 1.")
    return parsed


def _parse_positive_int_values(values: Sequence[str], parameter_name: str) -> tuple[int, ...]:
    parsed_values: list[int] = []
    for value in values:
        for part in str(value).split(","):
            stripped = part.strip()
            if stripped:
                parsed_values.append(_parse_positive_int(stripped, parameter_name))
    if not parsed_values:
        raise argparse.ArgumentTypeError(f"{parameter_name} requires at least one value.")
    return tuple(dict.fromkeys(parsed_values))


def _parse_non_negative_float_values(
    values: Sequence[str],
    parameter_name: str,
) -> tuple[float, ...]:
    parsed_values: list[float] = []
    for value in values:
        for part in str(value).split(","):
            stripped = part.strip()
            if not stripped:
                continue
            try:
                parsed = float(stripped)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid float for {parameter_name}: {stripped!r}"
                ) from exc
            if parsed < 0.0:
                raise argparse.ArgumentTypeError(
                    f"{parameter_name} values must be greater than or equal to 0."
                )
            parsed_values.append(parsed)
    if not parsed_values:
        raise argparse.ArgumentTypeError(f"{parameter_name} requires at least one value.")
    return tuple(dict.fromkeys(parsed_values))


def _parse_model_type_values(
    values: Sequence[str],
    parameter_name: str,
) -> tuple[str, ...]:
    parsed_values: list[str] = []
    for value in values:
        for part in str(value).split(","):
            stripped = part.strip()
            if stripped:
                parsed_values.append(stripped)
    if not parsed_values:
        raise argparse.ArgumentTypeError(f"{parameter_name} requires at least one value.")
    unsupported = [value for value in parsed_values if value not in BASELINE_MODEL_TYPES]
    if unsupported:
        valid = ", ".join(BASELINE_MODEL_TYPES)
        raise argparse.ArgumentTypeError(
            f"Unsupported baseline model type(s) for {parameter_name}: {unsupported}. "
            f"Expected one of: {valid}."
        )
    return tuple(dict.fromkeys(parsed_values))


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    formatted = subset.apply(lambda column: column.map(_format_table_value))
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


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


def _format_command(command: list[str]) -> str:
    separator = " " + "\\" + "\n  "
    return separator.join(shlex.quote(part) for part in command)


def _install_interrupt_handlers() -> None:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(run_tiingo_modern_baseline_sweep_from_args())
