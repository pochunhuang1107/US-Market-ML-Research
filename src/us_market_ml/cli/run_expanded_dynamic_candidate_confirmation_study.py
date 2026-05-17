"""Run the expanded local dynamic candidate confirmation study."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

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
    "etf_daily_prices_extended_tiingo_split/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/expanded_dynamic_candidate_confirmation_study")
DEFAULT_RUN_NAME = "expanded_dynamic_candidate_confirmation_study"
DEFAULT_DYNAMIC_START = "2013-01-01"
DEFAULT_DATA_END = "2026-05-15"

DEFAULT_TRAIN_WINDOW_YEARS = (3, 5, 10)
DEFAULT_VALIDATION_WINDOW_MONTHS = (6, 12)
DEFAULT_TEST_WINDOW_MONTHS = (3, 6, 12)
DEFAULT_STEP_MONTHS = 6
INCLUDE_PARTIAL_FINAL_TEST = True
MIN_PARTIAL_TEST_DAYS = 60

TARGET_COL = "excess_return_vs_spy_20d"
FEATURE_SET_VERSION = "default_features_v1"
LABEL_HORIZON_DAYS = 20
MIN_HISTORY_ROWS = 252
MIN_HOLDING_DAYS = 10
SWITCH_MARGIN = 0.002
RANDOM_STATE = 42
ALLOCATION_MODE = "equal_weight"
TRANSACTION_COST_BPS_VALUES = (10.0, 20.0)
UNIVERSE_NAME = "expanded_dynamic"
UNIVERSE_MODE = "dynamic"
FEATURE_SELECTION_SCOPE = "fold_train"

PREFLIGHT_CSV_NAME = "expanded_dynamic_candidate_confirmation_preflight.csv"
PREFLIGHT_MD_NAME = "expanded_dynamic_candidate_confirmation_preflight.md"
SKIPPED_KEYS_SUFFIX = "_skipped_keys.txt"

PREFLIGHT_COLUMNS = [
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
    "model_type",
    "top_k",
    "allocation_mode",
    "transaction_cost_bps",
    "selected_feature_count",
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
class ExpandedCandidateSpec:
    """One allowed candidate direction for the expanded confirmation study."""

    model_type: str
    top_k: int

    @property
    def model_config(self) -> SweepModelConfig:
        return SweepModelConfig(
            model_family="baseline",
            model_type=self.model_type,
            random_state=RANDOM_STATE,
        )

    @property
    def signal_config(self) -> SweepSignalConfig:
        return SweepSignalConfig(
            top_k=self.top_k,
            min_holding_days=MIN_HOLDING_DAYS,
            switch_margin=SWITCH_MARGIN,
            allocation_mode=ALLOCATION_MODE,
        )


@dataclass(frozen=True)
class ExpandedFoldPlan:
    """A sweep fold plus the window metadata used to generate it."""

    fold: SweepFoldConfig
    train_window_years: int
    validation_window_months: int
    test_window_months: int


@dataclass(frozen=True)
class ExpandedDynamicCandidateConfirmationPlan:
    """Executable expanded dynamic confirmation sweep plan."""

    fold_plans: tuple[ExpandedFoldPlan, ...]
    candidate_specs: tuple[ExpandedCandidateSpec, ...]
    run_configs: tuple[SweepRunConfig, ...]
    step_months: int

    @property
    def template_config(self) -> SweepRunConfig:
        return self.run_configs[0]

    @property
    def estimated_total_configs(self) -> int:
        return (
            len(self.fold_plans)
            * len(self.candidate_specs)
            * len(TRANSACTION_COST_BPS_VALUES)
        )


@dataclass(frozen=True)
class ExpandedDynamicCandidateConfirmationPreflight:
    """Diagnostic outputs generated before model training."""

    csv_path: Path
    markdown_path: Path
    skipped_keys_path: Path
    total_config_count: int
    valid_config_count: int
    skipped_config_count: int


CANDIDATE_SPECS = (
    ExpandedCandidateSpec("linear_regression", 1),
    ExpandedCandidateSpec("linear_regression", 3),
    ExpandedCandidateSpec("ridge", 3),
    ExpandedCandidateSpec("ridge", 5),
)


def run_expanded_dynamic_candidate_confirmation_study_from_args(
    argv: list[str] | None = None,
) -> int:
    """Run or dry-run the expanded dynamic candidate confirmation study."""

    args = _parse_args(argv)
    fold_plans = build_expanded_dynamic_candidate_confirmation_folds(
        data_start=args.data_start,
        data_end=args.data_end,
        train_window_years_values=args.train_window_years,
        validation_window_months_values=args.validation_window_months,
        test_window_months_values=args.test_window_months,
        step_months=args.step_months,
    )
    if not fold_plans:
        raise SystemExit("No expanded dynamic candidate confirmation folds were generated.")

    plan = build_expanded_dynamic_candidate_confirmation_plan(
        market_dataset_glob=args.market_dataset_glob,
        fold_plans=fold_plans,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
        min_history_rows=args.min_history_rows,
        min_feature_non_null_ratio=args.min_feature_non_null_ratio,
        step_months=args.step_months,
    )
    preflight = preflight_expanded_dynamic_candidate_confirmation(plan)

    _print_plan(plan=plan, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    if not args.resume:
        _clear_fresh_run_checkpoints(plan.template_config)
    for config in plan.run_configs:
        run_sweep(replace(config, resume=True))
    return 0


def build_expanded_dynamic_candidate_confirmation_folds(
    *,
    data_start: str = DEFAULT_DYNAMIC_START,
    data_end: str = DEFAULT_DATA_END,
    train_window_years_values: Sequence[int] = DEFAULT_TRAIN_WINDOW_YEARS,
    validation_window_months_values: Sequence[int] = DEFAULT_VALIDATION_WINDOW_MONTHS,
    test_window_months_values: Sequence[int] = DEFAULT_TEST_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[ExpandedFoldPlan]:
    """Build the focused rolling grid for expanded dynamic confirmation."""

    fold_plans: list[ExpandedFoldPlan] = []
    for train_window_years in train_window_years_values:
        for validation_window_months in validation_window_months_values:
            for test_window_months in test_window_months_values:
                folds = make_sliding_window_folds(
                    data_start=data_start,
                    data_end=data_end,
                    train_window_years=train_window_years,
                    validation_window_months=validation_window_months,
                    test_window_months=test_window_months,
                    step_months=step_months,
                    include_partial_final_test=INCLUDE_PARTIAL_FINAL_TEST,
                    min_partial_test_days=MIN_PARTIAL_TEST_DAYS,
                )
                for fold in folds:
                    fold_plans.append(
                        ExpandedFoldPlan(
                            fold=_to_sweep_fold_config(fold),
                            train_window_years=train_window_years,
                            validation_window_months=validation_window_months,
                            test_window_months=test_window_months,
                        )
                    )
    return fold_plans


def build_expanded_dynamic_candidate_confirmation_plan(
    *,
    market_dataset_glob: str,
    fold_plans: Sequence[ExpandedFoldPlan],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
    min_history_rows: int = MIN_HISTORY_ROWS,
    min_feature_non_null_ratio: float = 0.0,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> ExpandedDynamicCandidateConfirmationPlan:
    """Build exact candidate-specific sweep configs for the expanded study."""

    if not fold_plans:
        raise ValueError("At least one fold plan is required.")

    folds = [fold_plan.fold for fold_plan in fold_plans]
    universe = SweepUniverseConfig(name=UNIVERSE_NAME, data_path=market_dataset_glob)
    run_configs = []
    for candidate_spec in CANDIDATE_SPECS:
        run_configs.append(
            SweepRunConfig(
                universes=[universe],
                folds=folds,
                models=[candidate_spec.model_config],
                signals=[candidate_spec.signal_config],
                target_col=TARGET_COL,
                label_horizon_days=LABEL_HORIZON_DAYS,
                feature_set_versions=[FEATURE_SET_VERSION],
                min_holding_days=MIN_HOLDING_DAYS,
                switch_margin=SWITCH_MARGIN,
                transaction_cost_bps_values=list(TRANSACTION_COST_BPS_VALUES),
                output_dir=output_dir,
                run_name=run_name,
                resume=resume,
                max_workers=max_workers,
                skipped_keys_path=output_dir / f"{run_name}{SKIPPED_KEYS_SUFFIX}",
                skip_no_usable_training_rows=True,
                universe_mode=UNIVERSE_MODE,
                min_history_rows=min_history_rows,
                feature_selection_scope=FEATURE_SELECTION_SCOPE,
                min_feature_non_null_ratio=min_feature_non_null_ratio,
            )
        )

    return ExpandedDynamicCandidateConfirmationPlan(
        fold_plans=tuple(fold_plans),
        candidate_specs=CANDIDATE_SPECS,
        run_configs=tuple(run_configs),
        step_months=step_months,
    )


def preflight_expanded_dynamic_candidate_confirmation(
    plan: ExpandedDynamicCandidateConfirmationPlan,
) -> ExpandedDynamicCandidateConfirmationPreflight:
    """Write fold/config eligibility diagnostics and skipped invalid keys."""

    config = plan.template_config
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / PREFLIGHT_CSV_NAME
    markdown_path = output_dir / PREFLIGHT_MD_NAME
    skipped_keys_path = output_dir / f"{config.run_name}{SKIPPED_KEYS_SUFFIX}"

    rows = _build_preflight_rows(plan)
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(_render_preflight_markdown(frame, plan), encoding="utf-8")

    invalid_keys = set(frame.loc[~frame["valid_config"], "run_key"].astype(str).tolist())
    existing_skipped = _read_key_checkpoint(skipped_keys_path) if config.resume else set()
    skipped_keys = sorted(existing_skipped.union(invalid_keys))
    skipped_keys_path.write_text(
        "".join(f"{key}\n" for key in skipped_keys),
        encoding="utf-8",
    )

    return ExpandedDynamicCandidateConfirmationPreflight(
        csv_path=csv_path,
        markdown_path=markdown_path,
        skipped_keys_path=skipped_keys_path,
        total_config_count=int(len(frame)),
        valid_config_count=int(frame["valid_config"].sum()) if not frame.empty else 0,
        skipped_config_count=int((~frame["valid_config"]).sum()) if not frame.empty else 0,
    )


def _build_preflight_rows(
    plan: ExpandedDynamicCandidateConfirmationPlan,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    config = plan.template_config
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

            for candidate_spec in plan.candidate_specs:
                model_config = candidate_spec.model_config
                signal_config = candidate_spec.signal_config
                for transaction_cost_bps in TRANSACTION_COST_BPS_VALUES:
                    supports_top_k = (
                        fold_counts["min_symbols_per_test_date"] >= signal_config.top_k
                    )
                    invalid_reason = _invalid_reason(
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
                            "run_key": run_key,
                            "universe": universe.name,
                            "fold_name": fold_config.name,
                            "train_window_years": fold_plan.train_window_years,
                            "validation_window_months": fold_plan.validation_window_months,
                            "test_window_months": fold_plan.test_window_months,
                            "train_start": fold_config.train_start,
                            "train_end": fold_config.train_end,
                            "validation_start": fold_config.validation_start,
                            "validation_end": fold_config.validation_end,
                            "test_start": fold_config.test_start,
                            "test_end": fold_config.test_end,
                            "model_type": model_config.model_type,
                            "top_k": signal_config.top_k,
                            "allocation_mode": signal_config.allocation_mode,
                            "transaction_cost_bps": transaction_cost_bps,
                            **fold_counts,
                            "supports_top_k": supports_top_k,
                            "valid_config": invalid_reason == "",
                            "invalid_reason": invalid_reason,
                        }
                    )
    return rows


def _render_preflight_markdown(
    frame: pd.DataFrame,
    plan: ExpandedDynamicCandidateConfirmationPlan,
) -> str:
    config = plan.template_config
    valid_count = int(frame["valid_config"].sum()) if not frame.empty else 0
    skipped_count = int((~frame["valid_config"]).sum()) if not frame.empty else 0
    fold_summary = _fold_summary(frame)
    candidate_summary = _candidate_preflight_summary(frame)
    invalid_preview = frame.loc[~frame["valid_config"]].head(20)

    lines = [
        "# Expanded Dynamic Candidate Confirmation Preflight",
        "",
        "Local-only diagnostic before model training/backtesting.",
        "",
        "## Configuration",
        "",
        f"- Universe mode: `{config.universe_mode}`",
        f"- Feature selection scope: `{config.feature_selection_scope}`",
        f"- Minimum history rows: `{config.min_history_rows}`",
        f"- Target: `{config.target_col}`",
        f"- Label horizon days: `{config.label_horizon_days}`",
        f"- Feature set: `{config.feature_set_versions[0]}`",
        f"- Allocation mode: `{ALLOCATION_MODE}`",
        f"- Step months: `{plan.step_months}`",
        f"- Include partial final test: `{INCLUDE_PARTIAL_FINAL_TEST}`",
        f"- Minimum partial test days: `{MIN_PARTIAL_TEST_DAYS}`",
        "",
        "## Totals",
        "",
        f"- Generated fold count: `{len(plan.fold_plans)}`",
        f"- Candidate combinations: `{len(plan.candidate_specs)}`",
        f"- Transaction costs: `{list(TRANSACTION_COST_BPS_VALUES)}`",
        f"- Total configs: `{len(frame)}`",
        f"- Valid configs: `{valid_count}`",
        f"- Skipped configs: `{skipped_count}`",
        "",
        "## Candidate Summary",
        "",
        _markdown_table(
            candidate_summary,
            [
                "model_type",
                "top_k",
                "allocation_mode",
                "config_count",
                "valid_config_count",
                "invalid_config_count",
            ],
        ),
        "",
        "## Fold Summary",
        "",
        _markdown_table(
            fold_summary,
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
        "## Invalid Preview",
        "",
        _markdown_table(
            invalid_preview,
            [
                "fold_name",
                "model_type",
                "top_k",
                "allocation_mode",
                "transaction_cost_bps",
                "selected_feature_count",
                "min_symbols_per_test_date",
                "invalid_reason",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


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


def _candidate_preflight_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["model_type", "top_k", "allocation_mode"], sort=False, dropna=False)
        .agg(
            config_count=("valid_config", "size"),
            valid_config_count=("valid_config", "sum"),
            invalid_config_count=("valid_config", lambda values: int((~values).sum())),
        )
        .reset_index()
    )


def _print_plan(
    *,
    plan: ExpandedDynamicCandidateConfirmationPlan,
    preflight: ExpandedDynamicCandidateConfirmationPreflight,
    dry_run: bool,
) -> None:
    config = plan.template_config
    output_dir = Path(config.output_dir)
    print("US Market ML expanded dynamic candidate confirmation study")
    print("local-only: true")
    print("uses S3: false")
    print("includes GPU/NN work: false")
    print(f"dry_run: {dry_run}")
    print(f"dataset glob: {config.universes[0].data_path}")
    print(f"universe: {config.universes[0].name}")
    print(f"universe_mode: {config.universe_mode}")
    print(f"feature_selection_scope: {config.feature_selection_scope}")
    print(f"min_history_rows: {config.min_history_rows}")
    print(f"run_name: {config.run_name}")
    print(f"output_dir: {output_dir}")
    print(f"preflight CSV: {preflight.csv_path}")
    print(f"preflight Markdown: {preflight.markdown_path}")
    print(f"skipped checkpoint: {preflight.skipped_keys_path}")
    print(f"results CSV: {output_dir / f'{config.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{config.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{config.run_name}_failed_keys.txt'}")
    print(f"generated fold count: {len(plan.fold_plans)}")
    print(f"candidate combination count: {len(plan.candidate_specs)}")
    print(f"transaction costs: {list(TRANSACTION_COST_BPS_VALUES)}")
    print(f"estimated total configs: {preflight.total_config_count}")
    print(f"estimated valid configs: {preflight.valid_config_count}")
    print(f"estimated skipped configs: {preflight.skipped_config_count}")
    print(f"max workers: {config.max_workers}")
    print(f"resume: {config.resume}")
    print()
    print("Candidate model/top_k combinations:")
    for candidate_spec in plan.candidate_specs:
        print(f"- {candidate_spec.model_type} top{candidate_spec.top_k}")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(plan)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local expanded dynamic candidate confirmation study."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
        help="Local parquet dataset glob. Defaults to MARKET_DATASET_GLOB or local data.",
    )
    parser.add_argument("--data-start", default=DEFAULT_DYNAMIC_START)
    parser.add_argument("--data-end", default=DEFAULT_DATA_END)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from local checkpoint files by default.",
    )
    parser.add_argument(
        "--max-workers",
        type=_parse_max_workers,
        default=_parse_max_workers(os.environ.get("MAX_WORKERS", "4")),
        help="Number of local worker processes. Defaults to MAX_WORKERS or 4.",
    )
    parser.add_argument("--min-history-rows", type=_parse_positive_int, default=MIN_HISTORY_ROWS)
    parser.add_argument(
        "--step-months",
        type=_parse_positive_int,
        default=DEFAULT_STEP_MONTHS,
        help="Rolling step size in months. Defaults to 6 for a focused long run.",
    )
    parser.add_argument(
        "--train-window-years",
        nargs="+",
        default=[str(value) for value in DEFAULT_TRAIN_WINDOW_YEARS],
        help="One or more train window lengths in years. Defaults to 3 5 10.",
    )
    parser.add_argument(
        "--validation-window-months",
        nargs="+",
        default=[str(value) for value in DEFAULT_VALIDATION_WINDOW_MONTHS],
        help="One or more validation window lengths in months. Defaults to 6 12.",
    )
    parser.add_argument(
        "--test-window-months",
        nargs="+",
        default=[str(value) for value in DEFAULT_TEST_WINDOW_MONTHS],
        help="One or more test window lengths in months. Defaults to 3 6 12.",
    )
    parser.add_argument(
        "--min-feature-non-null-ratio",
        type=_parse_ratio,
        default=0.0,
        help="Fold-train feature availability threshold. Defaults to all-null exclusion only.",
    )
    args = parser.parse_args(argv)

    try:
        args.train_window_years = _parse_positive_int_values(
            args.train_window_years,
            "--train-window-years",
        )
        args.validation_window_months = _parse_positive_int_values(
            args.validation_window_months,
            "--validation-window-months",
        )
        args.test_window_months = _parse_positive_int_values(
            args.test_window_months,
            "--test-window-months",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    try:
        _parse_date(args.data_start, "--data-start")
        _parse_date(args.data_end, "--data-end")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if pd.Timestamp(args.data_start) > pd.Timestamp(args.data_end):
        parser.error("--data-start must be earlier than or equal to --data-end.")
    return args


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


def _equivalent_command(plan: ExpandedDynamicCandidateConfirmationPlan) -> list[str]:
    config = plan.template_config
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_expanded_dynamic_candidate_confirmation_study",
        "--run-name",
        config.run_name,
        "--output-dir",
        str(config.output_dir),
        "--market-dataset-glob",
        str(config.universes[0].data_path),
        "--max-workers",
        str(config.max_workers),
        "--min-history-rows",
        str(config.min_history_rows),
        "--step-months",
        str(plan.step_months),
    ]
    command.extend(_values_arg("--train-window-years", _unique_train_windows(plan)))
    command.extend(
        _values_arg("--validation-window-months", _unique_validation_windows(plan))
    )
    command.extend(_values_arg("--test-window-months", _unique_test_windows(plan)))
    if not config.resume:
        command.append("--no-resume")
    return command


def _unique_train_windows(plan: ExpandedDynamicCandidateConfirmationPlan) -> list[int]:
    return sorted({fold_plan.train_window_years for fold_plan in plan.fold_plans})


def _unique_validation_windows(plan: ExpandedDynamicCandidateConfirmationPlan) -> list[int]:
    return sorted({fold_plan.validation_window_months for fold_plan in plan.fold_plans})


def _unique_test_windows(plan: ExpandedDynamicCandidateConfirmationPlan) -> list[int]:
    return sorted({fold_plan.test_window_months for fold_plan in plan.fold_plans})


def _values_arg(name: str, values: Sequence[int]) -> list[str]:
    return [name, *[str(value) for value in values]]


def _clear_fresh_run_checkpoints(config: SweepRunConfig) -> None:
    output_dir = Path(config.output_dir)
    for suffix in ("_results.csv", "_completed_keys.txt", "_failed_keys.txt"):
        (output_dir / f"{config.run_name}{suffix}").unlink(missing_ok=True)


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
        message = f"Invalid integer for {parameter_name}: {value!r}"
        raise argparse.ArgumentTypeError(message) from exc
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


def _parse_ratio(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ratio: {value!r}") from exc
    if parsed < 0.0 or parsed > 1.0:
        message = "--min-feature-non-null-ratio must satisfy 0 <= value <= 1."
        raise argparse.ArgumentTypeError(message)
    return parsed


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
    return " \\\n  ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(run_expanded_dynamic_candidate_confirmation_study_from_args())
