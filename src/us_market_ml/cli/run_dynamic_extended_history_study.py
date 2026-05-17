"""Run a local-only dynamic extended-history ETF study."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
from us_market_ml.splits import (
    PurgeConfig,
    WalkForwardFold,
    make_sliding_window_folds_for_train_windows,
    purge_fold_boundaries,
)
from us_market_ml.universe import (
    DynamicEligibilityConfig,
    apply_dynamic_universe_eligibility,
    min_symbol_count_per_date,
    select_fold_train_feature_columns,
    symbol_count,
)

DEFAULT_DATASET_GLOB = (
    "/Users/casper/Projects/Research/Market Data/data/research/"
    "etf_daily_prices_extended_tiingo_split/**/*.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps/dynamic_extended_history_study")
DEFAULT_RUN_NAME = "dynamic_extended_history_study"
DEFAULT_DYNAMIC_START = "2013-01-01"
DEFAULT_FULL_CONTROL_START = "2016-01-01"
DEFAULT_DATA_END = "2026-05-15"

TARGET_COL = "excess_return_vs_spy_20d"
FEATURE_SET_VERSION = "default_features_v1"
LABEL_HORIZON_DAYS = 20
MIN_HISTORY_ROWS = 252
MIN_HOLDING_DAYS = 10
SWITCH_MARGIN = 0.002
RANDOM_STATE = 42

PREFLIGHT_CSV_NAME = "dynamic_extended_history_preflight.csv"
PREFLIGHT_MD_NAME = "dynamic_extended_history_preflight.md"
SKIPPED_KEYS_SUFFIX = "_skipped_keys.txt"

PREFLIGHT_COLUMNS = [
    "run_key",
    "universe",
    "fold_name",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "model_type",
    "top_k",
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
class DynamicExtendedHistoryPreflight:
    """Diagnostic outputs generated before a dynamic extended-history run."""

    csv_path: Path
    markdown_path: Path
    skipped_keys_path: Path
    total_config_count: int
    valid_config_count: int
    skipped_config_count: int


def run_dynamic_extended_history_study_from_args(argv: list[str] | None = None) -> int:
    """Run or dry-run the dynamic extended-history study."""

    args = _parse_args(argv)
    folds = build_dynamic_extended_history_folds(
        data_start=args.data_start,
        data_end=args.data_end,
    )
    if not folds:
        raise SystemExit("No dynamic extended-history folds were generated.")

    config = build_dynamic_extended_history_sweep_config(
        market_dataset_glob=args.market_dataset_glob,
        folds=folds,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
        study_mode=args.study_mode,
        min_history_rows=args.min_history_rows,
        min_feature_non_null_ratio=args.min_feature_non_null_ratio,
    )
    preflight = preflight_dynamic_extended_history(config)

    _print_plan(config=config, preflight=preflight, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    run_sweep(config)
    return 0


def build_dynamic_extended_history_folds(
    *,
    data_start: str = DEFAULT_DYNAMIC_START,
    data_end: str = DEFAULT_DATA_END,
) -> list[SweepFoldConfig]:
    """Build annual sliding folds for the 2013+ dynamic study."""

    folds = make_sliding_window_folds_for_train_windows(
        data_start=data_start,
        data_end=data_end,
        train_window_years_values=(5, 10),
        validation_window_months=12,
        test_window_months=12,
        step_months=12,
        include_partial_final_test=True,
        min_partial_test_days=60,
    )
    return [_to_sweep_fold_config(fold, prefix="dynamic") for fold in folds]


def build_dynamic_extended_history_sweep_config(
    *,
    market_dataset_glob: str,
    folds: list[SweepFoldConfig],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
    study_mode: str = "dynamic_2013",
    min_history_rows: int = MIN_HISTORY_ROWS,
    min_feature_non_null_ratio: float = 0.0,
) -> SweepRunConfig:
    """Build the focused dynamic extended-history sweep config."""

    if study_mode == "dynamic_2013":
        universe_name = "dynamic_extended"
        universe_mode = "dynamic"
        feature_selection_scope = "fold_train"
    elif study_mode == "full_2016_control":
        universe_name = "extended_full_2016_control"
        universe_mode = "static"
        feature_selection_scope = "global"
    else:
        raise ValueError("study_mode must be 'dynamic_2013' or 'full_2016_control'.")

    return SweepRunConfig(
        universes=[SweepUniverseConfig(name=universe_name, data_path=market_dataset_glob)],
        folds=folds,
        models=[
            SweepModelConfig(
                model_family="baseline",
                model_type="linear_regression",
                random_state=RANDOM_STATE,
            ),
            SweepModelConfig(
                model_family="baseline",
                model_type="ridge",
                random_state=RANDOM_STATE,
            ),
        ],
        signals=[
            _signal(top_k=1),
            _signal(top_k=3),
            _signal(top_k=5),
        ],
        target_col=TARGET_COL,
        label_horizon_days=LABEL_HORIZON_DAYS,
        feature_set_versions=[FEATURE_SET_VERSION],
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
        transaction_cost_bps_values=[10.0, 20.0],
        output_dir=output_dir,
        run_name=run_name,
        resume=resume,
        max_workers=max_workers,
        skipped_keys_path=output_dir / f"{run_name}{SKIPPED_KEYS_SUFFIX}",
        skip_no_usable_training_rows=True,
        universe_mode=universe_mode,
        min_history_rows=min_history_rows,
        feature_selection_scope=feature_selection_scope,
        min_feature_non_null_ratio=min_feature_non_null_ratio,
    )


def preflight_dynamic_extended_history(
    config: SweepRunConfig,
) -> DynamicExtendedHistoryPreflight:
    """Write fold/config eligibility diagnostics and skipped invalid keys."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / PREFLIGHT_CSV_NAME
    markdown_path = output_dir / PREFLIGHT_MD_NAME
    skipped_keys_path = output_dir / f"{config.run_name}{SKIPPED_KEYS_SUFFIX}"

    rows = _build_preflight_rows(config)
    frame = pd.DataFrame(rows, columns=PREFLIGHT_COLUMNS)
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(_render_preflight_markdown(frame, config), encoding="utf-8")

    invalid_keys = set(frame.loc[~frame["valid_config"], "run_key"].astype(str).tolist())
    existing_skipped = _read_key_checkpoint(skipped_keys_path) if config.resume else set()
    skipped_keys = sorted(existing_skipped.union(invalid_keys))
    skipped_keys_path.write_text(
        "".join(f"{key}\n" for key in skipped_keys),
        encoding="utf-8",
    )

    return DynamicExtendedHistoryPreflight(
        csv_path=csv_path,
        markdown_path=markdown_path,
        skipped_keys_path=skipped_keys_path,
        total_config_count=int(len(frame)),
        valid_config_count=int(frame["valid_config"].sum()) if not frame.empty else 0,
        skipped_config_count=len(skipped_keys),
    )


def _build_preflight_rows(config: SweepRunConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for universe in config.universes:
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

            for fold_config in config.folds:
                fold = _to_walk_forward_fold(fold_config)
                train_df, validation_df, test_df = _split_for_preflight(dataset, fold, config)
                selected_feature_cols = _selected_preflight_features(
                    train_df=train_df,
                    candidate_feature_cols=candidate_feature_cols,
                    config=config,
                )
                train_eligible, validation_eligible, test_eligible = (
                    _eligible_preflight_splits(
                        train_df=train_df,
                        validation_df=validation_df,
                        test_df=test_df,
                        eligibility_history_df=dataset,
                        feature_cols=selected_feature_cols,
                        config=config,
                    )
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
                            invalid_reason = _invalid_reason(
                                selected_feature_count=fold_counts[
                                    "selected_feature_count"
                                ],
                                train_eligible_row_count=fold_counts[
                                    "train_eligible_row_count"
                                ],
                                validation_eligible_row_count=fold_counts[
                                    "validation_eligible_row_count"
                                ],
                                test_eligible_row_count=fold_counts[
                                    "test_eligible_row_count"
                                ],
                                train_symbol_count=fold_counts["train_symbol_count"],
                                validation_symbol_count=fold_counts[
                                    "validation_symbol_count"
                                ],
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
                                    "train_start": fold_config.train_start,
                                    "train_end": fold_config.train_end,
                                    "validation_start": fold_config.validation_start,
                                    "validation_end": fold_config.validation_end,
                                    "test_start": fold_config.test_start,
                                    "test_end": fold_config.test_end,
                                    "model_type": model_config.model_type,
                                    "top_k": signal_config.top_k,
                                    "transaction_cost_bps": transaction_cost_bps,
                                    **fold_counts,
                                    "supports_top_k": supports_top_k,
                                    "valid_config": invalid_reason == "",
                                    "invalid_reason": invalid_reason,
                                }
                            )
    return rows


def _selected_preflight_features(
    *,
    train_df: pd.DataFrame,
    candidate_feature_cols: list[str],
    config: SweepRunConfig,
) -> list[str]:
    if config.feature_selection_scope == "global":
        return list(candidate_feature_cols)
    return select_fold_train_feature_columns(
        train_df,
        candidate_feature_cols,
        min_non_null_ratio=config.min_feature_non_null_ratio,
    )


def _eligible_preflight_splits(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eligibility_history_df: pd.DataFrame,
    feature_cols: list[str],
    config: SweepRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not feature_cols:
        return (
            train_df.iloc[0:0].copy(),
            validation_df.iloc[0:0].copy(),
            test_df.iloc[0:0].copy(),
        )
    if config.universe_mode == "dynamic":
        eligibility_config = DynamicEligibilityConfig(min_history_rows=config.min_history_rows)
        return (
            apply_dynamic_universe_eligibility(
                train_df,
                feature_cols=feature_cols,
                target_col=config.target_col,
                require_target=True,
                config=eligibility_config,
                history_df=eligibility_history_df,
            ),
            apply_dynamic_universe_eligibility(
                validation_df,
                feature_cols=feature_cols,
                target_col=config.target_col,
                require_target=False,
                config=eligibility_config,
                history_df=eligibility_history_df,
            ),
            apply_dynamic_universe_eligibility(
                test_df,
                feature_cols=feature_cols,
                target_col=config.target_col,
                require_target=False,
                config=eligibility_config,
                history_df=eligibility_history_df,
            ),
        )

    return (
        train_df.dropna(subset=[*feature_cols, config.target_col]).copy(),
        validation_df.dropna(subset=feature_cols).copy(),
        test_df.dropna(subset=feature_cols).copy(),
    )


def _preflight_fold_counts(
    *,
    selected_feature_cols: list[str],
    train_eligible: pd.DataFrame,
    validation_eligible: pd.DataFrame,
    test_eligible: pd.DataFrame,
) -> dict[str, int]:
    return {
        "selected_feature_count": int(len(selected_feature_cols)),
        "train_eligible_row_count": int(len(train_eligible)),
        "validation_eligible_row_count": int(len(validation_eligible)),
        "test_eligible_row_count": int(len(test_eligible)),
        "train_symbol_count": symbol_count(train_eligible),
        "validation_symbol_count": symbol_count(validation_eligible),
        "test_symbol_count": symbol_count(test_eligible),
        "min_symbols_per_test_date": min_symbol_count_per_date(test_eligible),
    }


def _invalid_reason(
    *,
    selected_feature_count: int,
    train_eligible_row_count: int,
    validation_eligible_row_count: int,
    test_eligible_row_count: int,
    train_symbol_count: int,
    validation_symbol_count: int,
    test_symbol_count: int,
    supports_top_k: bool,
) -> str:
    reasons: list[str] = []
    if selected_feature_count < 1:
        reasons.append("no_selected_features")
    if train_eligible_row_count < 1:
        reasons.append("insufficient_train_eligible_rows")
    if test_eligible_row_count < 1:
        reasons.append("insufficient_test_eligible_rows")
    if train_symbol_count < 1:
        reasons.append("insufficient_train_symbols")
    if test_symbol_count < 1:
        reasons.append("insufficient_test_symbols")
    if not supports_top_k:
        reasons.append("unsupported_top_k")
    return ";".join(reasons)


def _split_for_preflight(
    dataset: pd.DataFrame,
    fold: WalkForwardFold,
    config: SweepRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.label_horizon_days <= 0:
        return fold.split(dataset)
    return purge_fold_boundaries(
        dataset,
        fold,
        PurgeConfig(label_horizon_days=config.label_horizon_days),
    )


def _render_preflight_markdown(frame: pd.DataFrame, config: SweepRunConfig) -> str:
    valid_count = int(frame["valid_config"].sum()) if not frame.empty else 0
    skipped_count = int((~frame["valid_config"]).sum()) if not frame.empty else 0
    fold_summary = _fold_summary(frame)
    invalid_preview = frame.loc[~frame["valid_config"]].head(20)

    lines = [
        "# Dynamic Extended-History Preflight",
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
        "",
        "## Totals",
        "",
        f"- Total configs: `{len(frame)}`",
        f"- Valid configs: `{valid_count}`",
        f"- Skipped configs: `{skipped_count}`",
        "",
        "## Fold Summary",
        "",
        _markdown_table(
            fold_summary,
            [
                "fold_name",
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


def _print_plan(
    *,
    config: SweepRunConfig,
    preflight: DynamicExtendedHistoryPreflight,
    dry_run: bool,
) -> None:
    output_dir = Path(config.output_dir)
    print("US Market ML dynamic extended-history study")
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
    print(f"fold count: {len(config.folds)}")
    print(f"model count: {len(config.models)}")
    print(f"signal count: {len(config.signals)}")
    print(f"transaction costs: {config.transaction_cost_bps_values}")
    print(f"estimated total configs: {preflight.total_config_count}")
    print(f"estimated valid configs: {preflight.valid_config_count}")
    print(f"estimated skipped configs: {preflight.skipped_config_count}")
    print(f"max workers: {config.max_workers}")
    print(f"resume: {config.resume}")
    print()
    print("Equivalent Python command:")
    print(_format_command(_equivalent_command(config)))


def _equivalent_command(config: SweepRunConfig) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_dynamic_extended_history_study",
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
    ]
    if config.universes[0].name == "extended_full_2016_control":
        command.extend(["--study-mode", "full_2016_control"])
    if not config.resume:
        command.append("--no-resume")
    return command


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local dynamic extended-history ETF study."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--market-dataset-glob",
        default=os.environ.get("MARKET_DATASET_GLOB", DEFAULT_DATASET_GLOB),
        help="Local parquet dataset glob. Defaults to MARKET_DATASET_GLOB or the local dataset.",
    )
    parser.add_argument(
        "--study-mode",
        choices=["dynamic_2013", "full_2016_control"],
        default="dynamic_2013",
        help="Run the 2013 dynamic main study by default; opt into the 2016 control explicitly.",
    )
    parser.add_argument("--data-start", default=None)
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
        "--min-feature-non-null-ratio",
        type=_parse_ratio,
        default=0.0,
        help="Fold-train feature availability threshold. Defaults to all-null exclusion only.",
    )
    args = parser.parse_args(argv)

    if args.data_start is None:
        args.data_start = (
            DEFAULT_FULL_CONTROL_START
            if args.study_mode == "full_2016_control"
            else DEFAULT_DYNAMIC_START
        )
    _parse_date(args.data_start, "--data-start")
    _parse_date(args.data_end, "--data-end")
    if pd.Timestamp(args.data_start) > pd.Timestamp(args.data_end):
        parser.error("--data-start must be earlier than or equal to --data-end.")
    return args


def _signal(*, top_k: int) -> SweepSignalConfig:
    return SweepSignalConfig(
        top_k=top_k,
        min_holding_days=MIN_HOLDING_DAYS,
        switch_margin=SWITCH_MARGIN,
        allocation_mode="equal_weight",
    )


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


def _to_sweep_fold_config(fold: WalkForwardFold, *, prefix: str) -> SweepFoldConfig:
    return SweepFoldConfig(
        name=f"{prefix}_{fold.name}",
        train_start=_date_string(fold.train_start),
        train_end=_date_string(fold.train_end),
        validation_start=_date_string(fold.validation_start),
        validation_end=_date_string(fold.validation_end),
        test_start=_date_string(fold.test_start),
        test_end=_date_string(fold.test_end),
    )


def _date_string(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _read_key_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


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
    raise SystemExit(run_dynamic_extended_history_study_from_args())
