"""Run the focused local sliding-window champion stability study."""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.experiments import (
    SweepFoldConfig,
    SweepModelConfig,
    SweepRunConfig,
    SweepSignalConfig,
    SweepUniverseConfig,
    run_sweep,
    summarize_sweep_results,
)
from us_market_ml.splits import WalkForwardFold, make_sliding_window_folds_for_train_windows

CHAMPION_NAME = "linear_smooth10_margin002_v1"
DEFAULT_RUN_NAME = "sliding_window_champion_study"
DEFAULT_OUTPUT_DIR = Path("artifacts/sweeps")
DEFAULT_UNIVERSE = SweepUniverseConfig(
    name="full",
    data_path=Path("artifacts/cache/universe_full_2018_2026.parquet"),
)
DEFAULT_DATA_START = "2018-01-01"
DEFAULT_DATA_END = "2026-05-08"
TRAIN_WINDOW_YEARS = (2, 3, 5)
VALIDATION_WINDOW_MONTHS = 6
TEST_WINDOW_MONTHS = 3
RETRAIN_STEP_MONTHS = 3


def run_sliding_window_champion_study_from_args(argv: list[str] | None = None) -> int:
    """Run the local champion sliding-window study CLI."""

    args = _parse_args(argv)
    folds = build_sliding_window_champion_folds(
        data_start=args.data_start,
        data_end=args.data_end,
        include_partial_final_test=args.include_partial_final_test,
    )
    if not folds:
        raise SystemExit("No sliding-window folds were generated for the requested dates.")

    config = build_sliding_window_champion_config(
        folds=folds,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
    )

    if args.dry_run:
        _print_dry_run(config)
        return 0

    results = run_sweep(config)
    summary = summarize_sweep_results(results)
    output_dir = Path(config.output_dir)

    print("US Market ML sliding-window champion study")
    print(f"champion: {CHAMPION_NAME}")
    print(f"results CSV: {output_dir / f'{config.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{config.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{config.run_name}_failed_keys.txt'}")
    print(f"folds: {len(config.folds)}")
    print(f"max workers: {config.max_workers}")
    print(f"resume: {config.resume}")
    print(f"result rows: {len(results)}")
    print()
    if summary.empty:
        print("No completed test rows to summarize.")
    else:
        print(summary.to_string(index=False))
    return 0


def build_sliding_window_champion_folds(
    *,
    data_start: str,
    data_end: str,
    include_partial_final_test: bool = True,
) -> list[SweepFoldConfig]:
    """Return deterministic sweep folds for the fixed champion study design."""

    walk_forward_folds = make_sliding_window_folds_for_train_windows(
        data_start=data_start,
        data_end=data_end,
        train_window_years_values=TRAIN_WINDOW_YEARS,
        validation_window_months=VALIDATION_WINDOW_MONTHS,
        test_window_months=TEST_WINDOW_MONTHS,
        step_months=RETRAIN_STEP_MONTHS,
        include_partial_final_test=include_partial_final_test,
    )
    return [_to_sweep_fold_config(fold) for fold in walk_forward_folds]


def build_sliding_window_champion_config(
    *,
    folds: list[SweepFoldConfig],
    output_dir: Path,
    run_name: str,
    resume: bool,
    max_workers: int,
) -> SweepRunConfig:
    """Build the fixed champion ``SweepRunConfig`` used across all windows."""

    return SweepRunConfig(
        universes=[DEFAULT_UNIVERSE],
        folds=folds,
        models=[
            SweepModelConfig(
                model_family="baseline",
                model_type="linear_regression",
                random_state=42,
            )
        ],
        signals=[
            SweepSignalConfig(
                top_k=1,
                min_holding_days=10,
                switch_margin=0.002,
                allocation_mode="equal_weight",
                score_temperature=1.0,
                max_weight=1.0,
            )
        ],
        target_col="excess_return_vs_spy_20d",
        label_horizon_days=20,
        feature_set_versions=["default_features_v1"],
        min_holding_days=10,
        switch_margin=0.002,
        transaction_cost_bps_values=[10.0],
        output_dir=output_dir,
        run_name=run_name,
        resume=resume,
        max_workers=max_workers,
    )


def format_fold_arg(fold: SweepFoldConfig) -> str:
    """Format a fold as the sweep CLI ``--fold`` argument value."""

    return ":".join(
        [
            fold.name,
            fold.train_start,
            fold.train_end,
            fold.validation_start,
            fold.validation_end,
            fold.test_start,
            fold.test_end,
        ]
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only sliding-window retraining stability study for the "
            f"current champion ({CHAMPION_NAME})."
        )
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--data-end", default=DEFAULT_DATA_END)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from local checkpoint files by default.",
    )
    parser.add_argument(
        "--include-partial-final-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a final near-quarterly test window capped at --data-end.",
    )
    parser.add_argument(
        "--max-workers",
        type=_parse_max_workers,
        default=_parse_max_workers(os.environ.get("MAX_WORKERS", "4")),
        help="Number of local worker processes. Defaults to MAX_WORKERS or 4.",
    )
    args = parser.parse_args(argv)

    _parse_date(args.data_start, "--data-start")
    _parse_date(args.data_end, "--data-end")
    return args


def _print_dry_run(config: SweepRunConfig) -> None:
    print("US Market ML sliding-window champion study dry run")
    print("local-only: true")
    print(f"champion: {CHAMPION_NAME}")
    print(f"universe: {DEFAULT_UNIVERSE.name}:{DEFAULT_UNIVERSE.data_path}")
    print("model: baseline:linear_regression:seed42")
    print("signal: top_k=1,min_holding_days=10,switch_margin=0.002,equal_weight")
    print("target_col: excess_return_vs_spy_20d")
    print("label_horizon_days: 20")
    print("transaction_cost_bps: 10")
    print(f"run_name: {config.run_name}")
    print(f"output_dir: {config.output_dir}")
    print(f"resume: {config.resume}")
    print(f"max_workers: {config.max_workers}")
    print(f"folds: {len(config.folds)}")
    print()
    print("Generated folds:")
    for fold in config.folds:
        print(format_fold_arg(fold))
    print()
    print("Equivalent sweep command:")
    print(_format_command(_equivalent_sweep_command(config)))


def _equivalent_sweep_command(config: SweepRunConfig) -> list[str]:
    command = [
        "python",
        "-m",
        "us_market_ml.cli.run_sweep",
        "--run-name",
        config.run_name,
        "--output-dir",
        str(config.output_dir),
        "--universe",
        f"{DEFAULT_UNIVERSE.name}:{DEFAULT_UNIVERSE.data_path}",
        "--model-family",
        "baseline",
        "--baseline-models",
        "linear_regression",
        "--feature-set-versions",
        "default_features_v1",
        "--seeds",
        "42",
        "--top-k",
        "1",
        "--min-holding-days",
        "10",
        "--switch-margins",
        "0.002",
        "--allocation-modes",
        "equal_weight",
        "--score-temperatures",
        "1.0",
        "--max-weights",
        "1.0",
        "--transaction-cost-bps",
        "10",
        "--target-col",
        "excess_return_vs_spy_20d",
        "--label-horizon-days",
        "20",
        "--max-workers",
        str(config.max_workers),
    ]
    if not config.resume:
        command.append("--no-resume")
    for fold in config.folds:
        command.extend(["--fold", format_fold_arg(fold)])
    return command


def _format_command(command: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in command)


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


def _parse_date(value: str, parameter_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{parameter_name} must be a valid date.") from exc


def _parse_max_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid worker count: {value!r}") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("--max-workers must be greater than or equal to 1.")
    return workers


if __name__ == "__main__":
    raise SystemExit(run_sliding_window_champion_study_from_args())
