"""Run structured local-cache-only E2E experiment sweeps."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from itertools import product
from pathlib import Path

from us_market_ml.data import DEFAULT_FEATURE_SET_VERSION, validate_feature_set_version
from us_market_ml.experiments import (
    SweepFoldConfig,
    SweepModelConfig,
    SweepRunConfig,
    SweepSignalConfig,
    SweepUniverseConfig,
    run_sweep,
    summarize_sweep_results,
)


def run_sweep_from_args(argv: list[str] | None = None) -> int:
    """Run the structured sweep CLI."""

    args = _parse_args(argv)
    models = _build_model_configs(args)
    config = SweepRunConfig(
        universes=args.universe,
        folds=args.fold,
        models=models,
        signals=_build_signal_configs(args),
        target_col=args.target_col,
        label_horizon_days=args.label_horizon_days,
        feature_set_versions=args.feature_set_versions,
        transaction_cost_bps_values=args.transaction_cost_bps,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
        max_workers=args.max_workers,
    )

    results = run_sweep(config)
    summary = summarize_sweep_results(results)
    output_dir = Path(config.output_dir)

    print("US Market ML structured sweep")
    print(f"results CSV: {output_dir / f'{config.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{config.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{config.run_name}_failed_keys.txt'}")
    print(f"max workers: {config.max_workers}")
    print(f"result rows: {len(results)}")
    print()
    if summary.empty:
        print("No completed test rows to summarize.")
    else:
        print(summary.to_string(index=False))

    return 0


def parse_name_path(value: str) -> SweepUniverseConfig:
    """Parse ``name:path`` into a sweep universe config."""

    name, separator, path = value.partition(":")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--universe must use format name:path")
    return SweepUniverseConfig(name=name.strip(), data_path=Path(path.strip()))


def parse_fold(value: str) -> SweepFoldConfig:
    """Parse a named walk-forward fold from CLI text."""

    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 7 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--fold must use format "
            "name:train_start:train_end:validation_start:validation_end:test_start:test_end"
        )
    return SweepFoldConfig(
        name=parts[0],
        train_start=parts[1],
        train_end=parts[2],
        validation_start=parts[3],
        validation_end=parts[4],
        test_start=parts[5],
        test_end=parts[6],
    )


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers."""

    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats."""

    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one float is required.")
    return values


def parse_hidden_units(value: str) -> tuple[int, ...]:
    """Parse comma-separated hidden layer sizes."""

    units = tuple(parse_int_list(value))
    if any(unit <= 0 for unit in units):
        raise argparse.ArgumentTypeError("hidden units must be positive integers.")
    return units


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured local-cache-only baseline and MLP E2E sweeps."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sweeps"))
    parser.add_argument("--universe", action="append", type=parse_name_path, required=True)
    parser.add_argument("--fold", action="append", type=parse_fold, required=True)
    parser.add_argument(
        "--model-family",
        choices=["baseline", "mlp", "both"],
        default="both",
    )
    parser.add_argument("--baseline-models", default="ridge,linear_regression")
    parser.add_argument("--top-k", type=parse_int_list, default=parse_int_list("1,2,3,5"))
    parser.add_argument(
        "--min-holding-days",
        type=parse_int_list,
        default=parse_int_list("1"),
    )
    parser.add_argument(
        "--switch-margins",
        type=parse_float_list,
        default=parse_float_list("0"),
    )
    parser.add_argument(
        "--allocation-modes",
        type=parse_allocation_modes,
        default=parse_allocation_modes("equal_weight"),
        help=(
            "Comma-separated allocation mode grid. Supported values: equal_weight, "
            "score_weighted, softmax_score_weighted, "
            "volatility_adjusted_score_weighted, capped_score_weighted."
        ),
    )
    parser.add_argument(
        "--score-temperatures",
        type=parse_float_list,
        default=parse_float_list("1.0"),
    )
    parser.add_argument(
        "--max-weights",
        type=parse_float_list,
        default=parse_float_list("1.0"),
    )
    parser.add_argument("--hidden-units", action="append", type=parse_hidden_units)
    parser.add_argument(
        "--learning-rates",
        type=parse_float_list,
        default=parse_float_list("0.001"),
    )
    parser.add_argument("--dropouts", type=parse_float_list, default=parse_float_list("0.1"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--scale-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize MLP targets using train-only target mean/std.",
    )
    parser.add_argument("--target-col", default="excess_return_vs_spy_20d")
    parser.add_argument("--label-horizon-days", type=int, default=0)
    parser.add_argument(
        "--feature-set-versions",
        type=parse_feature_set_versions,
        default=parse_feature_set_versions(DEFAULT_FEATURE_SET_VERSION),
        help=(
            "Comma-separated feature set grid. Supported values: "
            "default_features_v1,technical_features_v2,benchmark_gap_features_v1,"
            "benchmark_gap_features_v2."
        ),
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=parse_float_list,
        default=parse_float_list("0"),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-workers",
        type=parse_max_workers,
        default=parse_max_workers(os.environ.get("MAX_WORKERS", "1")),
        help=(
            "Number of local worker processes for independent sweep runs. "
            "Defaults to MAX_WORKERS from the environment, or 1."
        ),
    )
    args = parser.parse_args(argv)

    if args.hidden_units is None:
        args.hidden_units = [parse_hidden_units("128,64")]

    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.label_horizon_days < 0:
        parser.error("--label-horizon-days must be greater than or equal to 0.")
    if any(min_holding_days < 1 for min_holding_days in args.min_holding_days):
        parser.error("--min-holding-days values must be greater than or equal to 1.")
    if any(switch_margin < 0.0 for switch_margin in args.switch_margins):
        parser.error("--switch-margins values must be greater than or equal to 0.")
    if any(score_temperature <= 0.0 for score_temperature in args.score_temperatures):
        parser.error("--score-temperatures values must be greater than 0.")
    if any(
        max_weight <= 0.0 or max_weight > 1.0
        for max_weight in args.max_weights
    ):
        parser.error("--max-weights values must satisfy 0 < max_weight <= 1.")
    if any(transaction_cost_bps < 0.0 for transaction_cost_bps in args.transaction_cost_bps):
        parser.error("--transaction-cost-bps values must be greater than or equal to 0.")

    return args


def parse_max_workers(value: str) -> int:
    """Parse a positive local worker count."""

    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid worker count: {value!r}") from exc
    if workers < 1:
        raise argparse.ArgumentTypeError("--max-workers must be greater than or equal to 1.")
    return workers


def _build_model_configs(args: argparse.Namespace) -> list[SweepModelConfig]:
    models: list[SweepModelConfig] = []
    if args.model_family in {"baseline", "both"}:
        baseline_model_types = _parse_string_list(args.baseline_models)
        for model_type, seed in product(baseline_model_types, args.seeds):
            models.append(
                SweepModelConfig(
                    model_family="baseline",
                    model_type=model_type,
                    random_state=seed,
                )
            )

    if args.model_family in {"mlp", "both"}:
        for hidden_units, learning_rate, dropout, seed in product(
            args.hidden_units,
            args.learning_rates,
            args.dropouts,
            args.seeds,
        ):
            models.append(
                SweepModelConfig(
                    model_family="mlp",
                    model_type="mlp",
                    hidden_units=hidden_units,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    random_state=seed,
                    scale_target=args.scale_target,
                )
            )
    return models


def _build_signal_configs(args: argparse.Namespace) -> list[SweepSignalConfig]:
    signals: list[SweepSignalConfig] = []
    for (
        top_k,
        min_holding_days,
        switch_margin,
        allocation_mode,
        score_temperature,
        max_weight,
    ) in product(
        args.top_k,
        args.min_holding_days,
        args.switch_margins,
        args.allocation_modes,
        args.score_temperatures,
        args.max_weights,
    ):
        if _is_invalid_capped_allocation_grid_point(
            top_k=top_k,
            allocation_mode=allocation_mode,
            max_weight=max_weight,
        ):
            continue
        signals.append(
            SweepSignalConfig(
                top_k=top_k,
                min_holding_days=min_holding_days,
                switch_margin=switch_margin,
                allocation_mode=allocation_mode,
                score_temperature=score_temperature,
                max_weight=max_weight,
            )
        )
    return signals


def _is_invalid_capped_allocation_grid_point(
    *,
    top_k: int,
    allocation_mode: str,
    max_weight: float,
) -> bool:
    return allocation_mode == "capped_score_weighted" and top_k * max_weight < 1.0


def parse_feature_set_versions(value: str) -> list[str]:
    """Parse comma-separated feature set versions."""

    versions = _parse_string_list(value)
    for version in versions:
        try:
            validate_feature_set_version(version)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return versions


def parse_allocation_modes(value: str) -> list[str]:
    """Parse comma-separated allocation modes."""

    modes = _parse_string_list(value)
    supported_modes = {
        "equal_weight",
        "score_weighted",
        "softmax_score_weighted",
        "volatility_adjusted_score_weighted",
        "capped_score_weighted",
    }
    invalid_modes = [mode for mode in modes if mode not in supported_modes]
    if invalid_modes:
        supported = ", ".join(sorted(supported_modes))
        invalid = ", ".join(invalid_modes)
        raise argparse.ArgumentTypeError(
            f"Unsupported allocation mode(s): {invalid}. Supported values: {supported}."
        )
    return modes


def _parse_string_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


if __name__ == "__main__":
    raise SystemExit(run_sweep_from_args())
