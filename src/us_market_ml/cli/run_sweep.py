"""Run structured local-cache-only E2E experiment sweeps."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import product
from pathlib import Path

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
        signals=[SweepSignalConfig(top_k=top_k) for top_k in args.top_k],
        target_col=args.target_col,
        label_horizon_days=args.label_horizon_days,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
    )

    results = run_sweep(config)
    summary = summarize_sweep_results(results)
    output_dir = Path(config.output_dir)

    print("US Market ML structured sweep")
    print(f"results CSV: {output_dir / f'{config.run_name}_results.csv'}")
    print(f"completed checkpoint: {output_dir / f'{config.run_name}_completed_keys.txt'}")
    print(f"failed checkpoint: {output_dir / f'{config.run_name}_failed_keys.txt'}")
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
    parser.add_argument("--target-col", default="excess_return_vs_spy_20d")
    parser.add_argument("--label-horizon-days", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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

    return args


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
                )
            )
    return models


def _parse_string_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


if __name__ == "__main__":
    raise SystemExit(run_sweep_from_args())
