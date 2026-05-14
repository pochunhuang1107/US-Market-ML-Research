"""Export runtime-compatible point-in-time model signal artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.cli.run_candidate_comparison_report import (
    _run_candidate_experiment,
    parse_candidate_config,
)
from us_market_ml.data import (
    DEFAULT_FEATURE_SET_VERSION,
    read_local_market_data,
    validate_feature_set_version,
)
from us_market_ml.exports import build_model_signal_artifact, validate_model_signal_artifact
from us_market_ml.splits import WalkForwardFold


def run_model_signal_export_from_args(argv: list[str] | None = None) -> int:
    """Run one selected local candidate and export platform model signals."""

    args = _parse_args(argv)
    market_data = read_local_market_data(args.data_path)
    fold = _build_export_fold(args)
    result = _run_candidate_experiment(args.candidate, market_data, [fold], args)
    test_signals = result.fold_results[0].test_signals
    if test_signals.empty:
        raise ValueError("No test signal rows were produced for the export window.")

    artifact = build_model_signal_artifact(
        test_signals,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        label_horizon_days=args.label_horizon_days,
        created_at=args.created_at,
    )
    validate_model_signal_artifact(artifact)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = _resolve_output_format(output_path, args.format)
    if output_format == "csv":
        artifact.to_csv(output_path, index=False)
    else:
        artifact.to_parquet(output_path, index=False)

    dates = pd.to_datetime(artifact["date"])
    print("US Market ML model signal export")
    print("research-only: point-in-time model signals, not advice or execution instructions")
    print(f"output path: {output_path}")
    print(f"model_version: {args.model_version}")
    print(f"row count: {len(artifact)}")
    print(f"date range: {dates.min().date()} to {dates.max().date()}")

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local candidate over an export window and write point-in-time "
            "model signal artifacts for the simulation platform."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=parse_candidate_config,
        required=True,
        help=(
            "Candidate spec: "
            "baseline:name:model_type:top_k:random_state[:min_holding_days:switch_margin] "
            "or mlp:name:top_k:hidden_units:learning_rate:dropout:random_state"
            "[:min_holding_days:switch_margin]."
        ),
    )
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--feature-set-version", default=DEFAULT_FEATURE_SET_VERSION)
    parser.add_argument("--created-at")
    parser.add_argument("--train-start", type=pd.Timestamp, required=True)
    parser.add_argument("--train-end", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-start", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-end", type=pd.Timestamp, required=True)
    parser.add_argument("--export-start", type=pd.Timestamp, required=True)
    parser.add_argument("--export-end", type=pd.Timestamp, required=True)
    parser.add_argument("--target-col", default="excess_return_vs_spy_20d")
    parser.add_argument(
        "--scale-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize MLP targets using train-only target mean/std.",
    )
    parser.add_argument("--max-epochs", default=120, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--label-horizon-days", default=20, type=int)
    parser.add_argument("--format", choices=["csv", "parquet"])
    args = parser.parse_args(argv)

    args.model_version = args.model_version.strip()
    if not args.model_version:
        parser.error("--model-version must be non-empty.")
    args.feature_set_version = args.feature_set_version.strip()
    if not args.feature_set_version:
        parser.error("--feature-set-version must be non-empty.")
    try:
        validate_feature_set_version(args.feature_set_version)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.label_horizon_days < 0:
        parser.error("--label-horizon-days must be greater than or equal to 0.")

    # Shared candidate runner options that are intentionally fixed for export v1.
    args.transaction_cost_bps = 0.0
    args.min_holding_days = 1
    args.switch_margin = 0.0

    return args


def _build_export_fold(args: argparse.Namespace) -> WalkForwardFold:
    return WalkForwardFold(
        name="model_signal_export",
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.export_start,
        test_end=args.export_end,
    )


def _resolve_output_format(output_path: Path, requested_format: str | None) -> str:
    if requested_format is not None:
        return requested_format
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    return "parquet"


__all__ = ["run_model_signal_export_from_args"]
