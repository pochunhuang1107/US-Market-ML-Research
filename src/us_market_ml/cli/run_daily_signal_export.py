"""Export latest local model signal artifacts for platform ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
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
from us_market_ml.exports import (
    build_model_signal_artifact,
    build_model_signal_manifest,
    validate_model_signal_artifact,
)
from us_market_ml.splits import WalkForwardFold


def run_daily_signal_export_from_args(argv: list[str] | None = None) -> int:
    """Run one local candidate and export latest model signal artifact + manifest."""

    args = _parse_args(argv)
    export_created_at = _utc_now_isoformat()
    market_data = read_local_market_data(args.data_path)
    max_data_date = pd.Timestamp(pd.to_datetime(market_data["date"]).max()).normalize()
    latest_date = _resolve_latest_date(args.latest_date, max_data_date)
    export_start = pd.Timestamp(args.lookback_start or latest_date).normalize()
    export_end = latest_date
    if export_start > export_end:
        raise ValueError("--lookback-start must be less than or equal to latest signal date.")

    warnings = _build_initial_warnings(latest_date, max_data_date)
    fold = _build_daily_signal_fold(args, export_start, export_end)
    result = _run_candidate_experiment(args.candidate, market_data, [fold], args)
    fold_result = result.fold_results[0]
    test_signals = fold_result.test_signals
    if test_signals.empty:
        raise ValueError("No test signal rows were produced for the export window.")

    artifact = build_model_signal_artifact(
        test_signals,
        args.model_version,
        feature_set_version=args.feature_set_version,
        label_horizon_days=args.label_horizon_days,
        created_at=export_created_at,
    )
    validate_model_signal_artifact(artifact)

    warnings.extend(_build_artifact_warnings(artifact))

    output_dir = Path(args.output_dir) / f"model_version={args.model_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"signals.{args.format}"
    manifest_path = output_dir / "model_signal_manifest.json"

    generated_row_count = len(artifact)
    if args.merge_existing and artifact_path.exists():
        existing_artifact = _read_artifact(artifact_path, args.format)
        written_artifact = _merge_artifacts(existing_artifact, artifact)
    else:
        written_artifact = artifact

    validate_model_signal_artifact(written_artifact)
    _write_artifact(written_artifact, artifact_path, args.format)

    manifest = build_model_signal_manifest(
        written_artifact,
        model_version=args.model_version,
        artifact_path=str(artifact_path),
        candidate_id=args.candidate_id,
        feature_set_version=args.feature_set_version,
        label_horizon_days=args.label_horizon_days,
        training_window=_window_dict(args.train_start, args.train_end),
        validation_window=_window_dict(args.validation_start, args.validation_end),
        tracking_window=_window_dict(export_start, export_end),
        status="ready",
        warnings=warnings,
        export_created_at=export_created_at,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    _print_daily_export_log(
        args=args,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        latest_date=latest_date,
        generated_row_count=generated_row_count,
        written_artifact=written_artifact,
        warnings=warnings,
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local candidate and export a runtime-safe latest model signal "
            "artifact plus manifest. Local research use only."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--train-start", type=pd.Timestamp, required=True)
    parser.add_argument("--train-end", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-start", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-end", type=pd.Timestamp, required=True)
    parser.add_argument("--latest-date", type=pd.Timestamp)
    parser.add_argument("--lookback-start", type=pd.Timestamp)
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
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument(
        "--merge-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge with an existing local artifact and deduplicate date/symbol/model_version.",
    )
    parser.add_argument("--transaction-cost-bps", default=0.0, type=float)
    parser.add_argument("--min-holding-days", default=1, type=int)
    parser.add_argument("--switch-margin", default=0.0, type=float)
    args = parser.parse_args(argv)

    if not args.model_version.strip():
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
    if args.transaction_cost_bps < 0.0:
        parser.error("--transaction-cost-bps must be greater than or equal to 0.")
    if args.min_holding_days < 1:
        parser.error("--min-holding-days must be greater than or equal to 1.")
    if args.switch_margin < 0.0:
        parser.error("--switch-margin must be greater than or equal to 0.")

    args.candidate_id = _candidate_id_from_args(argv, args.candidate)
    return args


def _candidate_id_from_args(argv: Sequence[str] | None, candidate: object) -> str:
    if argv is None:
        values = sys.argv[1:]
    else:
        values = list(argv)
    for index, value in enumerate(values[:-1]):
        if value == "--candidate":
            return values[index + 1]
    return getattr(candidate, "name", "")


def _resolve_latest_date(
    latest_date: pd.Timestamp | None,
    max_data_date: pd.Timestamp,
) -> pd.Timestamp:
    resolved = pd.Timestamp(latest_date).normalize() if latest_date is not None else max_data_date
    if resolved > max_data_date:
        raise ValueError(
            f"--latest-date {resolved.date()} is after max available data date "
            f"{max_data_date.date()}."
        )
    return resolved


def _build_daily_signal_fold(
    args: argparse.Namespace,
    export_start: pd.Timestamp,
    export_end: pd.Timestamp,
) -> WalkForwardFold:
    return WalkForwardFold(
        name="daily_signal_export",
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=export_start,
        test_end=export_end,
    )


def _build_initial_warnings(
    latest_date: pd.Timestamp,
    max_data_date: pd.Timestamp,
) -> list[str]:
    warnings: list[str] = []
    if latest_date < max_data_date:
        warnings.append(
            f"latest_date {latest_date.date()} is before max data date {max_data_date.date()}."
        )
    return warnings


def _build_artifact_warnings(artifact: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "signal_bucket" not in artifact or not (artifact["signal_bucket"] == "top").any():
        warnings.append("No rows have signal_bucket == 'top'.")
    symbol_count = int(artifact["symbol"].nunique()) if "symbol" in artifact else 0
    if symbol_count < 2:
        warnings.append(f"Symbol count is unexpectedly low: {symbol_count}.")
    return warnings


def _merge_artifacts(existing_artifact: pd.DataFrame, new_artifact: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([existing_artifact, new_artifact], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.drop_duplicates(
        subset=["date", "symbol", "model_version"],
        keep="last",
    )
    return merged.sort_values(
        by=["date", "rank", "symbol"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _read_artifact(path: Path, artifact_format: str) -> pd.DataFrame:
    if artifact_format == "csv":
        return pd.read_csv(path, parse_dates=["date"])
    return pd.read_parquet(path)


def _write_artifact(artifact: pd.DataFrame, path: Path, artifact_format: str) -> None:
    if artifact_format == "csv":
        artifact.to_csv(path, index=False)
        return
    artifact.to_parquet(path, index=False)


def _window_dict(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, str]:
    return {
        "start": pd.Timestamp(start).date().isoformat(),
        "end": pd.Timestamp(end).date().isoformat(),
    }


def _print_daily_export_log(
    *,
    args: argparse.Namespace,
    artifact_path: Path,
    manifest_path: Path,
    latest_date: pd.Timestamp,
    generated_row_count: int,
    written_artifact: pd.DataFrame,
    warnings: list[str],
) -> None:
    top_symbols = _top_symbols(written_artifact, latest_date)
    print("US Market ML daily model signal export")
    print("research-only: not live trading advice and not an order/execution system")
    print(f"model_version: {args.model_version}")
    print(f"output artifact path: {artifact_path}")
    print(f"manifest path: {manifest_path}")
    print(f"latest signal date: {latest_date.date()}")
    print(f"row count generated: {generated_row_count}")
    if args.merge_existing:
        print(f"merged row count: {len(written_artifact)}")
    print(f"symbol count: {written_artifact['symbol'].nunique()}")
    print(f"top selected symbols: {', '.join(top_symbols) if top_symbols else ''}")
    print("validation status: ready")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")


def _top_symbols(artifact: pd.DataFrame, latest_date: pd.Timestamp) -> list[str]:
    rows = artifact.loc[
        (pd.to_datetime(artifact["date"]) == latest_date)
        & (artifact.get("signal_bucket") == "top")
    ].copy()
    if rows.empty:
        return []
    symbols = (
        rows.sort_values(by=["rank", "score", "symbol"], ascending=[True, False, True])
        ["symbol"]
        .astype(str)
        .tolist()
    )
    return [str(symbol) for symbol in symbols]


def _utc_now_isoformat() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(run_daily_signal_export_from_args())
