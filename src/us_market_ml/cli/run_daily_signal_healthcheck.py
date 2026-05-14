"""Run a local healthcheck for daily model signal artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from us_market_ml.exports import run_model_signal_healthcheck


def run_daily_signal_healthcheck_from_args(argv: list[str] | None = None) -> int:
    """Load a local daily signal artifact and manifest, then print healthcheck status."""

    args = _parse_args(argv)
    artifact = _read_artifact(args.artifact_path)
    manifest = json.loads(args.manifest_path.read_text())
    summary, warnings = run_model_signal_healthcheck(
        artifact,
        manifest,
        model_version=args.model_version,
        expected_latest_date=args.expected_latest_date,
        min_symbol_count=args.min_symbol_count,
        require_top_signal=args.require_top_signal,
        artifact_path=str(args.artifact_path),
    )
    row = summary.iloc[0]

    print("US Market ML daily model signal healthcheck")
    print("research-only: not live trading advice and not an order/execution system")
    print(f"status: {row['status']}")
    print(f"model_version: {row['model_version']}")
    print(f"latest_signal_date: {row['latest_signal_date']}")
    print(f"row_count: {row['row_count']}")
    print(f"symbol_count: {row['symbol_count']}")
    print(f"top_signal_count: {row['top_signal_count']}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings:
        print(f"- {warning}")

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_path, index=False)
    if args.warnings_output_path is not None:
        args.warnings_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.warnings_output_path.write_text(
            "".join(f"{warning}\n" for warning in warnings),
            encoding="utf-8",
        )

    return 1 if row["status"] == "failed" else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a local daily model signal artifact and manifest."
    )
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--model-version")
    parser.add_argument("--expected-latest-date")
    parser.add_argument("--min-symbol-count", type=int)
    parser.add_argument(
        "--require-top-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warn when no artifact rows have signal_bucket == 'top'.",
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--warnings-output-path", type=Path)
    args = parser.parse_args(argv)

    if args.min_symbol_count is not None and args.min_symbol_count < 0:
        parser.error("--min-symbol-count must be greater than or equal to 0.")
    if args.model_version is not None:
        args.model_version = args.model_version.strip()
        if not args.model_version:
            parser.error("--model-version must be non-empty when provided.")
    return args


def _read_artifact(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported artifact suffix {path.suffix!r}; expected .csv or .parquet.")


if __name__ == "__main__":
    raise SystemExit(run_daily_signal_healthcheck_from_args())
