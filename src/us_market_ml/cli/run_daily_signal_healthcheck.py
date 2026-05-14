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
        max_signal_age_days=args.max_signal_age_days,
        reference_date=args.reference_date,
        stale_as_failure=args.stale_as_failure,
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
    print(f"reference_date: {row['reference_date']}")
    print(f"signal_age_days: {row['signal_age_days']}")
    print(f"max_signal_age_days: {row['max_signal_age_days']}")
    print(f"stale_signal: {row['stale_signal']}")
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
    parser.add_argument("--max-signal-age-days", type=int)
    parser.add_argument("--reference-date")
    stale_group = parser.add_mutually_exclusive_group()
    stale_group.add_argument(
        "--stale-as-failure",
        dest="stale_as_failure",
        action="store_true",
        default=False,
        help="Return failed status when latest_signal_date exceeds --max-signal-age-days.",
    )
    stale_group.add_argument(
        "--stale-as-warning",
        dest="stale_as_failure",
        action="store_false",
        help="Report stale latest_signal_date as warning status. This is the default.",
    )
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
    if args.max_signal_age_days is not None and args.max_signal_age_days < 0:
        parser.error("--max-signal-age-days must be greater than or equal to 0.")
    if args.reference_date is not None:
        args.reference_date = args.reference_date.strip()
        if not args.reference_date:
            parser.error("--reference-date must be non-empty when provided.")
        try:
            parsed_reference_date = pd.Timestamp(args.reference_date)
            if pd.isna(parsed_reference_date):
                raise ValueError
            args.reference_date = parsed_reference_date.date().isoformat()
        except (TypeError, ValueError):
            parser.error("--reference-date must be parseable as a date.")
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
